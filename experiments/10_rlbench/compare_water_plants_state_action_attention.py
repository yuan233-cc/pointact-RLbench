"""Compare state/action-token attention and causal influence between checkpoints.

The script runs one checkpoint at a time on a shared archive of paired
water-plants frame-2/frame-3 observations.  It reconstructs the two token-query
attention rows in every PTV3 encoder block and optionally evaluates state/point
counterfactuals.  All reported attention values are diagnostic: hooks do not
modify the model forward pass.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from pointact.model.ptv3.concerto.utils import offset2bincount
from scripts.run_server import MODEL_MAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-kind", choices=("complete", "incomplete"), required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Also run frame/state/point swaps and remove the state token.",
    )
    parser.add_argument(
        "--permuted-state-only",
        action="store_true",
        help=(
            "Run only an in-distribution intervention that pairs each pour cloud "
            "with the next episode's pour state."
        ),
    )
    return parser.parse_args()


def module_sort_key(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"enc\.enc(\d+)\.block(\d+)\.attn", name)
    if match is None:
        raise ValueError(f"Unexpected attention module name: {name}")
    return int(match.group(1)), int(match.group(2))


def normalized_entropy(weights: torch.Tensor) -> float:
    weights = weights.float()
    total = weights.sum()
    if len(weights) <= 1 or total <= 0:
        return 0.0
    probability = weights / total
    entropy = -(probability * probability.clamp_min(1e-15).log()).sum()
    return float(entropy / math.log(len(weights)))


def top_fraction_mass(weights: torch.Tensor, fraction: float) -> float:
    weights = weights.float()
    total = weights.sum()
    if total <= 0:
        return 0.0
    count = max(1, math.ceil(len(weights) * fraction))
    return float(torch.topk(weights, count).values.sum() / total)


def distribution_similarity(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    left = left.float() / left.float().sum().clamp_min(1e-15)
    right = right.float() / right.float().sum().clamp_min(1e-15)
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=0)
    middle = (left + right) / 2
    js = 0.5 * (
        (left * (left.clamp_min(1e-15) / middle.clamp_min(1e-15)).log()).sum()
        + (right * (right.clamp_min(1e-15) / middle.clamp_min(1e-15)).log()).sum()
    )
    return float(cosine), float(js / math.log(2))


class AllLayerAttentionProbe:
    """Reconstruct state/action query rows for every PTV3 encoder block."""

    def __init__(self, model) -> None:
        root = model.ptv3_model.ptv3_model
        modules = [
            (name, module)
            for name, module in root.named_modules()
            if name.startswith("enc.enc")
            and ".block" in name
            and name.endswith(".attn")
            and module.__class__.__name__ == "SerializedAttentionWithAction"
        ]
        if not modules:
            raise RuntimeError("No SerializedAttentionWithAction modules found")
        modules.sort(key=lambda item: module_sort_key(item[0]))
        self.module_names = [name for name, _module in modules]
        self.enabled = False
        self.values: list[dict[str, float | int | str]] = []
        for name, module in modules:
            module.register_forward_pre_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        stage, block = module_sort_key(name)

        @torch.no_grad()
        def hook(module, inputs) -> None:
            if not self.enabled:
                return
            point = inputs[0]
            counts = offset2bincount(point.offset)
            if len(counts) != 1:
                raise ValueError("Attention probe requires batch size 1")
            if point.action_feat.shape[1] != 2:
                raise ValueError(
                    f"Expected state and action tokens, got {point.action_feat.shape[1]}"
                )
            if module.enable_rpe:
                raise ValueError("RPE reconstruction is not implemented")

            pad, _unpad, cu_seqlens = module.get_padding_and_inverse(point)
            patch_lengths = torch.diff(cu_seqlens).tolist()
            order = point.serialized_order[module.order_index][pad]
            point_qkv = module.qkv(point.feat)[order]
            token_qkv = module.qkv(point.action_feat)[0]
            num_points = len(point.feat)
            num_heads = module.num_heads
            head_dim = module.channels // num_heads
            point_weights = torch.zeros(
                (2, num_points), dtype=torch.float32, device=point.feat.device
            )
            masses = torch.zeros((2, 3), device=point.feat.device)
            num_patches = len(patch_lengths)

            for point_chunk, index_chunk in zip(
                torch.split(point_qkv, patch_lengths, dim=0),
                torch.split(order, patch_lengths, dim=0),
            ):
                full_qkv = torch.cat([token_qkv, point_chunk], dim=0)
                full_qkv = full_qkv.reshape(-1, 3, num_heads, head_dim).to(torch.float16)
                query, key, _value = full_qkv.unbind(dim=1)
                logits = torch.einsum("qhd,khd->hqk", query, key).float() * module.scale
                probability = torch.softmax(logits, dim=-1)
                for query_index in range(2):
                    current = probability[:, query_index]
                    point_probability = current[:, 2:].mean(dim=0)
                    point_weights[query_index].index_add_(
                        0, index_chunk, point_probability / num_patches
                    )
                    masses[query_index, 0] += current[:, 0].mean() / num_patches
                    masses[query_index, 1] += current[:, 1].mean() / num_patches
                    masses[query_index, 2] += (
                        current[:, 2:].sum(-1).mean() / num_patches
                    )

            state_weights, action_weights = point_weights
            point_cosine, point_js = distribution_similarity(
                state_weights, action_weights
            )
            token_cosine = torch.nn.functional.cosine_similarity(
                point.action_feat[0, 0].float(),
                point.action_feat[0, 1].float(),
                dim=0,
            )
            self.values.append(
                {
                    "layer": name,
                    "stage": stage,
                    "block": block,
                    "num_points": num_points,
                    "num_patches": num_patches,
                    "state_to_state_mass": float(masses[0, 0]),
                    "state_to_action_mass": float(masses[0, 1]),
                    "state_to_point_mass": float(masses[0, 2]),
                    "state_to_point_entropy": normalized_entropy(state_weights),
                    "state_to_point_top1pct": top_fraction_mass(state_weights, 0.01),
                    "action_to_state_mass": float(masses[1, 0]),
                    "action_to_action_mass": float(masses[1, 1]),
                    "action_to_point_mass": float(masses[1, 2]),
                    "action_to_point_entropy": normalized_entropy(action_weights),
                    "action_to_point_top1pct": top_fraction_mass(action_weights, 0.01),
                    "state_action_point_cosine": point_cosine,
                    "state_action_point_js": point_js,
                    "state_action_feature_cosine": float(token_cosine),
                }
            )

        return hook


def euler_bins(euler: np.ndarray, resolution_deg: int = 5) -> np.ndarray:
    values = np.mod(euler, 2 * np.pi)
    bins = np.rint(values / np.deg2rad(resolution_deg)).astype(np.int64)
    return bins % (360 // resolution_deg)


def rotation_error_deg(pred_bins: np.ndarray, target_euler: np.ndarray) -> float:
    pred_euler = np.deg2rad(pred_bins * 5)
    pred = Rotation.from_euler("xyz", pred_euler)
    target = Rotation.from_euler("xyz", target_euler)
    return float(np.rad2deg((pred * target.inv()).magnitude()))


def margin(result: dict, pour_bins: np.ndarray, upright_bins: np.ndarray, axes: list[int]) -> float:
    logits = np.asarray(result.pop("rotation_logits"))
    values = [
        logits[pour_bins[axis], axis] - logits[upright_bins[axis], axis]
        for axis in axes
    ]
    return float(np.mean(values))


class DiagnosticRunner:
    def __init__(self, checkpoint: Path) -> None:
        config = json.loads((checkpoint / "config.json").read_text())
        model_class, processor_class = MODEL_MAP[config["architectures"][0]]
        self.model = model_class.from_pretrained(
            checkpoint, device_map={"": "cuda"}, local_files_only=True
        ).eval()
        self.processor = processor_class.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.repo_id = next(iter(self.processor.robot_config["state_action_norm"]))
        self.probe = AllLayerAttentionProbe(self.model)
        self.latest_head: dict[str, torch.Tensor] = {}

        def head_hook(_module, _inputs, output) -> None:
            self.latest_head = {
                "rotation_logits": output[1].detach().float().cpu(),
                "open_logit": output[2].detach().float().cpu(),
            }

        self.model.action_head.register_forward_hook(head_hook)

    def prepare(self, state: np.ndarray, points: np.ndarray) -> dict[str, torch.Tensor]:
        batch = {
            "observation.state": [state.astype(np.float32)],
            "observation.points": [points.astype(np.float32)],
            "task": ["water plant"],
            "repo_id": [self.repo_id],
        }
        messages, states, clouds, _centers, _repo_ids = self.processor._prepare_robot_inputs(batch)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"states": states},
        ).to(self.model.device)
        inputs["input_id_lens"] = inputs["attention_mask"].sum(dim=1).long()
        inputs["points"] = torch.cat(clouds, 0).to(self.model.device)
        inputs["npoints_in_batch"] = torch.tensor(
            [len(cloud) for cloud in clouds], dtype=torch.long, device=self.model.device
        )
        inputs["attention_mask"] = inputs["attention_mask"].bool()
        return inputs

    @torch.no_grad()
    def run(
        self,
        state: np.ndarray,
        points: np.ndarray,
        *,
        remove_state_token: bool = False,
        capture_attention: bool = False,
    ) -> dict:
        inputs = self.prepare(state, points)
        old_use_state = bool(self.model.config.use_robot_state)
        self.probe.enabled = capture_attention and not remove_state_token
        self.probe.values = []
        if remove_state_token:
            self.model.config.use_robot_state = False
        try:
            actions, _ = self.model.sample_actions(**inputs)
        finally:
            self.model.config.use_robot_state = old_use_state
            self.probe.enabled = False

        rotation_logits = self.latest_head["rotation_logits"][0, 0].numpy()
        open_logit = float(self.latest_head["open_logit"][0, 0])
        pred_bins = rotation_logits.argmax(axis=0)
        return {
            "pred_rotation_bins": pred_bins.tolist(),
            "pred_rotation_degrees": (pred_bins * 5).tolist(),
            "open_logit": open_logit,
            "open_probability": float(torch.sigmoid(torch.tensor(open_logit))),
            "rotation_logits": rotation_logits.tolist(),
            "action": actions[0, 0].detach().float().cpu().tolist(),
            "attention": list(self.probe.values),
        }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.samples, allow_pickle=True)
    episodes = np.asarray(archive["episode"], dtype=np.int64)
    states = np.asarray(archive["state"], dtype=np.float32)
    actions = np.asarray(archive["action"], dtype=np.float32)
    frames = np.asarray(archive["frame"], dtype=np.int64)
    runner = DiagnosticRunner(args.checkpoint)

    episode_ids = sorted(set(episodes.tolist()))
    if args.limit > 0:
        episode_ids = episode_ids[: args.limit]
    current_index_by_episode = {
        episode: int(
            np.flatnonzero((episodes == episode) & (frames == 3))[0]
        )
        for episode in episode_ids
    }

    with args.output.open("x") as stream:
        for order, episode in enumerate(episode_ids):
            indices = np.flatnonzero(episodes == episode)
            by_frame = {int(frames[index]): int(index) for index in indices}
            previous_index, current_index = by_frame[2], by_frame[3]
            previous_state, current_state = states[previous_index], states[current_index]
            point_key = f"{args.cloud_kind}_points"
            previous_points = np.asarray(archive[point_key][previous_index], dtype=np.float32)
            current_points = np.asarray(archive[point_key][current_index], dtype=np.float32)

            if args.permuted_state_only:
                next_episode = episode_ids[(order + 1) % len(episode_ids)]
                other_state = states[current_index_by_episode[next_episode]]
                full = runner.run(other_state, current_points)
                variants = {"other_pour_state_current_points": full}
            else:
                full = runner.run(current_state, current_points, capture_attention=True)
                variants = {"pour_full": full}
            if args.causal and not args.permuted_state_only:
                variants.update(
                    {
                        "previous_full": runner.run(previous_state, previous_points),
                        "previous_state_current_points": runner.run(
                            previous_state, current_points
                        ),
                        "current_state_previous_points": runner.run(
                            current_state, previous_points
                        ),
                        "pour_no_state_token": runner.run(
                            current_state, current_points, remove_state_token=True
                        ),
                    }
                )

            upright_bins = euler_bins(actions[previous_index, 3:6])
            pour_bins = euler_bins(actions[current_index, 3:6])
            changed_axes = np.flatnonzero(pour_bins != upright_bins).tolist() or [0, 1, 2]
            for value in variants.values():
                value["pour_vs_upright_margin"] = margin(
                    value, pour_bins, upright_bins, changed_axes
                )
                value["target_rotation_error_deg"] = rotation_error_deg(
                    np.asarray(value["pred_rotation_bins"]), actions[current_index, 3:6]
                )

            record = {
                "checkpoint": args.checkpoint_label,
                "checkpoint_path": str(args.checkpoint),
                "cloud_kind": args.cloud_kind,
                "episode": int(episode),
                "changed_axes": changed_axes,
                "upright_bins": upright_bins.tolist(),
                "pour_bins": pour_bins.tolist(),
                "permuted_state_source_episode": (
                    int(episode_ids[(order + 1) % len(episode_ids)])
                    if args.permuted_state_only
                    else None
                ),
                "variants": variants,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(
                f"{order + 1}/{len(episode_ids)} episode={episode} "
                f"margin={full['pour_vs_upright_margin']:.3f} "
                f"open={full['open_probability']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
