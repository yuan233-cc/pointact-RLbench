"""Measure state and point-cloud influence on the water-plants pour decision.

The input archive contains paired pre-pour (frame 2) and pour (frame 3)
observations from the RLBench training demonstrations.  For each pair, this
script compares the normal pour-frame prediction with three counterfactuals:

* previous robot state with the current point cloud;
* current robot state with the previous point cloud;
* the current observation with the state token removed.

It also reconstructs the final PTV3 block's state/action-query attention mass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts.run_server import MODEL_MAP
from pointact.model.ptv3.concerto.utils import offset2bincount


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cloud-kind", choices=("complete", "incomplete"), required=True
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def euler_bins(euler: np.ndarray, resolution_deg: int = 5) -> np.ndarray:
    values = np.mod(euler, 2 * np.pi)
    bins = np.rint(values / np.deg2rad(resolution_deg)).astype(np.int64)
    return bins % (360 // resolution_deg)


def rotation_error_deg(pred_bins: np.ndarray, target_euler: np.ndarray) -> float:
    pred_euler = np.deg2rad(pred_bins * 5)
    pred = Rotation.from_euler("xyz", pred_euler)
    target = Rotation.from_euler("xyz", target_euler)
    return float(np.rad2deg((pred * target.inv()).magnitude()))


class FinalAttentionProbe:
    """Reconstruct state/action query attention in the final PTV3 block."""

    def __init__(self, model) -> None:
        modules = {
            name: module
            for name, module in model.ptv3_model.ptv3_model.named_modules()
            if name.startswith("enc.enc")
            and ".block" in name
            and name.endswith(".attn")
            and module.__class__.__name__ == "SerializedAttentionWithAction"
        }
        if not modules:
            raise RuntimeError("No SerializedAttentionWithAction modules found")
        self.module_name = sorted(
            modules,
            key=lambda name: tuple(
                int(piece)
                for piece in name.replace("enc.enc", "").replace(".block", ".").split(".")[:2]
            ),
        )[-1]
        self.enabled = False
        self.value: dict[str, float] = {}
        modules[self.module_name].register_forward_pre_hook(self._hook)

    @staticmethod
    def _entropy(weights: torch.Tensor) -> float:
        weights = weights.float()
        total = weights.sum()
        if len(weights) <= 1 or total <= 0:
            return 0.0
        probability = weights / total
        entropy = -(probability * probability.clamp_min(1e-15).log()).sum()
        return float(entropy / math.log(len(weights)))

    @torch.no_grad()
    def _hook(self, module, inputs) -> None:
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

        pad, _unpad, cu_seqlens = module.get_padding_and_inverse(point)
        patch_lengths = torch.diff(cu_seqlens).tolist()
        order = point.serialized_order[module.order_index][pad]
        point_qkv = module.qkv(point.feat)[order]
        token_qkv = module.qkv(point.action_feat)[0]
        num_points = len(point.feat)
        num_heads = module.num_heads
        head_dim = module.channels // num_heads
        state_weights = torch.zeros(num_points, device=point.feat.device)
        action_weights = torch.zeros(num_points, device=point.feat.device)
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
            for query_index, weights in enumerate((state_weights, action_weights)):
                point_probability = probability[:, query_index, 2:].mean(dim=0)
                weights.index_add_(0, index_chunk, point_probability / num_patches)
                masses[query_index, 0] += probability[:, query_index, 0].mean() / num_patches
                masses[query_index, 1] += probability[:, query_index, 1].mean() / num_patches
                masses[query_index, 2] += probability[:, query_index, 2:].sum(-1).mean() / num_patches

        self.value = {
            "layer": self.module_name,
            "state_to_state_mass": float(masses[0, 0]),
            "state_to_action_mass": float(masses[0, 1]),
            "state_to_point_mass": float(masses[0, 2]),
            "state_to_point_entropy": self._entropy(state_weights),
            "action_to_state_mass": float(masses[1, 0]),
            "action_to_action_mass": float(masses[1, 1]),
            "action_to_point_mass": float(masses[1, 2]),
            "action_to_point_entropy": self._entropy(action_weights),
        }


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
        self.probe = FinalAttentionProbe(self.model)
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
        self.probe.value = {}
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
        result = {
            "pred_rotation_bins": pred_bins.tolist(),
            "pred_rotation_degrees": (pred_bins * 5).tolist(),
            "open_probability": float(torch.sigmoid(torch.tensor(open_logit))),
            "rotation_logits": rotation_logits.tolist(),
            "action": actions[0, 0].detach().float().cpu().tolist(),
        }
        result.update(self.probe.value)
        return result


def select_points(archive, kind: str, index: int) -> np.ndarray:
    key = f"{kind}_points"
    return np.asarray(archive[key][index], dtype=np.float32)


def margin(result: dict, pour_bins: np.ndarray, upright_bins: np.ndarray, axes: list[int]) -> float:
    logits = np.asarray(result["rotation_logits"])
    values = [logits[pour_bins[axis], axis] - logits[upright_bins[axis], axis] for axis in axes]
    return float(np.mean(values))


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

    with args.output.open("x") as stream:
        for order, episode in enumerate(episode_ids):
            indices = np.flatnonzero(episodes == episode)
            by_frame = {int(frames[index]): int(index) for index in indices}
            if 2 not in by_frame or 3 not in by_frame:
                raise ValueError(f"Episode {episode} lacks frame 2 or 3")
            previous_index = by_frame[2]
            current_index = by_frame[3]
            previous_state = states[previous_index]
            current_state = states[current_index]
            previous_points = select_points(archive, args.cloud_kind, previous_index)
            current_points = select_points(archive, args.cloud_kind, current_index)

            previous = runner.run(previous_state, previous_points)
            full = runner.run(
                current_state, current_points, capture_attention=True
            )
            previous_state_current_points = runner.run(previous_state, current_points)
            current_state_previous_points = runner.run(current_state, previous_points)
            no_state = runner.run(
                current_state, current_points, remove_state_token=True
            )

            upright_bins = euler_bins(actions[previous_index, 3:6])
            pour_bins = euler_bins(actions[current_index, 3:6])
            changed_axes = np.flatnonzero(pour_bins != upright_bins).tolist()
            if not changed_axes:
                changed_axes = [0, 1, 2]
            variants = {
                "previous_full": previous,
                "pour_full": full,
                "previous_state_current_points": previous_state_current_points,
                "current_state_previous_points": current_state_previous_points,
                "pour_no_state_token": no_state,
            }
            for value in variants.values():
                value["pour_vs_upright_margin"] = margin(
                    value, pour_bins, upright_bins, changed_axes
                )
                value["target_rotation_error_deg"] = rotation_error_deg(
                    np.asarray(value["pred_rotation_bins"]), actions[current_index, 3:6]
                )
                del value["rotation_logits"]

            record = {
                "checkpoint": str(args.checkpoint),
                "cloud_kind": args.cloud_kind,
                "episode": int(episode),
                "changed_axes": changed_axes,
                "upright_bins": upright_bins.tolist(),
                "pour_bins": pour_bins.tolist(),
                "variants": variants,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(
                order + 1,
                episode,
                "margin",
                round(full["pour_vs_upright_margin"], 3),
                "no_state",
                round(no_state["pour_vs_upright_margin"], 3),
                flush=True,
            )


if __name__ == "__main__":
    main()
