"""Local PointACT server with training-matched realistic-failures-v2 input."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import tyro

from create_rlbench_10task_realistic_failure_dataset import apply_corruption
from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed
from scripts.run_server import Policy


EXPECTED_REPO_ID = "hybridvla_10tasks_train_keysteps_realistic_failures_v2"


@dataclasses.dataclass
class Args:
    pretrained_path: str
    seed: int = 7
    host: str = "127.0.0.1"
    port: int = 15556
    num_denoise_steps: int = 10
    save_data: bool = False
    save_dir: str = "/tmp/pointact_realistic_failures_v2_inputs"
    audit_path: str = ""


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_realistic_failure(
    self, mini_batch, repo_id, workspace, *, remove_arm=False, voxel_size=0.01
):
    if repo_id != EXPECTED_REPO_ID:
        raise ValueError(f"Expected repo_id={EXPECTED_REPO_ID!r}, got {repo_id!r}")
    if remove_arm:
        raise ValueError("remove_arm must be false: robot holes are part of v2 corruption")
    for key in (
        "observation.points.front",
        "observation.images.front_image",
        "realistic_failure_task_index",
        "realistic_failure_source_episode",
        "realistic_failure_base_seed",
    ):
        if key not in mini_batch:
            raise ValueError(f"Missing training-matched input field: {key}")

    xyz = _numpy(mini_batch["observation.points.front"]).astype(np.float32, copy=False)
    rgb = _numpy(mini_batch["observation.images.front_image"])
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or rgb.shape != xyz.shape:
        raise ValueError(f"Expected matching HxWx3 xyz/rgb, got {xyz.shape} and {rgb.shape}")
    rgb = rgb.astype(np.float32, copy=False)
    if rgb.size and rgb.max() > 1.0:
        rgb = rgb / 255.0
    clean = np.concatenate((xyz, rgb), axis=-1).reshape(-1, 6)
    clean = self._filter_points_by_workspace(clean, workspace)
    clean = self._voxel_downsample_point_cloud(clean, voxel_size=voxel_size)

    result = apply_corruption(
        clean,
        task_index=int(mini_batch["realistic_failure_task_index"]),
        source_episode=int(mini_batch["realistic_failure_source_episode"]),
        base_seed=int(mini_batch["realistic_failure_base_seed"]),
    )
    point_cloud = result.cloud.copy()
    point_cloud[:, 3:6] = point_cloud[:, 3:6] * 2.0 - 1.0
    point_cloud = self._subsample_point_cloud(
        point_cloud, self.robot_config["max_npoints"][repo_id]
    )
    self._last_realistic_failure_audit = {
        "task_index": int(mini_batch["realistic_failure_task_index"]),
        "source_episode": int(mini_batch["realistic_failure_source_episode"]),
        "base_seed": int(mini_batch["realistic_failure_base_seed"]),
        "step": int(mini_batch.get("realistic_failure_step", -1)),
        "clean_voxel_points": int(len(clean)),
        "model_input_points": int(len(point_cloud)),
        "corruption": result.stats,
    }
    return np.ascontiguousarray(point_cloud, dtype=np.float32)


class RealisticFailuresV2Policy(Policy):
    def __init__(self, args: Args):
        super().__init__(args)
        self.audit_path = Path(args.audit_path) if args.audit_path else None
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def get_action(self, batch, options):
        output = super().get_action(batch, options)
        audit = getattr(self.processor, "_last_realistic_failure_audit", None)
        if self.audit_path is not None and audit is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(audit, separators=(",", ":")) + "\n")
        return output


def main(args: Args) -> None:
    checkpoint = Path(args.pretrained_path)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("ptv3_input_channels") != 6:
        raise ValueError("Checkpoint is not configured for six-channel XYZRGB points")
    if config.get("architectures") != ["VLAEncDec3DWithActionClassificationModel"]:
        raise ValueError(f"Unexpected architecture: {config.get('architectures')}")
    set_seed(args.seed)
    policy = RealisticFailuresV2Policy(args)
    policy.processor._prepare_point_cloud_for_sample = MethodType(
        _prepare_realistic_failure, policy.processor
    )
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    main(tyro.cli(Args))
