"""Serve a classifier trained on precomputed filled XYZRGB point clouds."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import MethodType

import numpy as np
import tyro

from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed
from scripts.run_server import Policy


@dataclasses.dataclass
class Args:
    pretrained_path: str
    seed: int = 7
    host: str = "127.0.0.1"
    port: int = 5555
    num_denoise_steps: int = 10
    save_data: bool = False
    save_dir: str = "/tmp/pointact_filled6_server_inputs"


def _prepare_filled6(
    self, mini_batch, repo_id, workspace, *, remove_arm=False, voxel_size=0.01
):
    """Match filled6 training preprocessing without a second voxelization.

    The repaired archive already stores the selected representatives of a
    12 mm voxel grid. Training reads the first six columns of that archive,
    workspace-filters them, and samples at most ``max_npoints``. Repeating
    Open3D voxelization here would change both geometry and RGB representatives.
    """
    del voxel_size  # This input has already been voxelized offline/live.
    if "observation.points" not in mini_batch:
        raise ValueError("filled6 server requires precomputed observation.points")
    raw = self._as_numpy_point_cloud(mini_batch["observation.points"])
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError(f"Expected filled XYZRGB point cloud with shape Nx6, got {raw.shape}")
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError("filled6 point cloud must be non-empty and finite")

    copied_batch = dict(mini_batch)
    copied_batch["observation.points"] = np.ascontiguousarray(raw, dtype=np.float32)
    cloud = self._build_existing_point_cloud(copied_batch, workspace)
    if remove_arm and "observation.robot_joints_bbox" in copied_batch:
        cloud = self._remove_robot_arm_points(cloud, copied_batch)
    return self._subsample_point_cloud(
        cloud, self.robot_config["max_npoints"][repo_id]
    )


def main(args: Args) -> None:
    checkpoint = Path(args.pretrained_path)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("ptv3_input_channels") != 6:
        raise ValueError("Checkpoint is not configured for six-channel XYZRGB points")
    if config.get("architectures") != ["VLAEncDec3DWithActionClassificationModel"]:
        raise ValueError(f"Unexpected architecture: {config.get('architectures')}")
    if config.get("use_target_reconstruction", False):
        raise ValueError("Expected the non-reconstruction classifier checkpoint")
    if config.get("use_polar_material_conditioning", False):
        raise ValueError("Expected the XYZRGB-only checkpoint without polar conditioning")

    processor = json.loads((checkpoint / "processor_config.json").read_text())
    robot_config = processor["robot_config"]
    expected_repo = "hybridvla_10tasks_train_keysteps_polar_rlbench9_v2"
    if set(robot_config.get("features", {})) != {expected_repo}:
        raise ValueError(
            "Checkpoint processor was not trained on the repaired RLBench9 v2 dataset"
        )
    if robot_config.get("max_npoints", {}).get(expected_repo) != 4096:
        raise ValueError("Checkpoint processor does not use the expected 4096-point budget")
    if any(robot_config.get("select_video_keys_for_vlm", {}).values()):
        raise ValueError("Checkpoint unexpectedly sends images to the VLM")

    set_seed(args.seed)
    policy = Policy(args)
    policy.processor._prepare_point_cloud_for_sample = MethodType(
        _prepare_filled6, policy.processor
    )
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    main(tyro.cli(Args))
