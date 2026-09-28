"""Serve an XYZ+polar classifier from live/precomputed filled9 observations."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import MethodType

import numpy as np
import tyro

from pointact.data.transforms.pointcloud import normalize_polar_like_rgb
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
    save_dir: str = "/tmp/pointact_xyzpolar_filled6_server_inputs"


def _prepare_xyzpolar(
    self, mini_batch, repo_id, workspace, *, remove_arm=False, voxel_size=0.01
):
    """Select XYZ and polar columns and apply training-matched normalization."""
    del voxel_size
    if "observation.points" not in mini_batch:
        raise ValueError("XYZ+polar server requires a precomputed filled9 observation.points")
    raw = self._as_numpy_point_cloud(mini_batch["observation.points"])
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError(f"Expected filled9 point cloud with shape Nx9, got {raw.shape}")
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError("filled9 point cloud must be non-empty and finite")
    if np.any((raw[:, 6] < 0) | (raw[:, 6] > 1)) or np.any(np.abs(raw[:, 7:9]) > 1.001):
        raise ValueError("filled9 point cloud contains invalid polarization features")

    cloud = np.ascontiguousarray(
        np.column_stack((raw[:, :3], raw[:, 6:9])),
        dtype=np.float32,
    )
    cloud[:, 3:6] = normalize_polar_like_rgb(cloud[:, 3:6])
    cloud = self._filter_points_by_workspace(cloud, workspace)
    if remove_arm and "observation.robot_joints_bbox" in mini_batch:
        cloud = self._remove_robot_arm_points(cloud, mini_batch)
    return self._subsample_point_cloud(cloud, self.robot_config["max_npoints"][repo_id])


def main(args: Args) -> None:
    checkpoint = Path(args.pretrained_path)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("ptv3_input_channels") != 6:
        raise ValueError("Checkpoint is not configured for six-channel XYZ+polar points")
    if config.get("architectures") != ["VLAEncDec3DWithActionClassificationModel"]:
        raise ValueError(f"Unexpected architecture: {config.get('architectures')}")

    processor = json.loads((checkpoint / "processor_config.json").read_text())
    modes = processor.get("robot_config", {}).get("point_feature_mode", {})
    if not modes or set(modes.values()) != {"xyz_polar"}:
        raise ValueError("Checkpoint processor is not configured for xyz_polar point features")
    normalizations = processor.get("robot_config", {}).get("polar_feature_normalization", {})
    if not normalizations or set(normalizations.values()) != {"rgb"}:
        raise ValueError("Checkpoint processor does not use RGB-like polar normalization")
    if any(processor["robot_config"].get("select_video_keys_for_vlm", {}).values()):
        raise ValueError("Checkpoint unexpectedly sends images to the VLM")

    set_seed(args.seed)
    policy = Policy(args)
    policy.processor._prepare_point_cloud_for_sample = MethodType(
        _prepare_xyzpolar, policy.processor
    )
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    main(tyro.cli(Args))
