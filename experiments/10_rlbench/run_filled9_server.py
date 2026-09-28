"""Local PointACT server for precomputed training-matched filled9 clouds."""

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
    save_dir: str = "/tmp/pointact_filled9_server_inputs"


def _prepare_filled9(self, mini_batch, repo_id, workspace, *, remove_arm=False, voxel_size=0.01):
    """Preserve all nine features and the training dataset's existing sampling.

    The ordinary server voxelizes an existing cloud through Open3D and thereby
    drops channels 6:9. Filled9 was already voxelized before corruption/fill,
    so a second voxel pass would also differ from training.
    """
    if "observation.points" not in mini_batch:
        raise ValueError("filled9 server requires precomputed observation.points")
    raw = self._as_numpy_point_cloud(mini_batch["observation.points"])
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError(f"Expected filled9 point cloud with shape Nx9, got {raw.shape}")
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError("filled9 point cloud must be non-empty and finite")
    copied_batch = dict(mini_batch)
    copied_batch["observation.points"] = np.ascontiguousarray(raw.copy(), dtype=np.float32)
    cloud = self._build_existing_point_cloud(copied_batch, workspace)
    normalizations = self.robot_config.get("polar_feature_normalization", {})
    if normalizations.get(repo_id, "raw") == "rgb":
        cloud[:, 6:9] = normalize_polar_like_rgb(cloud[:, 6:9])
    if remove_arm and "observation.robot_joints_bbox" in copied_batch:
        cloud = self._remove_robot_arm_points(cloud, copied_batch)
    return self._subsample_point_cloud(cloud, self.robot_config["max_npoints"][repo_id])


def main(args: Args) -> None:
    checkpoint = Path(args.pretrained_path)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("ptv3_input_channels") != 9:
        raise ValueError("Checkpoint is not configured for nine-channel points")
    if config.get("architectures") != ["VLAEncDec3DWithActionClassificationModel"]:
        raise ValueError(f"Unexpected architecture: {config.get('architectures')}")
    set_seed(args.seed)
    policy = Policy(args)
    policy.processor._prepare_point_cloud_for_sample = MethodType(
        _prepare_filled9, policy.processor
    )
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    main(tyro.cli(Args))
