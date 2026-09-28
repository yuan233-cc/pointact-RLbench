"""Reseedable PointACT server with training-matched 25% structured missingness."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import tyro

from create_episode_consistent_incomplete_pointcloud_dataset import (
    consistent_missing_mask,
    episode_seed,
    make_hole_field,
)
from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed
from scripts.run_server import ServerArgs
from run_reseedable_policy_server import ReseedablePolicy


TRAINING_CORRUPTION_SEED = 20260917


@dataclasses.dataclass
class IncompleteArgs(ServerArgs):
    missing_rate: float = 0.25
    corruption_seed: int = 2026091801
    num_holes: int = 6
    manifest_path: str = ""
    replay_training_corruption: bool = False


class Incomplete25Policy(ReseedablePolicy):
    def __init__(self, args: IncompleteArgs):
        super().__init__(args)
        if not 0.0 < args.missing_rate < 1.0:
            raise ValueError("missing_rate must be in (0, 1)")
        if args.num_holes <= 0:
            raise ValueError("num_holes must be positive")
        if (
            args.corruption_seed == TRAINING_CORRUPTION_SEED
            and not args.replay_training_corruption
        ):
            raise ValueError(
                "Set replay_training_corruption=True to intentionally reproduce "
                f"the training corruption seed {TRAINING_CORRUPTION_SEED}"
            )
        self._episode_id = None
        self._request_in_episode = 0
        self._hole_field = None
        self._manifest_path = Path(args.manifest_path) if args.manifest_path else None
        if self._manifest_path is not None:
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if self._manifest_path.exists():
                raise FileExistsError(self._manifest_path)

        processor = self.processor

        def prepare_incomplete_point_cloud(
            mini_batch,
            repo_id,
            workspace,
            *,
            remove_arm=False,
            voxel_size=0.01,
        ):
            if self._episode_id is None:
                raise RuntimeError("Client must reset with episode_id before inference")
            has_existing = "observation.points" in mini_batch
            if has_existing:
                point_cloud = processor._build_existing_point_cloud(mini_batch, workspace)
            else:
                point_cloud = processor._build_point_cloud_from_cameras(
                    mini_batch, repo_id, workspace
                )
            point_cloud = processor._voxel_downsample_point_cloud(
                point_cloud, voxel_size=voxel_size
            )
            if remove_arm and (
                not has_existing or "observation.robot_joints_bbox" in mini_batch
            ):
                point_cloud = processor._remove_robot_arm_points(point_cloud, mini_batch)
            if self._hole_field is None:
                self._hole_field = make_hole_field(
                    point_cloud[:, :3],
                    episode_seed(args.corruption_seed, self._episode_id),
                    args.num_holes,
                )
            missing = consistent_missing_mask(
                point_cloud[:, :3], args.missing_rate, self._hole_field
            )
            incomplete = np.ascontiguousarray(point_cloud[~missing], dtype=np.float32)
            incomplete = processor._subsample_point_cloud(
                incomplete, processor.robot_config["max_npoints"][repo_id]
            )
            if self._manifest_path is not None:
                record = {
                    "episode_id": self._episode_id,
                    "request_in_episode": self._request_in_episode,
                    "input_points": int(len(point_cloud)),
                    "removed_points": int(missing.sum()),
                    "model_input_points": int(len(incomplete)),
                    "missing_rate": float(missing.mean()),
                    "corruption_seed": args.corruption_seed,
                    "num_holes": args.num_holes,
                }
                with self._manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            self._request_in_episode += 1
            return incomplete

        processor._prepare_point_cloud_for_sample = prepare_incomplete_point_cloud

    def reset(self, options=None):
        options = options or {}
        if "episode_id" not in options:
            raise ValueError("reset requires episode_id")
        self._episode_id = int(options["episode_id"])
        self._request_in_episode = 0
        self._hole_field = None
        return super().reset(options=options)


def main(args: IncompleteArgs) -> None:
    set_seed(args.seed)
    policy = Incomplete25Policy(args)
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    tyro.cli(main)
