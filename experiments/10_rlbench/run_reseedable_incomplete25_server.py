"""Plain PointACT policy server with optional episode-consistent missing points.

This server intentionally omits feature and attention capture.  It is used for
success-rate evaluation where the baseline receives complete point clouds and
the incomplete-trained checkpoint receives its training-matched 25% structured
missingness with an evaluation-only corruption seed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import tyro

from create_episode_consistent_incomplete_pointcloud_dataset import (
    consistent_missing_mask,
    episode_seed,
    make_hole_field,
)
from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed
from scripts.run_server import Policy, ServerArgs


@dataclasses.dataclass
class Args(ServerArgs):
    missing_rate: float = 0.0
    corruption_seed: int = 2026092001
    num_holes: int = 6


class ReseedableIncompletePolicy(Policy):
    def __init__(self, args: Args):
        super().__init__(args)
        self.args = args
        self.episode_id: int | None = None
        self.hole_field = None

        if args.missing_rate == 0:
            return
        if not 0 < args.missing_rate < 1:
            raise ValueError("missing_rate must be zero or in (0, 1)")

        processor = self.processor

        def prepare_incomplete_point_cloud(
            mini_batch,
            repo_id,
            workspace,
            *,
            remove_arm=False,
            voxel_size=0.01,
        ):
            if self.episode_id is None:
                raise RuntimeError("Client reset must provide episode_id")
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
            if self.hole_field is None:
                self.hole_field = make_hole_field(
                    point_cloud[:, :3],
                    episode_seed(args.corruption_seed, self.episode_id),
                    args.num_holes,
                )
            missing = consistent_missing_mask(
                point_cloud[:, :3], args.missing_rate, self.hole_field
            )
            incomplete = np.ascontiguousarray(point_cloud[~missing], dtype=np.float32)
            return processor._subsample_point_cloud(
                incomplete, processor.robot_config["max_npoints"][repo_id]
            )

        processor._prepare_point_cloud_for_sample = prepare_incomplete_point_cloud

    def reset(self, options=None):
        options = options or {}
        self.episode_id = int(options.get("episode_id", 0))
        self.hole_field = None
        set_seed(int(options.get("seed", self.episode_id)))
        return super().reset(options=options)


def main(args: Args) -> None:
    set_seed(args.seed)
    policy = ReseedableIncompletePolicy(args)
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    tyro.cli(main)
