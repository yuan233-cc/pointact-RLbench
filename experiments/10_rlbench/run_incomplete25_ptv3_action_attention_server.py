"""PointACT attention server with training-matched 25% structured missing points."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import tyro

from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed
from create_episode_consistent_incomplete_pointcloud_dataset import (
    consistent_missing_mask,
    episode_seed,
    make_hole_field,
)
from run_ptv3_action_attention_server import Args, AttentionPolicy


TRAINING_CORRUPTION_SEED = 20260917


@dataclasses.dataclass
class IncompleteArgs(Args):
    missing_rate: float = 0.25
    corruption_seed: int = 2026091801
    num_holes: int = 6


class Incomplete25AttentionPolicy(AttentionPolicy):
    """Apply the exact training corruption between voxelization and subsampling."""

    def __init__(self, args: IncompleteArgs):
        super().__init__(args)
        if not 0.0 < args.missing_rate < 1.0:
            raise ValueError("missing_rate must be in (0, 1)")
        if args.num_holes <= 0:
            raise ValueError("num_holes must be positive")
        if args.corruption_seed == TRAINING_CORRUPTION_SEED:
            raise ValueError(
                "Evaluation corruption_seed must differ from the training seed "
                f"{TRAINING_CORRUPTION_SEED}"
            )
        self._episode_id: int | None = None
        self._request_in_episode = 0
        self._hole_field = None
        self._latest_corruption: dict[str, np.ndarray] | None = None
        self._corruption_manifest = Path(args.capture_dir) / "corruption_manifest.jsonl"
        if self._corruption_manifest.exists():
            raise FileExistsError(self._corruption_manifest)

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
                raise RuntimeError(
                    "Client must reset the policy with episode_id before inference"
                )
            has_existing = "observation.points" in mini_batch
            if has_existing:
                point_cloud = processor._build_existing_point_cloud(
                    mini_batch, workspace
                )
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
                point_cloud = processor._remove_robot_arm_points(
                    point_cloud, mini_batch
                )
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
            self._latest_corruption = {
                "corruption_full_coordinates_world": np.ascontiguousarray(
                    point_cloud[:, :3], dtype=np.float32
                ),
                "corruption_full_rgb": np.ascontiguousarray(
                    point_cloud[:, 3:6], dtype=np.float32
                ),
                "corruption_removed_coordinates_world": np.ascontiguousarray(
                    point_cloud[missing, :3], dtype=np.float32
                ),
                "corruption_removed_rgb": np.ascontiguousarray(
                    point_cloud[missing, 3:6], dtype=np.float32
                ),
                "corruption_field_centers_world": self._hole_field.centers.copy(),
                "corruption_field_radii": self._hole_field.radii.copy(),
                "corruption_episode_id": np.asarray(self._episode_id, dtype=np.int64),
                "corruption_request_in_episode": np.asarray(
                    self._request_in_episode, dtype=np.int64
                ),
                "corruption_input_points": np.asarray(len(point_cloud), dtype=np.int64),
                "corruption_removed_points": np.asarray(missing.sum(), dtype=np.int64),
                "corruption_kept_points_before_subsample": np.asarray(
                    (~missing).sum(), dtype=np.int64
                ),
                "corruption_model_input_points": np.asarray(
                    len(incomplete), dtype=np.int64
                ),
                "corruption_missing_rate": np.asarray(
                    missing.mean(), dtype=np.float32
                ),
                "corruption_requested_missing_rate": np.asarray(
                    args.missing_rate, dtype=np.float32
                ),
                "corruption_seed": np.asarray(
                    args.corruption_seed, dtype=np.int64
                ),
                "corruption_num_holes": np.asarray(
                    args.num_holes, dtype=np.int64
                ),
            }
            record = {
                "episode_id": self._episode_id,
                "request_in_episode": self._request_in_episode,
                "input_points": int(len(point_cloud)),
                "removed_points": int(missing.sum()),
                "kept_points_before_subsample": int((~missing).sum()),
                "model_input_points": int(len(incomplete)),
                "missing_rate": float(missing.mean()),
                "rule": (
                    f"round(N * {args.missing_rate}) "
                    "lowest shared-ellipsoid-score points"
                ),
                "corruption_seed": args.corruption_seed,
                "num_holes": args.num_holes,
            }
            with self._corruption_manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            self._request_in_episode += 1
            return incomplete

        processor._prepare_point_cloud_for_sample = prepare_incomplete_point_cloud

        outer_ptv3 = self.model.ptv3_model

        def attach_corruption(_module, _inputs):
            if self.capture.pending_arrays is not None and self._latest_corruption:
                self.capture.pending_arrays.update(self._latest_corruption)

        outer_ptv3.register_forward_pre_hook(attach_corruption)

    def reset(self, options=None):
        options = options or {}
        if "episode_id" not in options:
            raise ValueError("reset requires episode_id")
        self._episode_id = int(options["episode_id"])
        self._request_in_episode = 0
        self._hole_field = None
        self._latest_corruption = None
        set_seed(int(options.get("seed", self._episode_id)))
        return super().reset(options=options)


def main(args: IncompleteArgs) -> None:
    set_seed(args.seed)
    policy = Incomplete25AttentionPolicy(args)
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    tyro.cli(main)
