#!/usr/bin/env python3
"""Validate clean LIBERO-Spatial NPZs against official HDF5 demonstrations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from replay_libero_spatial_clean import is_noop, openvla_action, state8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-image-mae", type=float, default=15.0)
    args = parser.parse_args()

    manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    summaries = sorted(args.source.glob("task_*/episode_*/summary.json"))
    if not manifest.get("complete") or len(summaries) != 500:
        raise AssertionError("Expected a complete 500-episode source manifest")

    episodes = steps = 0
    task_counts: Counter[int] = Counter()
    max_image_mae = 0.0
    max_reprojection_error_px = 0.0
    pc_min = np.full(3, np.inf, dtype=np.float64)
    pc_max = np.full(3, -np.inf, dtype=np.float64)
    sample_pixels = np.array(
        [[0, 0], [0, 255], [255, 0], [255, 255], [64, 64], [128, 128], [192, 192]],
        dtype=np.int64,
    )

    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("success") or summary.get("state_source") != "official_hdf5_robot_states":
            raise AssertionError(f"Invalid summary provenance: {summary_path}")
        task_id = int(summary["task_id"])
        episode_index = int(summary["source_episode"])
        frame_paths = sorted((summary_path.parent / "frames").glob("*.npz"))
        k = np.asarray(summary["camera_intrinsics"], dtype=np.float64)
        camera_to_world = np.asarray(summary["camera_to_world"], dtype=np.float64)
        if k.shape != (3, 3) or camera_to_world.shape != (4, 4):
            raise AssertionError(f"Invalid calibration: {summary_path}")

        with h5py.File(summary["source"], "r") as source_file:
            demo = source_file["data"][f"demo_{episode_index}"]
            actions = np.asarray(demo["actions"])
            robot_states = np.asarray(demo["robot_states"])
            expected_source_steps = []
            previous = None
            for source_step, action in enumerate(actions):
                if not is_noop(action, previous):
                    expected_source_steps.append(source_step)
                    previous = action
            if not (
                len(frame_paths)
                == len(expected_source_steps)
                == int(summary["saved_frames"])
            ):
                raise AssertionError(f"Frame-count mismatch: {summary_path.parent}")

            image_check_indices = {0, len(frame_paths) // 2, len(frame_paths) - 1}
            for frame_index, (frame_path, source_step) in enumerate(
                zip(frame_paths, expected_source_steps)
            ):
                with np.load(frame_path) as frame:
                    image = np.asarray(frame["image"])
                    point_cloud = np.asarray(frame["base_pc"])
                    state = np.asarray(frame["state"])
                    action = np.asarray(frame["action"])
                    stored_source_step = int(frame["source_step"])
                if stored_source_step != source_step:
                    raise AssertionError(f"source_step mismatch: {frame_path}")
                if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                    raise AssertionError(f"Invalid image: {frame_path}")
                if point_cloud.shape != (256, 256, 3) or point_cloud.dtype != np.float32:
                    raise AssertionError(f"Invalid base_pc: {frame_path}")
                if not np.isfinite(point_cloud).all():
                    raise AssertionError(f"Non-finite base_pc: {frame_path}")

                official = robot_states[source_step]
                raw_state = np.concatenate((official[2:], official[:2]))
                if not np.array_equal(state, state8(raw_state)):
                    raise AssertionError(f"State mismatch: {frame_path}")
                if not np.array_equal(action, openvla_action(actions[source_step])):
                    raise AssertionError(f"Action mismatch: {frame_path}")

                if frame_index in image_check_indices:
                    downsampled = np.asarray(
                        Image.fromarray(image).resize((128, 128), Image.Resampling.BILINEAR)
                    )
                    expected_image = np.asarray(demo["obs/agentview_rgb"][source_step])[::-1, ::-1]
                    image_mae = float(
                        np.mean(np.abs(downsampled.astype(np.float32) - expected_image.astype(np.float32)))
                    )
                    max_image_mae = max(max_image_mae, image_mae)
                    if image_mae > args.max_image_mae:
                        raise AssertionError(f"Rendered/source image MAE {image_mae:.3f}: {frame_path}")

                rows, cols = sample_pixels[:, 0], sample_pixels[:, 1]
                world = point_cloud[rows, cols].astype(np.float64)
                camera = (world - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
                projected_u = camera[:, 0] / camera[:, 2] * k[0, 0] + k[0, 2]
                projected_v = camera[:, 1] / camera[:, 2] * k[1, 1] + k[1, 2]
                error = np.maximum(
                    np.abs(projected_u - (255 - cols)), np.abs(projected_v - rows)
                )
                max_reprojection_error_px = max(max_reprojection_error_px, float(error.max()))
                if float(error.max()) > 1e-3:
                    raise AssertionError(f"RGB/XYZ reprojection mismatch: {frame_path}")

                pc_min = np.minimum(pc_min, point_cloud.reshape(-1, 3).min(axis=0))
                pc_max = np.maximum(pc_max, point_cloud.reshape(-1, 3).max(axis=0))
                steps += 1

        episodes += 1
        task_counts[task_id] += 1

    expected_tasks = Counter({task_id: 50 for task_id in range(10)})
    if episodes != 500 or steps != 62153 or task_counts != expected_tasks:
        raise AssertionError(
            f"Expected 500/62153/50-per-task, got {episodes}/{steps}/{dict(task_counts)}"
        )
    report = {
        "status": "passed",
        "episodes": episodes,
        "steps": steps,
        "episodes_per_task": {str(key): task_counts[key] for key in sorted(task_counts)},
        "max_sampled_rendered_vs_source_image_mae": max_image_mae,
        "max_sampled_rgb_xyz_reprojection_error_px": max_reprojection_error_px,
        "base_pc_min_xyz": pc_min.tolist(),
        "base_pc_max_xyz": pc_max.tolist(),
        "source_state_action_exact": True,
    }
    report_path = args.report or (args.source / "validation_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
