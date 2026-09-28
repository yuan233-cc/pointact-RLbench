#!/usr/bin/env python3
"""Rebuild incomplete/filled branches from an existing clean LIBERO replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from modalities import (
    complete_corrupted_cloud,
    corrupt_libero_cloud,
    interaction_reconstruction_supervision,
    realign_corrupted_features,
    unproject_depth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source one-episode dataset root")
    parser.add_argument("output", type=Path, help="new dataset root; must not exist")
    parser.add_argument("--robot-drop-fraction", type=float, required=True)
    parser.add_argument("--target-affected-fraction", type=float, required=True)
    parser.add_argument("--target-drop-fraction", type=float, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for name in ("robot_drop_fraction", "target_affected_fraction", "target_drop_fraction"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["records"]) != 1:
        raise ValueError("This utility currently requires exactly one source episode")
    record = manifest["records"][0]
    relative_episode = Path(f"task_{record['task_id']:02d}") / f"episode_{record['source_episode']:06d}"
    source_episode = args.source / relative_episode
    output_episode = args.output / relative_episode
    output_frames = output_episode / "frames"
    output_frames.mkdir(parents=True)

    summary = json.loads((source_episode / "summary.json").read_text(encoding="utf-8"))
    geom_names = {int(key): value for key, value in summary["geom_names"].items()}
    seed = int(manifest["seed"])
    global_episode = int(summary["global_episode"])
    frame_stats = []
    for frame_path in sorted((source_episode / "frames").glob("*.npz")):
        frame_number = int(frame_path.stem)
        with np.load(frame_path) as source:
            arrays = {key: source[key] for key in source.files}
            polar = {
                "DoLP": arrays["DoLP"],
                "cos2AoLP": arrays["cos2AoLP"],
                "sin2AoLP": arrays["sin2AoLP"],
                "valid_mask": arrays["polar_valid_mask"],
            }
            corrupted = corrupt_libero_cloud(
                arrays["clean9"], arrays["clean_pixel_index"], arrays["clean_geom_id"],
                geom_names, arrays["camera_to_world"][:3, 3], episode_index=global_episode,
                seed=seed, robot_drop_fraction=args.robot_drop_fraction,
                target_affected_fraction=args.target_affected_fraction,
                target_drop_fraction=args.target_drop_fraction,
            )
            corrupted = realign_corrupted_features(
                corrupted, arrays["rgb"], polar, arrays["camera_intrinsics"],
                arrays["camera_to_world"])
            filled = complete_corrupted_cloud(
                corrupted, arrays["clean_pixel_index"], arrays["rgb"], polar,
                arrays["camera_intrinsics"], arrays["camera_to_world"])
            world = unproject_depth(
                arrays["depth_m"], arrays["camera_intrinsics"], arrays["camera_to_world"])
            supervision = interaction_reconstruction_supervision(
                filled.cloud, world, arrays["depth_m"], arrays["geom_id"], geom_names,
                arrays["camera_intrinsics"], arrays["camera_to_world"],
                seed=seed + global_episode * 100000 + frame_number,
                max_points=int(manifest["target_max_points"]),
                voxel_size=float(manifest["target_voxel_size_m"]),
                workspace_low=np.array([-0.5, -0.5, 0.85]),
                workspace_high=np.array([0.5, 0.5, 1.7]),
            )
            arrays.update(
                incomplete9=corrupted.cloud,
                incomplete_source_pixel_index=corrupted.source_pixels,
                incomplete_current_pixel_index=corrupted.current_pixels,
                incomplete_corruption_code=corrupted.codes,
                filled9=filled.cloud,
                filled_source_pixel_index=filled.source_pixels,
                filled_current_pixel_index=filled.current_pixels,
                filled_synthetic_mask=filled.synthetic_mask,
                filled_corruption_code=filled.codes,
                interaction_target_points=supervision.target_points,
                interaction_input_mask=supervision.input_mask,
            )
            np.savez_compressed(output_frames / frame_path.name, **arrays)

        old_stats = summary["frames"][frame_number]
        frame_stats.append({
            "frame": frame_number,
            "source_step": old_stats["source_step"],
            "polar": old_stats["polar"],
            "corruption": corrupted.stats,
            "completion": filled.stats,
            "interaction_supervision": supervision.stats,
        })
        print(f"frame {frame_number:03d}: {len(arrays['clean9'])} -> {len(corrupted.cloud)}", flush=True)

    config = {
        "robot_drop_fraction": args.robot_drop_fraction,
        "target_affected_fraction": args.target_affected_fraction,
        "target_drop_fraction": args.target_drop_fraction,
    }
    summary["frames"] = frame_stats
    summary["corruption_config"] = config
    summary["derived_from"] = str(args.source.resolve())
    output_episode.joinpath("summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest["corruption_config"] = config
    manifest["derived_from"] = str(args.source.resolve())
    args.output.joinpath("manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
