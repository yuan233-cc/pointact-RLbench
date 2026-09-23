"""Prepare a run-local RLBench classifier config without normalizing actions.

The classifier predicts absolute XYZ, Euler angles, and a binary gripper state.
Normalizing these targets makes the BCE target leave [0, 1] and corrupts the
rotation bins. Keep the dataset's state normalization, but use identity action
mean/std in a copy of its statistics file. The source dataset is never edited.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml


def _read_stats(path: Path) -> dict:
    with path.open() as handle:
        stats = json.load(handle)
    for key in ("state_mean", "state_std", "action_mean", "action_std", "action_min", "action_max"):
        values = stats.get(key)
        if not isinstance(values, list) or len(values) != 7:
            raise ValueError(f"{path}: {key} must contain seven values")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError(f"{path}: {key} contains a non-finite/non-numeric value")
    if any(value <= 0 for value in stats["state_std"] + stats["action_std"]):
        raise ValueError(f"{path}: normalization standard deviations must be positive")
    if stats["action_min"][6] < -1e-5 or stats["action_max"][6] > 1 + 1e-5:
        raise ValueError(f"{path}: expected raw gripper labels in [0, 1]")
    return stats


def prepare(input_config: Path, output_dir: Path, dataset_root: Path | None = None) -> Path:
    input_config = input_config.resolve(strict=True)
    with input_config.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("lerobot_datasets"), list):
        raise ValueError(f"{input_config}: expected a lerobot_datasets list")
    if not config["lerobot_datasets"]:
        raise ValueError(f"{input_config}: no LeRobot dataset configured")
    if dataset_root is not None:
        if len(config["lerobot_datasets"]) != 1:
            raise ValueError("--dataset-root requires exactly one LeRobot dataset")
        dataset_root = dataset_root.resolve(strict=True)
        if not (dataset_root / "meta/info.json").is_file():
            raise ValueError(f"{dataset_root}: meta/info.json is missing")

    replacements = []
    for index, dataset in enumerate(config["lerobot_datasets"]):
        if not isinstance(dataset, dict):
            raise ValueError(f"Dataset {index}: expected a mapping")
        if (dataset.get("point_cloud_dirname") != "points_frontview_polar_filled9"
                or dataset.get("point_feature_mode") != "xyzrgb_polar"):
            raise ValueError(f"Dataset {index}: expected filled9 XYZRGB+polar point clouds")
        if dataset.get("converted_rot_type") != "euler" or dataset.get("is_delta_action") is not False:
            raise ValueError(f"Dataset {index}: classifier requires absolute Euler actions")
        source = dataset.get("state_action_norm_file")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Dataset {index}: state_action_norm_file is required")
        if dataset_root is not None:
            if dataset.get("repo_id") != dataset_root.name:
                raise ValueError(
                    f"Dataset {index}: repo_id={dataset.get('repo_id')!r} does not match "
                    f"mounted dataset {dataset_root.name!r}"
                )
            dataset["root"] = str(dataset_root.parent)
            source_path = dataset_root / "robot_state_action_stats" / Path(source).name
        else:
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = Path.cwd() / source_path
        source_path = source_path.resolve(strict=True)
        stats = _read_stats(source_path)
        original_mean = stats["action_mean"]
        original_std = stats["action_std"]
        stats["action_mean"] = [0.0] * 7
        stats["action_std"] = [1.0] * 7
        target = output_dir / f"action-stats-{index}.json"
        dataset["state_action_norm_file"] = str(target.resolve())
        replacements.append((source_path, target, stats, original_mean, original_std))

    output_dir.mkdir(parents=True, exist_ok=False)
    for source, target, stats, original_mean, original_std in replacements:
        with target.open("w") as handle:
            json.dump(stats, handle, indent=2)
            handle.write("\n")
        changed = original_mean != stats["action_mean"] or original_std != stats["action_std"]
        print(f"Classifier action stats: {source} -> {target}; identity override={changed}", file=sys.stderr)
    output_config = output_dir / "data.yaml"
    with output_config.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return output_config.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--dataset-root", type=Path,
        help="Mounted dataset directory; rewrites root and normalization paths in the run-local config",
    )
    args = parser.parse_args()
    print(prepare(args.input_config, args.output_dir, args.dataset_root))


if __name__ == "__main__":
    main()
