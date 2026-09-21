"""Create a classifier data config for a mounted realistic-failures v2 dataset."""

import argparse
import json
from pathlib import Path

import yaml


DATASET_NAME = "hybridvla_10tasks_train_keysteps_realistic_failures_v2"
TEMPLATE = Path(__file__).resolve().parent / "data_configs/data-hybridvla-point-clf-frontview-realistic-failures-v2.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve(strict=True)
    if dataset_root.name != DATASET_NAME:
        parser.error(f"Expected dataset directory {DATASET_NAME}, got {dataset_root}")

    required = (
        dataset_root / "meta/info.json",
        dataset_root / "points_frontview/data.mdb",
        dataset_root / "robot_state_action_stats/euler_points_frontview_clf.json",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"Missing or empty dataset file: {path}")

    with (dataset_root / "meta/info.json").open(encoding="utf-8") as stream:
        info = json.load(stream)
    expected = {"total_tasks": 10, "total_episodes": 1000, "total_frames": 5056}
    if any(info.get(key) != value for key, value in expected.items()):
        parser.error(f"Dataset counts differ from expected v2 counts: {info}")

    with TEMPLATE.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dataset = config["lerobot_datasets"][0]
    dataset["root"] = str(dataset_root.parent)
    dataset["repo_id"] = DATASET_NAME
    dataset["state_action_norm_file"] = str(required[2])

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(output)


if __name__ == "__main__":
    main()
