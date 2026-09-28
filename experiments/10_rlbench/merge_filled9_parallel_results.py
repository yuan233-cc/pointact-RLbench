"""Merge task-per-process filled9 RLBench evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ID = "hybridvla_10tasks_train_keysteps_polar_rlbench9_v2"
TASKS = (
    "close_box", "close_laptop_lid", "toilet_seat_down", "sweep_to_dustpan",
    "close_fridge", "phone_on_base", "take_umbrella_out_of_umbrella_stand",
    "take_frame_off_hanger", "stack_wine", "water_plants",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--corruption-seed", type=int, required=True)
    parser.add_argument("--eval-episode-offset", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    worker_results = []
    for task_index, task_name in enumerate(TASKS):
        path = args.worker_dir / f"{task_name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing worker result: {path}")
        result = json.loads(path.read_text())
        task_records = result.get("episodes", [])
        expected = [(task_name, episode) for episode in range(args.episodes)]
        actual = [(row.get("task"), row.get("episode")) for row in task_records]
        if actual != expected:
            raise ValueError(f"Unexpected episode sequence in {path}: {actual}")
        expected_sources = [
            args.eval_episode_offset + task_index * args.episodes + episode
            for episode in range(args.episodes)
        ]
        if [row.get("source_episode") for row in task_records] != expected_sources:
            raise ValueError(f"Unexpected source_episode sequence in {path}")
        records.extend(task_records)
        worker_results.append(str(path))

    summary = {
        "checkpoint_step": args.checkpoint_step,
        "repo_id": REPO_ID,
        "polar": {
            "backend": "native",
            "spp": 512,
            "max_depth": 8,
            "lighting": "reference",
            "geometry_source": "rlbench",
        },
        "corruption_seed": args.corruption_seed,
        "eval_episode_offset": args.eval_episode_offset,
        "randomization": {
            "mode": "original_rlbench_worker_stream",
            "seed": args.seed,
            "reseed_each_episode": False,
        },
        "worker_results": worker_results,
        "episodes": records,
        "successes": sum(bool(record["success"]) for record in records),
        "total": len(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary={args.output} success={summary['successes']}/{summary['total']}")


if __name__ == "__main__":
    main()
