"""Summarize success-only evaluation for the remaining seven RLBench tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TASKS = (
    "sweep_to_dustpan",
    "close_fridge",
    "phone_on_base",
    "take_umbrella_out_of_umbrella_stand",
    "take_frame_off_hanger",
    "stack_wine",
    "water_plants",
)
CONDITIONS = ("old_complete", "new_incomplete25")


def wilson(successes: int, episodes: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / episodes
    denominator = 1 + z * z / episodes
    center = (proportion + z * z / (2 * episodes)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / episodes
        + z * z / (4 * episodes * episodes)
    ) / denominator
    return [center - half, center + half]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--rng-mode", choices=("original", "episode-reset"), default="original")
    parser.add_argument("--corruption-seed", type=int, default=2026092001)
    args = parser.parse_args()

    rng_description = (
        "server seeded once; no per-episode NumPy/PyTorch RNG reset"
        if args.rng_mode == "original"
        else "both servers reset Python/NumPy/PyTorch RNG to 7 + episode_id before each episode"
    )

    output = {
        "protocol": {
            "episodes_per_task_per_checkpoint": args.episodes,
            "variation": 0,
            "environment_seed": 7,
            "old_checkpoint_input": "complete point cloud (training-consistent)",
            "new_checkpoint_input": (
                "25% episode-consistent structured missing point cloud "
                f"(training-consistent algorithm, unseen corruption seed {args.corruption_seed})"
            ),
            "rng_protocol": rng_description,
            "policy_seed_base": 7,
            "corruption_seed": args.corruption_seed,
            "replan_steps": 1,
            "max_policy_steps": 25,
            "num_workers": 1,
            "saved_video": False,
        },
        "tasks": {},
    }

    for task in TASKS:
        output["tasks"][task] = {}
        for condition in CONDITIONS:
            path = args.output_root / condition / task / "episode_results.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines() if line]
            if len(records) != args.episodes:
                raise ValueError(f"{path}: expected {args.episodes}, found {len(records)}")
            if sorted(int(record["episode"]) for record in records) != list(range(args.episodes)):
                raise ValueError(f"{path}: episode IDs must be unique and cover 0..{args.episodes - 1}")
            if any(record["task"] != task or int(record["variation"]) != 0 for record in records):
                raise ValueError(f"{path}: unexpected task or variation")
            successes = sum(bool(record["success"]) for record in records)
            output["tasks"][task][condition] = {
                "successes": successes,
                "episodes": len(records),
                "success_rate": successes / len(records),
                "success_rate_wilson_95ci": wilson(successes, len(records)),
                "failure_episodes": [
                    int(record["episode"])
                    for record in records
                    if not record["success"]
                ],
            }

    output["macro_average_success_rate"] = {
        condition: sum(
            output["tasks"][task][condition]["success_rate"] for task in TASKS
        ) / len(TASKS)
        for condition in CONDITIONS
    }
    (args.output_root / "success_summary.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    rows = [
        "# Remaining seven tasks: training-consistent checkpoint comparison",
        "",
        "| Task | Old / complete | New / 25% incomplete |",
        "|---|---:|---:|",
    ]
    for task in TASKS:
        old = output["tasks"][task]["old_complete"]
        new = output["tasks"][task]["new_incomplete25"]
        rows.append(
            f"| {task} | {old['successes']}/{old['episodes']} "
            f"({old['success_rate']:.1%}) | {new['successes']}/{new['episodes']} "
            f"({new['success_rate']:.1%}) |"
        )
    rows.extend(
        [
            "",
            "Both conditions use environment seed 7, variation 0, one worker, "
            "one replan per policy step, and at most 25 policy steps.",
            rng_description + ".",
            f"New-checkpoint corruption seed: {args.corruption_seed}; "
            "one fixed spatial missing field per episode, removing 25% per frame.",
        ]
    )
    (args.output_root / "README.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
