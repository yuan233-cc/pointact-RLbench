"""Summarize success-only evaluation for the three close-manipulation tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TASKS = ("close_box", "close_laptop_lid", "toilet_seat_down")
CONDITIONS = ("old_complete", "new_incomplete25")


def wilson(successes: int, episodes: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / episodes
    denominator = 1 + z * z / episodes
    center = (proportion + z * z / (2 * episodes)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / episodes + z * z / (4 * episodes * episodes)
    ) / denominator
    return [center - half, center + half]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    output = {
        "protocol": {
            "episodes_per_task_per_checkpoint": args.episodes,
            "variation": 0,
            "environment_seed": 7,
            "old_checkpoint_input": "complete point cloud (training-consistent)",
            "new_checkpoint_input": (
                "25% episode-consistent structured missing point cloud "
                "(training-consistent algorithm, unseen corruption seed 2026092001)"
            ),
            "policy_rng_protocol": (
                "legacy inference mode: seed each policy server once; "
                "no per-episode RNG reset"
            ),
            "replan_steps": 1,
            "max_policy_steps": 25,
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
            successes = sum(bool(record["success"]) for record in records)
            output["tasks"][task][condition] = {
                "successes": successes,
                "episodes": len(records),
                "success_rate": successes / len(records),
                "success_rate_wilson_95ci": wilson(successes, len(records)),
                "failure_episodes": [
                    int(record["episode"]) for record in records if not record["success"]
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
        "# Training-consistent checkpoint success comparison",
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
            "Both conditions use environment seed 7, variation 0, the same episode order, "
            "one replan per policy step, and at most 25 policy steps.",
            "Policy RNG follows the original inference mode: each server is seeded once and "
            "is not reseeded between episodes. The new condition resets only its "
            "episode-consistent missing-region template.",
        ]
    )
    (args.output_root / "README.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
