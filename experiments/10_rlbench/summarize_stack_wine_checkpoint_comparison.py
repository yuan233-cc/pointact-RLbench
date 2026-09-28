"""Create a compact comparison report for the two stack_wine checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--task", default="stack_wine")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--episodes-per-checkpoint", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [json.loads((run / "summary.json").read_text()) for run in args.runs]

    def input_condition(item: dict, *, compact: bool = False) -> str:
        corruption = item.get("point_cloud_corruption")
        if corruption is None:
            return "complete"
        if corruption.get("seed_independent_from_training"):
            return (
                "25% episode-consistent missing (unseen evaluation seed)"
                if compact
                else "25% episode-consistent structured missingness "
                "(training-matched algorithm, unseen evaluation seed)"
            )
        return (
            "25% episode-consistent missing (training-matched)"
            if compact
            else "25% episode-consistent structured missingness (training-matched)"
        )

    comparison = {
        "protocol": {
            "task": args.task,
            "variation": args.variation,
            "episodes_per_checkpoint": args.episodes_per_checkpoint,
            "seed": 7,
            "replan_steps": 1,
            "max_policy_steps": 25,
            "num_denoise_steps": 10,
            "paired_seed_and_episode_order": True,
            "videos": "keyframe action-overlay video plus continuous 2 Hz simulator video",
            "attention": (
                "one exact Stage-4 block-2 action-query→point-key capture per policy step; "
                "attention MP4 and first/middle/final PNGs for every success and failure"
            ),
            "input_conditions": {
                item["checkpoint_label"]: input_condition(item)
                for item in summaries
            },
        },
        "checkpoints": summaries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    labels = [item["checkpoint_label"] for item in summaries]
    rates = np.asarray([item["success_rate"] for item in summaries])
    intervals = np.asarray([item["success_rate_wilson_95ci"] for item in summaries])
    errors = np.vstack([rates - intervals[:, 0], intervals[:, 1] - rates])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, rates, yerr=errors, capsize=7, color=("#5276A7", "#D97A43"))
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("RLBench success rate")
    ax.set_title(
        f"{args.task} variation {args.variation} — "
        f"{args.episodes_per_checkpoint} episodes per checkpoint"
    )
    ax.grid(axis="y", alpha=0.25)
    for bar, item in zip(bars, summaries):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{item['successes']}/{item['episodes']}",
            ha="center",
            va="bottom",
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(args.output_root / "success_rate_comparison.png", dpi=180)
    plt.close(fig)

    rows = [
        f"# {args.task} checkpoint comparison",
        "",
        "| Checkpoint | Point-cloud input | Successes | Success rate | 95% Wilson CI |",
        "|---|---|---:|---:|---:|",
    ]
    for item in summaries:
        low, high = item["success_rate_wilson_95ci"]
        condition = input_condition(item, compact=True)
        rows.append(
            f"| {item['checkpoint_label']} | {condition} | "
            f"{item['successes']}/{item['episodes']} | "
            f"{item['success_rate']:.1%} | [{low:.1%}, {high:.1%}] |"
        )
    rows.extend([
        "",
        "Both checkpoints use seed 7, the same task/variation and episode order, "
        "one policy replan per environment step, and at most 25 policy steps.",
        "",
        "Each run directory contains ordinary rollout videos, continuous 2 Hz videos, "
        "raw attention captures, attention videos for every episode (including failures), "
        "first/middle/final attention PNGs, logs, and a machine-readable summary.",
    ])
    (args.output_root / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
