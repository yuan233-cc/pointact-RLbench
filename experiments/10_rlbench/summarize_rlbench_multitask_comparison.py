"""Combine per-task paired checkpoint summaries into one report."""

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
    parser.add_argument(
        "--task-run",
        action="append",
        required=True,
        help="TASK=task-level-output-directory",
    )
    parser.add_argument("--baseline-label", default="baseline_job25521_complete")
    parser.add_argument(
        "--incomplete-label", default="incomplete25_clf_concerto_matched"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = (args.baseline_label, args.incomplete_label)
    task_roots = {}
    for value in args.task_run:
        task, separator, path = value.partition("=")
        if not separator or not task or not path:
            raise ValueError(f"Invalid --task-run value: {value}")
        task_roots[task] = Path(path)

    tasks = {}
    for task, root in task_roots.items():
        task_summaries = {}
        for label in labels:
            summary = json.loads((root / label / "summary.json").read_text())
            if summary["task"] != task:
                raise ValueError(f"{root}: expected {task}, got {summary['task']}")
            task_summaries[label] = summary
        tasks[task] = task_summaries

    macro = {
        label: float(np.mean([tasks[task][label]["success_rate"] for task in tasks]))
        for label in labels
    }
    report = {
        "protocol": {
            "episodes_per_task_per_checkpoint": 20,
            "variation": 0,
            "seed": 7,
            "baseline_input": "complete point cloud",
            "incomplete25_input": (
                "training-matched 25% episode-consistent structured missingness "
                "with an unseen evaluation-only random seed"
            ),
        },
        "tasks": tasks,
        "macro_average_success_rate": macro,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "comparison_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    task_names = list(tasks)
    x = np.arange(len(task_names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for offset, label, color in (
        (-width / 2, labels[0], "#5276A7"),
        (width / 2, labels[1], "#D97A43"),
    ):
        values = [tasks[task][label]["success_rate"] for task in task_names]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, task in zip(bars, task_names):
            item = tasks[task][label]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{item['successes']}/{item['episodes']}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x, task_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("RLBench success rate")
    ax.set_title("Two-task checkpoint comparison — 20 episodes per condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_root / "success_rate_comparison.png", dpi=180)
    plt.close(fig)

    rows = [
        "# RLBench two-task checkpoint comparison",
        "",
        "| Task | Checkpoint | Point-cloud input | Successes | Success rate | 95% Wilson CI |",
        "|---|---|---|---:|---:|---:|",
    ]
    for task in task_names:
        for label in labels:
            item = tasks[task][label]
            low, high = item["success_rate_wilson_95ci"]
            condition = (
                "complete"
                if label == labels[0]
                else "25% episode-consistent missing (unseen evaluation seed)"
            )
            rows.append(
                f"| {task} | {label} | {condition} | "
                f"{item['successes']}/{item['episodes']} | "
                f"{item['success_rate']:.1%} | [{low:.1%}, {high:.1%}] |"
            )
    rows.extend([
        "",
        "Macro-average success rate:",
        "",
        f"- `{labels[0]}`: {macro[labels[0]]:.1%}",
        f"- `{labels[1]}`: {macro[labels[1]]:.1%}",
        "",
        "Every condition uses seed 7, variation 0, 20 episodes, one replan per "
        "environment step, and at most 25 policy steps.",
    ])
    (args.output_root / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
