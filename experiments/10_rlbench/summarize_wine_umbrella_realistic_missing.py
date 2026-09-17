"""Summarize paired robustness rollouts for stack_wine and umbrella_out."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_wine_umbrella_realistic_missing_20260917"
RENDERED = OUTPUT / "rendered"
SEVERITIES = (0.0, 0.25, 0.50, 0.75)
TASKS = {
    "stack_wine": {"title": "Stack wine", "target": "wine_bottle"},
    "take_umbrella_out_of_umbrella_stand": {
        "title": "Umbrella out",
        "target": "umbrella",
    },
}
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(CJK_FONT).exists():
    font_manager.fontManager.addfont(CJK_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=CJK_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return max(0.0, center-half), min(1.0, center+half)


def task_report(task_name: str) -> tuple[dict, list[np.ndarray]]:
    task_dir = OUTPUT / task_name
    records = {
        severity: read_jsonl(task_dir / f"severity_{severity:.2f}" / "episode_results.jsonl")
        for severity in SEVERITIES
    }
    baseline = records[0.0]
    report = {
        "task": task_name,
        "target_object": TASKS[task_name]["target"],
        "episodes_per_severity": 10,
        "seed": 7,
        "severity_definition": "fraction of visible target-object pixels assigned a sensor failure",
        "severities": {},
        "pairing_validation": {},
    }
    drift_sets = []
    baseline_predicted = np.asarray(
        [item["first_predicted_position_world"] for item in baseline], dtype=float
    )
    baseline_pose = np.asarray(
        [item["initial_scene"]["target_pose"] for item in baseline], dtype=float
    )
    for severity in SEVERITIES:
        items = records[severity]
        successes = sum(bool(item["success"]) for item in items)
        low, high = wilson(successes, len(items))
        predicted = np.asarray(
            [item["first_predicted_position_world"] for item in items], dtype=float
        )
        drift = np.linalg.norm(predicted-baseline_predicted, axis=1)
        drift_sets.append(100.0*drift)
        stats = [item["first_corruption"] for item in items]
        report["severities"][str(severity)] = {
            "successes": successes,
            "episodes": len(items),
            "success_rate": successes/len(items),
            "success_rate_wilson_95ci": [low, high],
            "first_action_position_drift_m": {
                "mean": float(drift.mean()),
                "std": float(drift.std()),
                "median": float(np.median(drift)),
                "max": float(drift.max()),
                "per_episode": drift.tolist(),
            },
            "first_frame_corruption_mean": {
                key: float(np.mean([stat[key] for stat in stats]))
                for key in (
                    "correct_target_fraction",
                    "target_invalid_fraction",
                    "target_wrong_depth_fraction",
                    "valid_xyz_fraction",
                    "raw_point_center_shift_m",
                    "target_pixels",
                )
            },
        }
        pose = np.asarray(
            [item["initial_scene"]["target_pose"] for item in items], dtype=float
        )
        report["pairing_validation"][str(severity)] = {
            "target_pose_max_abs_difference_vs_clean": float(
                np.max(np.abs(pose-baseline_pose))
            ),
            "episode_seeds_identical": [item["episode_seed"] for item in items]
            == [item["episode_seed"] for item in baseline],
        }
    return report, drift_sets


def render_rollout_summary(report: dict, drift_by_task: dict[str, list[np.ndarray]]) -> None:
    rates = 100*np.asarray(SEVERITIES)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    colors = {"stack_wine": "#8c564b", "take_umbrella_out_of_umbrella_stand": "#1f77b4"}
    for task_name, meta in TASKS.items():
        task = report["tasks"][task_name]
        success = np.asarray([
            task["severities"][str(s)]["success_rate"] for s in SEVERITIES
        ])
        correct = np.asarray([
            task["severities"][str(s)]["first_frame_corruption_mean"]["correct_target_fraction"]
            for s in SEVERITIES
        ])
        drift_mean = np.asarray([
            task["severities"][str(s)]["first_action_position_drift_m"]["mean"]
            for s in SEVERITIES
        ])
        valid = np.asarray([
            task["severities"][str(s)]["first_frame_corruption_mean"]["valid_xyz_fraction"]
            for s in SEVERITIES
        ])
        color = colors[task_name]
        axes[0, 0].plot(rates, 100*success, marker="o", lw=2, color=color, label=meta["title"])
        for x, y, s in zip(rates, 100*success, SEVERITIES):
            n = task["severities"][str(s)]["successes"]
            axes[0, 0].text(x, y+2, f"{n}/10", ha="center", color=color, fontsize=8)
        axes[0, 1].plot(rates, 100*drift_mean, marker="o", lw=2, color=color, label=meta["title"])
        axes[1, 0].plot(rates, 100*correct, marker="o", lw=2, color=color, label=meta["title"])
        axes[1, 1].plot(rates, 100*valid, marker="o", lw=2, color=color, label=meta["title"])

    axes[0, 0].set(title="Rollout success (10 trials each)", ylabel="success (%)", ylim=(-3, 112))
    axes[0, 1].set(title="Mean first predicted-position drift vs clean", ylabel="cm")
    axes[1, 0].set(title="Correct target surface remaining", ylabel="% of visible target")
    axes[1, 1].set(title="Valid front-camera XYZ points", ylabel="% of all pixels")
    for ax in axes.ravel():
        ax.set_xlabel("simulated target failure severity (%)")
        ax.set_xticks(rates)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("PointACT robustness: stack_wine and umbrella_out")
    fig.savefig(RENDERED / "success_and_action_drift_comparison.png", dpi=220)
    plt.close(fig)


def render_corruption_examples(task_name: str) -> None:
    samples = [
        np.load(OUTPUT / task_name / f"severity_{s:.2f}" / "episode_000_initial.npz")
        for s in SEVERITIES
    ]
    cmap = ListedColormap(["#00000000", "#d73027", "#4575b4", "#fdae61", "#984ea3"])
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for col, (severity, sample) in enumerate(zip(SEVERITIES, samples)):
        rgb = sample["rgb"]
        target = sample["target_mask"].astype(bool)
        labels = sample["corruption_labels"]
        overlay = np.zeros((*target.shape, 4), dtype=float)
        overlay[target] = (0.0, 0.95, 0.95, .48)
        axes[0, col].imshow(rgb)
        axes[0, col].imshow(overlay)
        axes[0, col].set_title(f"{severity:.0%}: target mask ({target.sum():,} px)")
        axes[1, col].imshow(rgb, alpha=.45)
        axes[1, col].imshow(np.ma.masked_where(labels == 0, labels), cmap=cmap, vmin=0, vmax=4, alpha=.88)
        axes[1, col].set_title("red=missing, blue=leakage, orange=range error")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle(f"{TASKS[task_name]['title']}: target mask and structured point-cloud failure")
    fig.savefig(RENDERED / f"{task_name}_corruption_examples.png", dpi=220)
    plt.close(fig)


def main() -> None:
    RENDERED.mkdir(parents=True, exist_ok=True)
    combined = {
        "checkpoint": "pointact-rlbench-job25521/checkpoint-40000",
        "protocol": {
            "severities": SEVERITIES,
            "episodes_per_severity": 10,
            "paired_episode_seeds": True,
            "rgb_instruction_robot_state_unchanged": True,
        },
        "tasks": {},
    }
    drift_by_task = {}
    for task_name in TASKS:
        task, drift = task_report(task_name)
        combined["tasks"][task_name] = task
        drift_by_task[task_name] = drift
        render_corruption_examples(task_name)
    render_rollout_summary(combined, drift_by_task)
    (OUTPUT / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
