"""Summarize and visualize structured close_fridge depth corruption."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_close_fridge_realistic_missing_20260917"
RENDERED = OUTPUT / "rendered"
SEVERITIES = (0.0, 0.25, 0.50, 0.75)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return max(0.0, center-half), min(1.0, center+half)


def render_examples():
    samples = [np.load(OUTPUT / f"severity_{s:.2f}" / "episode_000_initial.npz") for s in SEVERITIES]
    displacements = []
    for sample in samples:
        clean, corrupt = sample["clean_points"], sample["corrupted_points"]
        displacements.append(np.linalg.norm(corrupt-clean, axis=-1))
    finite_values = np.concatenate([x[np.isfinite(x)] for x in displacements])
    upper = max(float(np.percentile(finite_values, 99)), 1e-4)
    cmap = ListedColormap(["#00000000", "#d73027", "#4575b4", "#fdae61", "#984ea3"])
    labels = ["clean", "target invalid", "background leakage", "range distortion", "other hole"]

    fig, axes = plt.subplots(4, 3, figsize=(13, 17), constrained_layout=True)
    for row, (severity, sample, displacement) in enumerate(zip(SEVERITIES, samples, displacements)):
        rgb = sample["rgb"]
        target = sample["target_mask"]
        corruption = sample["corruption_labels"]
        axes[row, 0].imshow(rgb)
        axes[row, 0].imshow(np.ma.masked_where(~target, target), cmap=ListedColormap(["#00ffff"]), alpha=.38)
        axes[row, 0].set_title(f"severity {severity:.0%}: fridge target mask")
        axes[row, 1].imshow(rgb, alpha=.40)
        axes[row, 1].imshow(np.ma.masked_where(corruption == 0, corruption), cmap=cmap, vmin=0, vmax=4, alpha=.82)
        axes[row, 1].set_title("categorical simulated sensor failure")
        view = displacement.copy()
        view[corruption == 1] = np.nan
        im = axes[row, 2].imshow(view, cmap="magma", vmin=0, vmax=upper)
        axes[row, 2].imshow(np.ma.masked_where(corruption != 1, corruption == 1),
                            cmap=ListedColormap(["#00ffff"]), alpha=.85)
        axes[row, 2].set_title("XYZ displacement; cyan = missing")
        for ax in axes[row]:
            ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(i), label=labels[i]) for i in range(1, 5)]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.colorbar(im, ax=axes[:, 2], fraction=.02, pad=.01, label="3-D point displacement (m), p99 clip")
    fig.suptitle("close_fridge: transparent-object-like structured depth corruption", fontsize=16)
    fig.savefig(RENDERED / "corruption_examples.png", dpi=220)
    plt.close(fig)


def main():
    RENDERED.mkdir(parents=True, exist_ok=True)
    records = {s: read_jsonl(OUTPUT / f"severity_{s:.2f}" / "episode_results.jsonl") for s in SEVERITIES}
    baseline = records[0.0]
    report = {
        "task": "close_fridge", "episodes_per_severity": 10, "seed": 7,
        "severity_definition": "fraction of visible fridge pixels assigned a simulated sensor failure",
        "severities": {}, "pairing_validation": {},
    }
    drift_sets = []
    for severity in SEVERITIES:
        items = records[severity]
        successes = sum(bool(x["success"]) for x in items)
        low, high = wilson(successes, len(items))
        predicted = np.asarray([x["first_predicted_position_world"] for x in items])
        baseline_predicted = np.asarray([x["first_predicted_position_world"] for x in baseline])
        drift = np.linalg.norm(predicted-baseline_predicted, axis=1)
        drift_sets.append(100*drift)
        stats = [x["first_corruption"] for x in items]
        metric_names = (
            "correct_target_fraction", "target_invalid_fraction",
            "target_wrong_depth_fraction", "valid_xyz_fraction",
            "raw_point_center_shift_m",
        )
        zero_missing_defaults = {
            "target_invalid_fraction": 0.0,
            "target_wrong_depth_fraction": 0.0,
        }
        report["severities"][str(severity)] = {
            "successes": successes, "episodes": len(items),
            "success_rate": successes/len(items),
            "success_rate_wilson_95ci": [low, high],
            "first_action_position_drift_m": {
                "mean": float(drift.mean()), "std": float(drift.std()),
                "median": float(np.median(drift)), "max": float(drift.max()),
                "per_episode": drift.tolist(),
            },
            "first_frame_corruption_mean": {
                name: float(np.mean([
                    x.get(name, zero_missing_defaults[name])
                    if name in zero_missing_defaults else x[name]
                    for x in stats
                ]))
                for name in metric_names
            },
        }

    clean_pose = np.asarray([x["initial_scene"]["fridge_root_pose"] for x in baseline])
    clean_joint = np.asarray([x["initial_scene"]["top_joint_position"] for x in baseline])
    for severity in SEVERITIES:
        pose = np.asarray([x["initial_scene"]["fridge_root_pose"] for x in records[severity]])
        joint = np.asarray([x["initial_scene"]["top_joint_position"] for x in records[severity]])
        report["pairing_validation"][str(severity)] = {
            "fridge_root_pose_max_abs_difference": float(np.max(np.abs(pose-clean_pose))),
            "top_joint_max_abs_difference": float(np.max(np.abs(joint-clean_joint))),
        }

    rates = 100*np.asarray(SEVERITIES)
    success = np.asarray([report["severities"][str(x)]["success_rate"] for x in SEVERITIES])
    ci = np.asarray([report["severities"][str(x)]["success_rate_wilson_95ci"] for x in SEVERITIES])
    correct = 100*np.asarray([
        report["severities"][str(x)]["first_frame_corruption_mean"]["correct_target_fraction"]
        for x in SEVERITIES
    ])
    center_shift = 100*np.asarray([
        report["severities"][str(x)]["first_frame_corruption_mean"]["raw_point_center_shift_m"]
        for x in SEVERITIES
    ])
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].errorbar(rates, 100*success,
                        yerr=np.maximum(
                            0.0,
                            np.vstack((100*(success-ci[:, 0]), 100*(ci[:, 1]-success))),
                        ),
                        marker="o", capsize=5, lw=2)
    for x, y, severity in zip(rates, 100*success, SEVERITIES):
        item = report["severities"][str(severity)]
        axes[0, 0].text(x, y+1.5, f"{item['successes']}/10", ha="center")
    axes[0, 0].set(title="Rollout success (95% Wilson CI)", ylabel="success (%)", ylim=(0, 112))
    axes[0, 1].boxplot(drift_sets, positions=rates, widths=12, showmeans=True)
    axes[0, 1].set(title="First predicted position drift vs clean", ylabel="cm")
    axes[1, 0].plot(rates, correct, marker="o", lw=2)
    axes[1, 0].set(title="Correct fridge-surface pixels remaining", ylabel="% of visible fridge")
    axes[1, 1].plot(rates, center_shift, marker="o", lw=2)
    axes[1, 1].set(title="Raw point-cloud center shift", ylabel="cm")
    for ax in axes.ravel():
        ax.set_xlabel("simulated target failure severity (%)")
        ax.set_xticks(rates, [f"{x:.0f}" for x in rates])
        ax.grid(alpha=.25)
    fig.suptitle("close_fridge under structured transparent-depth failure")
    fig.savefig(RENDERED / "success_feature_input_and_action_drift.png", dpi=220)
    plt.close(fig)
    render_examples()
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
