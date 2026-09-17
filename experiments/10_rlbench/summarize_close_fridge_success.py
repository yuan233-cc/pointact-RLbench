"""Summarize close_fridge rollout success and first-action position drift."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_close_fridge_incomplete_study_20260916"
EVAL = OUTPUT / "success_eval"
RENDERED = OUTPUT / "rendered"
RATES = (0.0, 0.25, 0.50, 0.75)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(successes, total, z=1.959963984540054):
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0.0, center-half), min(1.0, center+half)


def first_predictions(rate):
    directory = EVAL / f"missing_{rate:.2f}"
    episodes = read_jsonl(directory / "episode_results.jsonl")
    requests = read_jsonl(directory / "proxy.jsonl")
    result, offset = [], 0
    for episode in episodes:
        steps = int(episode["policy_steps"])
        if offset >= len(requests):
            raise RuntimeError(f"Missing request for rate={rate}, episode={episode['episode']}")
        result.append(np.asarray(requests[offset]["first_predicted_position_world"], dtype=float))
        offset += steps
    if offset != len(requests):
        raise RuntimeError(f"Request count mismatch for rate={rate}: used {offset}, have {len(requests)}")
    return episodes, np.asarray(result)


def main():
    RENDERED.mkdir(parents=True, exist_ok=True)
    baseline_episodes, baseline_positions = first_predictions(0.0)
    report = {
        "task": "close_fridge", "variation": 0, "episodes_per_rate": 20,
        "seed": 7, "checkpoint_selected_camera": "front",
        "rates": {},
    }
    drift_sets = []
    for rate in RATES:
        episodes, positions = first_predictions(rate)
        successes = sum(bool(item["success"]) for item in episodes)
        low, high = wilson(successes, len(episodes))
        drift = np.linalg.norm(positions - baseline_positions, axis=1)
        drift_sets.append(100 * drift)
        report["rates"][str(rate)] = {
            "successes": successes, "episodes": len(episodes),
            "success_rate": successes / len(episodes),
            "success_rate_wilson_95ci": [low, high],
            "first_predicted_position_drift_m": {
                "mean": float(drift.mean()), "std": float(drift.std()),
                "median": float(np.median(drift)), "p95": float(np.percentile(drift, 95)),
                "max": float(drift.max()), "per_episode": drift.tolist(),
            },
        }

    rates_percent = 100*np.asarray(RATES)
    success = np.asarray([report["rates"][str(x)]["success_rate"] for x in RATES])
    ci = np.asarray([report["rates"][str(x)]["success_rate_wilson_95ci"] for x in RATES])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].errorbar(rates_percent, 100*success,
                     yerr=np.vstack((100*(success-ci[:, 0]), 100*(ci[:, 1]-success))),
                     marker="o", lw=2, capsize=5)
    for x, y, item in zip(rates_percent, 100*success, report["rates"].values()):
        axes[0].text(x, y+1.5, f"{item['successes']}/20", ha="center")
    axes[0].set(xlabel="Requested missing front-camera XYZ pixels (%)",
                ylabel="Rollout success rate (%)", title="Success (95% Wilson CI)", ylim=(0, 110))
    axes[0].grid(alpha=.25)

    axes[1].boxplot(drift_sets, positions=rates_percent, widths=12, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"})
    axes[1].set(xlabel="Requested missing front-camera XYZ pixels (%)",
                ylabel="First predicted position drift vs clean (cm)",
                title="Per-episode first-action drift")
    axes[1].set_xticks(rates_percent, [f"{x:.0f}" for x in rates_percent])
    axes[1].grid(alpha=.25)
    fig.suptitle("close_fridge point-cloud incompleteness: task outcome and action drift")
    fig.savefig(RENDERED / "success_rate_and_rollout_drift.png", dpi=220)
    plt.close(fig)
    (OUTPUT / "success_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
