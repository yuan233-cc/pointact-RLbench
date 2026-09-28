"""Summarize paired RLBench water-plants geometry diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = ("old_complete", "new_incomplete25_matched")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quat_error_degrees(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denom = np.linalg.norm(first) * np.linalg.norm(second)
    if denom == 0:
        return None
    cosine = np.clip(abs(float(np.dot(first, second))) / denom, 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(cosine)))


def distribution(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def episode_detail(episode, steps):
    records = sorted(steps, key=lambda item: item["step"])
    reached_records = [
        record
        for record in records
        if record["head_detected_any"] or record["after"]["task_reached_once"]
    ]
    spawned_records = [
        record
        for record in records
        if record["drops_spawned_any"] or record["after"]["drop_count"] > 0
    ]
    episode = {
        **episode,
        "pour_sensor_triggered": bool(reached_records),
        "first_pour_sensor_step": (
            reached_records[0]["step"] if reached_records else None
        ),
        "drops_spawned": bool(spawned_records),
        "first_drops_spawned_step": (
            spawned_records[0]["step"] if spawned_records else None
        ),
        "max_drops_detected": max(
            (record["after"]["num_drops_detected"] for record in records),
            default=0,
        ),
    }
    min_record = min(records, key=lambda item: item["min_head_to_pour_center_m"])
    pour_records = [record for record in records if record["head_detected_any"]]
    first_pour = pour_records[0] if pour_records else None
    grasp_snapshot = None
    grasp_step = None
    for record in records:
        for key in ("after", "before"):
            snapshot = record.get(key)
            if snapshot and snapshot["waterer_grasped"]:
                grasp_snapshot = snapshot
                grasp_step = record["step"]
                break
        if grasp_snapshot is not None:
            break
    min_snapshot = min_record["min_head_to_pour_snapshot"]
    pour_snapshot = (
        first_pour["first_head_detected_snapshot"] if first_pour else None
    )
    if episode["success"] and not episode["pour_sensor_triggered"]:
        outcome = "success_without_pour"
    elif episode["success"]:
        outcome = "success"
    elif not episode["pour_sensor_triggered"]:
        outcome = "no_pour_trigger"
    elif episode["max_drops_detected"] < 5:
        outcome = "poured_but_drops_missed"
    else:
        outcome = "all_drops_transient_but_no_terminal_success"
    return {
        **episode,
        "outcome": outcome,
        "grasp_step": grasp_step,
        "grasp_waterer_pose_in_eef_frame": (
            grasp_snapshot["waterer_pose_in_eef_frame"] if grasp_snapshot else None
        ),
        "min_head_position_in_pour_frame": min_snapshot["head_position_in_pour_frame"],
        "min_head_waterer_pose_world": min_snapshot["waterer_pose_world"],
        "min_head_eef_pose_world": min_snapshot["eef_pose_world"],
        "first_pour_head_position_in_pour_frame": (
            pour_snapshot["head_position_in_pour_frame"] if pour_snapshot else None
        ),
        "first_pour_waterer_pose_world": (
            pour_snapshot["waterer_pose_world"] if pour_snapshot else None
        ),
        "first_pour_eef_pose_world": (
            pour_snapshot["eef_pose_world"] if pour_snapshot else None
        ),
    }


def summarize_checkpoint(details):
    groups = defaultdict(list)
    for detail in details:
        groups[detail["outcome"]].append(detail)
    by_success = {}
    for name, selected in (
        ("success", [item for item in details if item["success"]]),
        ("failure", [item for item in details if not item["success"]]),
    ):
        by_success[name] = {
            "episodes": [item["episode"] for item in selected],
            "min_head_to_pour_center_cm": distribution(
                [100.0 * item["min_head_to_pour_center_m"] for item in selected]
            ),
            "max_drops_detected": distribution(
                [item["max_drops_detected"] for item in selected]
            ),
            "first_pour_sensor_step": distribution(
                [item["first_pour_sensor_step"] for item in selected]
            ),
            "max_eef_target_position_error_cm": distribution(
                [100.0 * item["max_eef_target_position_error_m"] for item in selected]
            ),
            "max_eef_target_rotation_error_deg": distribution(
                [item["max_eef_target_rotation_error_deg"] for item in selected]
            ),
        }
    return {
        "episodes": len(details),
        "successes": sum(item["success"] for item in details),
        "outcome_counts": {name: len(items) for name, items in sorted(groups.items())},
        "outcome_episodes": {
            name: [item["episode"] for item in items]
            for name, items in sorted(groups.items())
        },
        "by_success": by_success,
    }


def paired_detail(old, new):
    if old["success"] and new["success"]:
        transition = "both_success"
    elif not old["success"] and new["success"]:
        transition = "old_fail_new_success"
    elif old["success"] and not new["success"]:
        transition = "old_success_new_fail"
    else:
        transition = "both_fail"

    grasp_translation_delta_cm = None
    grasp_rotation_delta_deg = None
    if (
        old["grasp_waterer_pose_in_eef_frame"] is not None
        and new["grasp_waterer_pose_in_eef_frame"] is not None
    ):
        old_pose = old["grasp_waterer_pose_in_eef_frame"]
        new_pose = new["grasp_waterer_pose_in_eef_frame"]
        grasp_translation_delta_cm = 100.0 * float(
            np.linalg.norm(np.asarray(old_pose[:3]) - np.asarray(new_pose[:3]))
        )
        grasp_rotation_delta_deg = quat_error_degrees(old_pose[3:7], new_pose[3:7])

    pour_orientation_delta_deg = None
    if (
        old["first_pour_waterer_pose_world"] is not None
        and new["first_pour_waterer_pose_world"] is not None
    ):
        pour_orientation_delta_deg = quat_error_degrees(
            old["first_pour_waterer_pose_world"][3:7],
            new["first_pour_waterer_pose_world"][3:7],
        )

    return {
        "episode": old["episode"],
        "transition": transition,
        "old_outcome": old["outcome"],
        "new_outcome": new["outcome"],
        "old_first_pour_step": old["first_pour_sensor_step"],
        "new_first_pour_step": new["first_pour_sensor_step"],
        "old_min_head_to_pour_cm": 100.0 * old["min_head_to_pour_center_m"],
        "new_min_head_to_pour_cm": 100.0 * new["min_head_to_pour_center_m"],
        "old_max_drops_detected": old["max_drops_detected"],
        "new_max_drops_detected": new["max_drops_detected"],
        "grasp_translation_delta_cm": grasp_translation_delta_cm,
        "grasp_rotation_delta_deg": grasp_rotation_delta_deg,
        "first_pour_waterer_orientation_delta_deg": pour_orientation_delta_deg,
        "old_waterer_lost_after_grasp": old["waterer_lost_after_grasp"],
        "new_waterer_lost_after_grasp": new["waterer_lost_after_grasp"],
    }


def make_figure(output_root, checkpoint_summaries, pairs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    outcomes = [
        "success",
        "success_without_pour",
        "no_pour_trigger",
        "poured_but_drops_missed",
        "all_drops_transient_but_no_terminal_success",
    ]
    colors = ["#2ca02c", "#17becf", "#d62728", "#ff7f0e", "#9467bd"]
    bottoms = np.zeros(2)
    for outcome, color in zip(outcomes, colors):
        values = [
            checkpoint_summaries[label]["outcome_counts"].get(outcome, 0)
            for label in LABELS
        ]
        axes[0].bar([0, 1], values, bottom=bottoms, label=outcome, color=color)
        bottoms += values
    axes[0].set_xticks([0, 1], ["old", "new"])
    axes[0].set_ylabel("episodes")
    axes[0].set_title("Failure stage")
    axes[0].legend(fontsize=8)

    for x, label in enumerate(LABELS):
        summary = checkpoint_summaries[label]["by_success"]
        axes[1].bar(
            x - 0.18,
            summary["success"]["min_head_to_pour_center_cm"]["median"] or 0,
            width=0.36,
            color="#2ca02c",
            label="success" if x == 0 else None,
        )
        axes[1].bar(
            x + 0.18,
            summary["failure"]["min_head_to_pour_center_cm"]["median"] or 0,
            width=0.36,
            color="#d62728",
            label="failure" if x == 0 else None,
        )
    axes[1].set_xticks([0, 1], ["old", "new"])
    axes[1].set_ylabel("median distance (cm)")
    axes[1].set_title("Closest head to pour-point center")
    axes[1].legend()

    rescues = [pair for pair in pairs if pair["transition"] == "old_fail_new_success"]
    x = np.arange(len(rescues))
    axes[2].bar(x - 0.18, [item["old_max_drops_detected"] for item in rescues], 0.36, label="old")
    axes[2].bar(x + 0.18, [item["new_max_drops_detected"] for item in rescues], 0.36, label="new")
    axes[2].set_xticks(x, [item["episode"] for item in rescues])
    axes[2].set_ylim(0, 5.5)
    axes[2].set_xlabel("episode")
    axes[2].set_ylabel("max detected drops")
    axes[2].set_title("Same episode index (scene may differ)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_root / "water_plants_geometry_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    details_by_label = {}
    checkpoint_summaries = {}
    for label in LABELS:
        run_dir = args.output_root / label
        episodes = read_jsonl(run_dir / "water_plants_geometry_episodes.jsonl")
        steps = read_jsonl(run_dir / "water_plants_geometry_steps.jsonl")
        steps_by_episode = defaultdict(list)
        for step in steps:
            steps_by_episode[step["episode"]].append(step)
        details = [
            episode_detail(episode, steps_by_episode[episode["episode"]])
            for episode in episodes
        ]
        details_by_label[label] = {item["episode"]: item for item in details}
        checkpoint_summaries[label] = summarize_checkpoint(details)

    episode_ids = sorted(set(details_by_label[LABELS[0]]) & set(details_by_label[LABELS[1]]))
    pairs = [
        paired_detail(
            details_by_label[LABELS[0]][episode_id],
            details_by_label[LABELS[1]][episode_id],
        )
        for episode_id in episode_ids
    ]
    transition_groups = defaultdict(list)
    for pair in pairs:
        transition_groups[pair["transition"]].append(pair)
    summary = {
        "checkpoints": checkpoint_summaries,
        "paired_transition_episodes": {
            name: [item["episode"] for item in items]
            for name, items in sorted(transition_groups.items())
        },
        "old_fail_new_success": {
            "episodes": [
                item["episode"]
                for item in transition_groups["old_fail_new_success"]
            ],
            "grasp_translation_delta_cm": distribution(
                [
                    item["grasp_translation_delta_cm"]
                    for item in transition_groups["old_fail_new_success"]
                ]
            ),
            "grasp_rotation_delta_deg": distribution(
                [
                    item["grasp_rotation_delta_deg"]
                    for item in transition_groups["old_fail_new_success"]
                ]
            ),
            "first_pour_waterer_orientation_delta_deg": distribution(
                [
                    item["first_pour_waterer_orientation_delta_deg"]
                    for item in transition_groups["old_fail_new_success"]
                ]
            ),
        },
        "paired_episodes": pairs,
    }
    (args.output_root / "geometry_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (args.output_root / "paired_geometry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    make_figure(args.output_root, checkpoint_summaries, pairs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
