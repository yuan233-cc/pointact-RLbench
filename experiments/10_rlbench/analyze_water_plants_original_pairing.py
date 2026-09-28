"""Audit scene pairing and the reproducible episode-0 watering geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def quat_error_degrees(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    cosine = np.clip(
        abs(float(first @ second)) / (np.linalg.norm(first) * np.linalg.norm(second)),
        0.0,
        1.0,
    )
    return float(np.degrees(2.0 * np.arccos(cosine)))


def initial_world_cloud(run_dir, episode):
    summary = json.loads((run_dir / "summary.json").read_text())
    detail = summary["episodes_detail"][episode]
    capture = np.load(
        run_dir
        / "attention_captures"
        / f"capture_{detail['attention_capture_start']:06d}.npz"
    )
    if "corruption_full_coordinates_world" in capture:
        return capture["corruption_full_coordinates_world"].astype(np.float64)
    return (
        capture["input_coordinates"].astype(np.float64)
        + capture["scene_center"].astype(np.float64)
    )


def first_grasp_snapshot(records):
    for record in records:
        if record["after"]["waterer_grasped"]:
            return record["step"], record["after"]
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--captured-actions-root", type=Path, required=True)
    parser.add_argument("--old-replay", type=Path, required=True)
    parser.add_argument("--new-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    old_run = args.source_root / "baseline_job25521_complete"
    new_run = args.source_root / "incomplete25_clf_concerto_matched"
    cloud_rows = []
    for episode in range(20):
        old_cloud = initial_world_cloud(old_run, episode)
        new_cloud = initial_world_cloud(new_run, episode)
        old_to_new = cKDTree(new_cloud).query(old_cloud, k=1, workers=-1)[0]
        new_to_old = cKDTree(old_cloud).query(new_cloud, k=1, workers=-1)[0]
        row = {
            "episode": episode,
            "symmetric_median_nn_cm": float(
                50.0 * (np.median(old_to_new) + np.median(new_to_old))
            ),
            "symmetric_p95_nn_cm": float(
                50.0
                * (np.percentile(old_to_new, 95) + np.percentile(new_to_old, 95))
            ),
            "centroid_delta_cm": float(
                100.0 * np.linalg.norm(old_cloud.mean(0) - new_cloud.mean(0))
            ),
        }
        row["scene_matched"] = (
            row["symmetric_median_nn_cm"] < 0.1
            and row["symmetric_p95_nn_cm"] < 0.1
        )
        cloud_rows.append(row)

    old_actions = json.loads(
        (args.captured_actions_root / "old_complete_captured_actions.json").read_text()
    )["episodes"][0]
    new_actions = json.loads(
        (
            args.captured_actions_root
            / "new_incomplete25_matched_captured_actions.json"
        ).read_text()
    )["episodes"][0]
    old_records = [
        item
        for item in read_jsonl(args.old_replay / "water_plants_geometry_steps.jsonl")
        if item["episode"] == 0
    ]
    new_records = [
        item
        for item in read_jsonl(args.new_replay / "water_plants_geometry_steps.jsonl")
        if item["episode"] == 0
    ]
    old_episode = read_jsonl(args.old_replay / "water_plants_geometry_episodes.jsonl")[0]
    new_episode = read_jsonl(args.new_replay / "water_plants_geometry_episodes.jsonl")[0]
    old_grasp_step, old_grasp = first_grasp_snapshot(old_records)
    new_grasp_step, new_grasp = first_grasp_snapshot(new_records)
    old_grasp_pose = old_grasp["waterer_pose_in_eef_frame"]
    new_grasp_pose = new_grasp["waterer_pose_in_eef_frame"]

    action_deltas = []
    for step, (old_action, new_action) in enumerate(
        zip(old_actions["actions"], new_actions["actions"])
    ):
        old_action = np.asarray(old_action)
        new_action = np.asarray(new_action)
        action_deltas.append(
            {
                "step": step,
                "target_position_delta_cm": float(
                    100.0 * np.linalg.norm(old_action[:3] - new_action[:3])
                ),
                "target_rotation_delta_deg": quat_error_degrees(
                    old_action[3:7], new_action[3:7]
                ),
                "old_target_position": old_action[:3].tolist(),
                "new_target_position": new_action[:3].tolist(),
            }
        )

    original_old_summary = json.loads((old_run / "summary.json").read_text())
    original_new_summary = json.loads((new_run / "summary.json").read_text())
    original_transitions = []
    for old, new in zip(
        original_old_summary["episodes_detail"],
        original_new_summary["episodes_detail"],
    ):
        if old["success"] and new["success"]:
            transition = "both_success"
        elif not old["success"] and new["success"]:
            transition = "old_fail_new_success"
        elif old["success"] and not new["success"]:
            transition = "old_success_new_fail"
        else:
            transition = "both_fail"
        original_transitions.append(
            {
                "episode": old["episode"],
                "transition": transition,
                "scene_matched": cloud_rows[old["episode"]]["scene_matched"],
            }
        )

    result = {
        "scene_pairing": {
            "criterion": "symmetric median and p95 nearest-neighbor distances both < 0.1 cm",
            "matched_episodes": [row["episode"] for row in cloud_rows if row["scene_matched"]],
            "mismatched_episodes": [row["episode"] for row in cloud_rows if not row["scene_matched"]],
            "per_episode": cloud_rows,
        },
        "original_success_transitions": original_transitions,
        "matched_old_fail_new_success_episodes": [
            item["episode"]
            for item in original_transitions
            if item["scene_matched"] and item["transition"] == "old_fail_new_success"
        ],
        "episode_0": {
            "source_old_success": old_actions["source_success"],
            "source_new_success": new_actions["source_success"],
            "replay_old_success": old_episode["replay_success"],
            "replay_new_success": new_episode["replay_success"],
            "old_min_head_to_pour_center_cm": 100.0
            * old_episode["min_head_to_pour_center_m"],
            "new_min_head_to_pour_center_cm": 100.0
            * new_episode["min_head_to_pour_center_m"],
            "old_pour_sensor_triggered": old_episode["pour_sensor_triggered"],
            "new_pour_sensor_triggered": new_episode["pour_sensor_triggered"],
            "old_max_drops_detected": old_episode["max_drops_detected"],
            "new_max_drops_detected": new_episode["max_drops_detected"],
            "old_grasp_step": old_grasp_step,
            "new_grasp_step": new_grasp_step,
            "grasp_translation_delta_cm": float(
                100.0
                * np.linalg.norm(
                    np.asarray(old_grasp_pose[:3]) - np.asarray(new_grasp_pose[:3])
                )
            ),
            "grasp_rotation_delta_deg": quat_error_degrees(
                old_grasp_pose[3:7], new_grasp_pose[3:7]
            ),
            "action_target_deltas": action_deltas,
        },
        "interpretation": (
            "Episode 0 is the only scene-matched old-fail/new-success sample. "
            "Its grasp transforms are nearly identical; post-grasp translation "
            "targets place the new checkpoint's spout near the pour sensor."
        ),
    }
    (args.output_dir / "pairing_and_episode0_geometry.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (args.output_dir / "scene_pairing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cloud_rows[0]))
        writer.writeheader()
        writer.writerows(cloud_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    episodes = [row["episode"] for row in cloud_rows]
    medians = [row["symmetric_median_nn_cm"] for row in cloud_rows]
    axes[0].bar(
        episodes,
        medians,
        color=["#2ca02c" if row["scene_matched"] else "#d62728" for row in cloud_rows],
    )
    axes[0].axhline(0.1, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("symmetric median NN distance (cm)")
    axes[0].set_title("Old/new initial-scene pairing")

    steps = [item["step"] for item in action_deltas]
    axes[1].bar(
        steps,
        [item["target_position_delta_cm"] for item in action_deltas],
        color="#1f77b4",
        label="target position delta",
    )
    axes[1].set_xlabel("policy step")
    axes[1].set_ylabel("old/new target delta (cm)")
    axes[1].set_title("Episode 0: divergence starts after grasp")
    axes[1].set_xticks(steps)
    fig.tight_layout()
    fig.savefig(args.output_dir / "pairing_and_episode0_geometry.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
