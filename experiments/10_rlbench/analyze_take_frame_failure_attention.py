"""Summarize paired take-frame failures and semantic attention locations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


REGIONS = ("frame", "hanger", "gripper", "target_table", "wall", "other")
CHECKPOINTS = ("old_complete", "new_incomplete25")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def points_in_obb(points: np.ndarray, geometry: dict, padding: float) -> np.ndarray:
    pose = np.asarray(geometry["pose_world"], dtype=np.float64)
    bounds = np.asarray(geometry["bbox_local"], dtype=np.float64)
    local = Rotation.from_quat(pose[3:7]).inv().apply(points - pose[:3])
    lower = bounds[[0, 2, 4]] - padding
    upper = bounds[[1, 3, 5]] + padding
    return np.all((local >= lower) & (local <= upper), axis=1)


def semantic_masks(points: np.ndarray, snapshot: dict) -> dict[str, np.ndarray]:
    raw = {
        "frame": points_in_obb(points, snapshot["frame"], 0.015),
        "hanger": points_in_obb(points, snapshot["hanger"], 0.025),
        "gripper": np.linalg.norm(
            points - np.asarray(snapshot["eef_pose_world"][:3]), axis=1
        ) <= 0.075,
        "target_table": points_in_obb(points, snapshot["success_sensor"], 0.025),
        "wall": points_in_obb(points, snapshot["wall"], 0.015),
    }
    assigned = np.zeros(len(points), dtype=bool)
    masks = {}
    for name in REGIONS[:-1]:
        masks[name] = raw[name] & ~assigned
        assigned |= masks[name]
    masks["other"] = ~assigned
    return masks


def stage_last_prefixes(keys: list[str]) -> list[str]:
    pattern = re.compile(r"(action_attention_stage(\d+)_block(\d+))_point_coordinates$")
    stages: dict[int, tuple[int, str]] = {}
    for key in keys:
        match = pattern.fullmatch(key)
        if match:
            prefix, stage, block = match.group(1), int(match.group(2)), int(match.group(3))
            if stage not in stages or block > stages[stage][0]:
                stages[stage] = (block, prefix)
    return [stages[stage][1] for stage in sorted(stages)]


def phase(step: int) -> str:
    if step == 0:
        return "initial"
    if step <= 3:
        return "early_manipulation"
    return "late_retry"


def mean_or_none(values):
    values = list(values)
    return None if not values else float(np.mean(values))


def median_or_none(values):
    values = list(values)
    return None if not values else float(np.median(values))


def quaternion_error_degrees(first, second) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    cosine = abs(float(np.dot(first, second))) / (np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.degrees(2 * np.arccos(np.clip(cosine, 0.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    episode_rows = {}
    step_rows = {}
    summary = {"checkpoints": {}, "paired_outcomes": {}}
    for checkpoint in CHECKPOINTS:
        eval_dir = root / checkpoint / "eval"
        episodes = read_jsonl(eval_dir / "take_frame_geometry_episodes.jsonl")
        steps = read_jsonl(eval_dir / "take_frame_geometry_steps.jsonl")
        episode_rows[checkpoint] = {int(row["episode"]): row for row in episodes}
        step_rows[checkpoint] = {
            (int(row["episode"]), int(row["step"])): row for row in steps
        }
        counts = Counter(row["failure_category"] for row in episodes)
        summary["checkpoints"][checkpoint] = {
            "successes": int(sum(row["success"] for row in episodes)),
            "episodes": len(episodes),
            "success_rate": mean_or_none(row["success"] for row in episodes),
            "outcome_counts": dict(sorted(counts.items())),
            "mean_policy_steps": mean_or_none(row["policy_steps"] for row in episodes),
        }

    paired = Counter()
    common = sorted(set(episode_rows[CHECKPOINTS[0]]) & set(episode_rows[CHECKPOINTS[1]]))
    for episode in common:
        old = episode_rows[CHECKPOINTS[0]][episode]["success"]
        new = episode_rows[CHECKPOINTS[1]][episode]["success"]
        paired[f"old_{'success' if old else 'failure'}__new_{'success' if new else 'failure'}"] += 1
    summary["paired_outcomes"] = dict(paired)

    for checkpoint in CHECKPOINTS:
        errors = [row for row in step_rows[checkpoint].values() if row.get("error")]
        summary["checkpoints"][checkpoint]["planner_error_details"] = [
            {
                "episode": row["episode"],
                "step": row["step"],
                "position_jump_m": float(np.linalg.norm(
                    np.asarray(row["commanded_action"][:3])
                    - np.asarray(row["observed_eef_before"][:3])
                )),
                "rotation_jump_deg": quaternion_error_degrees(
                    row["commanded_action"][3:7], row["observed_eef_before"][3:7]
                ),
            }
            for row in errors
        ]

    attention_rows = []
    for checkpoint in CHECKPOINTS:
        capture_dir = root / checkpoint / "attention"
        manifest = read_jsonl(capture_dir / "manifest.jsonl")
        for item in manifest:
            episode = int(item["episode_id"])
            request = int(item["request_in_episode"])
            geometry = step_rows[checkpoint].get((episode, request))
            if geometry is None:
                continue
            outcome = episode_rows[checkpoint][episode]
            with np.load(capture_dir / item["file"]) as data:
                center = np.asarray(data["scene_center"]).reshape(-1, 3)[0]
                for prefix in stage_last_prefixes(data.files):
                    stage = int(re.search(r"stage(\d+)", prefix).group(1))
                    world = np.asarray(data[f"{prefix}_point_coordinates"]) + center
                    masks = semantic_masks(world, geometry["before"])
                    query_arrays = {"action": np.asarray(data[f"{prefix}_point_weights"])}
                    state_key = f"{prefix}_state_point_weights"
                    if state_key in data:
                        query_arrays["state"] = np.asarray(data[state_key])
                    for query, weights in query_arrays.items():
                        total = float(weights.sum())
                        if total <= 0:
                            continue
                        top_count = max(1, int(np.ceil(0.05 * len(weights))))
                        top = np.zeros(len(weights), dtype=bool)
                        top[np.argpartition(weights, -top_count)[-top_count:]] = True
                        for region in REGIONS:
                            mask = masks[region]
                            point_fraction = float(mask.mean())
                            attention_fraction = float(weights[mask].sum() / total)
                            attention_rows.append(
                                {
                                    "checkpoint": checkpoint,
                                    "episode": episode,
                                    "success": bool(outcome["success"]),
                                    "failure_category": outcome["failure_category"],
                                    "request": request,
                                    "phase": phase(request),
                                    "stage": stage,
                                    "query": query,
                                    "region": region,
                                    "point_fraction": point_fraction,
                                    "attention_fraction": attention_fraction,
                                    "enrichment": (
                                        attention_fraction / point_fraction
                                        if point_fraction > 0 else None
                                    ),
                                    "top5_point_fraction": float((top & mask).sum() / top_count),
                                    "point_attention_mass": total,
                                }
                            )

    csv_path = root / "attention_semantic_rows.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attention_rows[0]))
        writer.writeheader()
        writer.writerows(attention_rows)

    grouped = defaultdict(list)
    for row in attention_rows:
        outcome = "success" if row["success"] else "failure"
        key = (
            row["checkpoint"], outcome, row["phase"], row["stage"],
            row["query"], row["region"],
        )
        grouped[key].append(row["attention_fraction"])
    aggregate = []
    for key, values in sorted(grouped.items()):
        checkpoint, outcome, phase_name, stage, query, region = key
        aggregate.append(
            {
                "checkpoint": checkpoint,
                "outcome": outcome,
                "phase": phase_name,
                "stage": stage,
                "query": query,
                "region": region,
                "mean_attention_fraction": float(np.mean(values)),
                "std_attention_fraction": float(np.std(values)),
                "n": len(values),
            }
        )
    with (root / "attention_semantic_aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    summary["attention_captures"] = {
        checkpoint: sum(
            1 for row in attention_rows
            if row["checkpoint"] == checkpoint and row["region"] == "frame"
            and row["query"] == "action" and row["stage"] == 0
        )
        for checkpoint in CHECKPOINTS
    }

    initial_geometry = defaultdict(list)
    for checkpoint in CHECKPOINTS:
        capture_dir = root / checkpoint / "attention"
        for item in read_jsonl(capture_dir / "manifest.jsonl"):
            if int(item["request_in_episode"]) != 0:
                continue
            episode = int(item["episode_id"])
            episode_result = episode_rows[checkpoint][episode]
            outcome = "success" if episode_result["success"] else "failure"
            snapshot = step_rows[checkpoint][(episode, 0)]["before"]
            with np.load(capture_dir / item["file"]) as data:
                prefix = [p for p in stage_last_prefixes(data.files) if "stage0_" in p][0]
                world = np.asarray(data[f"{prefix}_point_coordinates"]) + np.asarray(
                    data["scene_center"]
                ).reshape(-1, 3)[0]
                frame_points = int(semantic_masks(world, snapshot)["frame"].sum())
                record = {"frame_input_points": frame_points}
                if "corruption_full_coordinates_world" in data:
                    full = np.asarray(data["corruption_full_coordinates_world"])
                    removed = np.asarray(data["corruption_removed_coordinates_world"])
                    full_frame = int(points_in_obb(full, snapshot["frame"], 0.015).sum())
                    removed_frame = int(points_in_obb(removed, snapshot["frame"], 0.015).sum())
                    record.update(
                        {
                            "frame_points_before_corruption": full_frame,
                            "frame_points_removed": removed_frame,
                            "frame_removal_fraction": removed_frame / max(1, full_frame),
                        }
                    )
                initial_geometry[(checkpoint, outcome)].append(record)
                if checkpoint == "new_incomplete25":
                    old_success = episode_rows["old_complete"][episode]["success"]
                    pair_label = (
                        f"old_{'success' if old_success else 'failure'}__"
                        f"new_{'success' if episode_result['success'] else 'failure'}"
                    )
                    initial_geometry[(checkpoint, pair_label)].append(record)
    summary["initial_frame_geometry"] = {}
    for (checkpoint, outcome), records in sorted(initial_geometry.items()):
        key = f"{checkpoint}__{outcome}"
        summary["initial_frame_geometry"][key] = {
            "n": len(records),
            "mean_frame_input_points": mean_or_none(r["frame_input_points"] for r in records),
            "median_frame_input_points": median_or_none(r["frame_input_points"] for r in records),
        }
        if "frame_removal_fraction" in records[0]:
            summary["initial_frame_geometry"][key].update(
                {
                    "mean_frame_removal_fraction": mean_or_none(
                        r["frame_removal_fraction"] for r in records
                    ),
                    "median_frame_removal_fraction": median_or_none(
                        r["frame_removal_fraction"] for r in records
                    ),
                }
            )

    last_stage = max(row["stage"] for row in attention_rows)
    summary["initial_last_stage_attention"] = {}
    for checkpoint in CHECKPOINTS:
        for outcome in ("success", "failure"):
            for query in ("action", "state"):
                matches = [
                    row for row in attention_rows
                    if row["checkpoint"] == checkpoint and row["request"] == 0
                    and row["stage"] == last_stage and row["query"] == query
                    and ("success" if row["success"] else "failure") == outcome
                ]
                summary["initial_last_stage_attention"][
                    f"{checkpoint}__{outcome}__{query}"
                ] = {
                    region: mean_or_none(
                        row["attention_fraction"] for row in matches if row["region"] == region
                    )
                    for region in REGIONS
                }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    categories = sorted(
        set().union(*(summary["checkpoints"][c]["outcome_counts"] for c in CHECKPOINTS))
    )
    x = np.arange(len(categories))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(9, len(categories) * 1.6), 5))
    for i, checkpoint in enumerate(CHECKPOINTS):
        counts = summary["checkpoints"][checkpoint]["outcome_counts"]
        ax.bar(x + (i - 0.5) * width, [counts.get(c, 0) for c in categories], width, label=checkpoint)
    ax.set_xticks(x, [c.replace("_", "\n") for c in categories])
    ax.set_ylabel("episodes")
    ax.set_title("Take frame off hanger: outcome categories (paired scenes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "failure_categories.png", dpi=180)
    plt.close(fig)

    # Last encoder stage, early task phase: spatial destination of each query.
    early = [
        row for row in aggregate
        if row["stage"] == last_stage and row["phase"] in {"initial", "early_manipulation"}
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ax, (checkpoint, query) in zip(
        axes.flat,
        [(c, q) for c in CHECKPOINTS for q in ("action", "state")],
    ):
        for i, outcome in enumerate(("success", "failure")):
            values = []
            for region in REGIONS:
                matches = [
                    row["mean_attention_fraction"] for row in early
                    if row["checkpoint"] == checkpoint and row["query"] == query
                    and row["outcome"] == outcome and row["region"] == region
                ]
                values.append(float(np.mean(matches)) if matches else 0.0)
            ax.bar(x=np.arange(len(REGIONS)) + (i - 0.5) * width, height=values,
                   width=width, label=outcome)
        ax.set_title(f"{checkpoint} · {query} query")
        ax.set_xticks(np.arange(len(REGIONS)), [r.replace("_", "\n") for r in REGIONS])
        ax.set_ylabel("attention fraction")
        ax.legend()
    fig.suptitle(f"Semantic attention location, encoder stage {last_stage}, early phases")
    fig.tight_layout()
    fig.savefig(root / "attention_regions_stage_last.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
