"""Summarize paired old/new state-action attention experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


METRICS = (
    "state_to_state_mass",
    "state_to_action_mass",
    "state_to_point_mass",
    "state_to_point_entropy",
    "state_to_point_top1pct",
    "action_to_state_mass",
    "action_to_action_mass",
    "action_to_point_mass",
    "action_to_point_entropy",
    "action_to_point_top1pct",
    "state_action_point_cosine",
    "state_action_point_js",
    "state_action_feature_cosine",
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def mean(values) -> float:
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = list(values)
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def median(values) -> float:
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = list(values)
    return float(np.median(np.asarray(values, dtype=np.float64)))


def rotation_distance(left: list[int], right: list[int]) -> float:
    left_rotation = Rotation.from_euler("xyz", np.deg2rad(np.asarray(left) * 5))
    right_rotation = Rotation.from_euler("xyz", np.deg2rad(np.asarray(right) * 5))
    return float(np.rad2deg((left_rotation * right_rotation.inv()).magnitude()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir
    records = {
        (checkpoint, cloud): load_jsonl(root / f"{checkpoint}_{cloud}.jsonl")
        for checkpoint in ("old", "new")
        for cloud in ("complete", "incomplete")
    }

    layer_rows: list[dict] = []
    stage_rows: list[dict] = []
    for cloud in ("complete", "incomplete"):
        by_checkpoint_layer: dict[tuple[str, str], dict[str, list[float]]] = {}
        for checkpoint in ("old", "new"):
            values = defaultdict(lambda: defaultdict(list))
            for record in records[(checkpoint, cloud)]:
                for layer in record["variants"]["pour_full"]["attention"]:
                    for metric in METRICS:
                        values[layer["layer"]][metric].append(layer[metric])
            for layer_name, layer_values in values.items():
                by_checkpoint_layer[(checkpoint, layer_name)] = layer_values
                row = {"cloud_kind": cloud, "checkpoint": checkpoint, "layer": layer_name}
                row.update({metric: mean(layer_values[metric]) for metric in METRICS})
                layer_rows.append(row)

        for stage in range(5):
            old_layers = [
                key[1]
                for key in by_checkpoint_layer
                if key[0] == "old" and key[1].startswith(f"enc.enc{stage}.")
            ]
            for checkpoint in ("old", "new"):
                row = {"cloud_kind": cloud, "checkpoint": checkpoint, "stage": stage}
                for metric in METRICS:
                    row[metric] = mean(
                        value
                        for layer in old_layers
                        for value in by_checkpoint_layer[(checkpoint, layer)][metric]
                    )
                stage_rows.append(row)

    with (root / "layer_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=layer_rows[0].keys())
        writer.writeheader()
        writer.writerows(layer_rows)
    with (root / "stage_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=stage_rows[0].keys())
        writer.writeheader()
        writer.writerows(stage_rows)

    def final_values(checkpoint: str, cloud: str, metric: str) -> np.ndarray:
        return np.asarray(
            [
                record["variants"]["pour_full"]["attention"][-1][metric]
                for record in records[(checkpoint, cloud)]
            ]
        )

    final_layer = {}
    for cloud in ("complete", "incomplete"):
        final_layer[cloud] = {}
        for checkpoint in ("old", "new"):
            final_layer[cloud][checkpoint] = {
                metric: mean(final_values(checkpoint, cloud, metric))
                for metric in METRICS
            }
        final_layer[cloud]["new_minus_old"] = {
            metric: mean(
                final_values("new", cloud, metric)
                - final_values("old", cloud, metric)
            )
            for metric in METRICS
        }

    robustness = {}
    for checkpoint in ("old", "new"):
        complete = records[(checkpoint, "complete")]
        incomplete = records[(checkpoint, "incomplete")]
        complete_by_episode = {record["episode"]: record for record in complete}
        incomplete_by_episode = {record["episode"]: record for record in incomplete}
        episodes = sorted(complete_by_episode)
        complete_full = [
            complete_by_episode[episode]["variants"]["pour_full"] for episode in episodes
        ]
        incomplete_full = [
            incomplete_by_episode[episode]["variants"]["pour_full"] for episode in episodes
        ]
        robustness[checkpoint] = {
            "same_rotation_bins_rate": mean(
                left["pred_rotation_bins"] == right["pred_rotation_bins"]
                for left, right in zip(complete_full, incomplete_full)
            ),
            "same_open_class_rate": mean(
                (left["open_probability"] > 0.5)
                == (right["open_probability"] > 0.5)
                for left, right in zip(complete_full, incomplete_full)
            ),
            "final_attention_incomplete_minus_complete": {
                metric: mean(
                    final_values(checkpoint, "incomplete", metric)
                    - final_values(checkpoint, "complete", metric)
                )
                for metric in METRICS
            },
            "final_attention_mean_absolute_change": {
                metric: mean(
                    np.abs(
                        final_values(checkpoint, "incomplete", metric)
                        - final_values(checkpoint, "complete", metric)
                    )
                )
                for metric in METRICS
            },
        }

    causal = {}
    variants = (
        "previous_full",
        "pour_full",
        "previous_state_current_points",
        "current_state_previous_points",
        "pour_no_state_token",
    )
    for checkpoint in ("old", "new"):
        current = records[(checkpoint, "incomplete")]
        causal[checkpoint] = {"variants": {}}
        for variant in variants:
            values = [record["variants"][variant] for record in current]
            causal[checkpoint]["variants"][variant] = {
                "open_class_rate": mean(value["open_probability"] > 0.5 for value in values),
                "mean_open_probability": mean(value["open_probability"] for value in values),
                "median_target_rotation_error_deg": median(
                    value["target_rotation_error_deg"] for value in values
                ),
            }
        full = [record["variants"]["pour_full"] for record in current]
        for variant in variants[2:]:
            values = [record["variants"][variant] for record in current]
            causal[checkpoint][variant + "_vs_full"] = {
                "same_open_class_rate": mean(
                    (left["open_probability"] > 0.5)
                    == (right["open_probability"] > 0.5)
                    for left, right in zip(full, values)
                ),
                "same_rotation_bins_rate": mean(
                    left["pred_rotation_bins"] == right["pred_rotation_bins"]
                    for left, right in zip(full, values)
                ),
                "median_rotation_change_deg": median(
                    rotation_distance(left["pred_rotation_bins"], right["pred_rotation_bins"])
                    for left, right in zip(full, values)
                ),
                "median_position_change_m": median(
                    np.linalg.norm(
                        np.asarray(left["action"][:3]) - np.asarray(right["action"][:3])
                    )
                    for left, right in zip(full, values)
                ),
            }

        permuted_records = load_jsonl(root / f"{checkpoint}_incomplete_permuted_state.jsonl")
        full_by_episode = {
            record["episode"]: record["variants"]["pour_full"] for record in current
        }
        permuted_by_episode = {
            record["episode"]: record["variants"]["other_pour_state_current_points"]
            for record in permuted_records
        }
        episodes = sorted(full_by_episode)
        causal[checkpoint]["other_episode_pour_state_vs_full"] = {
            "open_class_rate": mean(
                permuted_by_episode[episode]["open_probability"] > 0.5
                for episode in episodes
            ),
            "same_open_class_rate": mean(
                (full_by_episode[episode]["open_probability"] > 0.5)
                == (permuted_by_episode[episode]["open_probability"] > 0.5)
                for episode in episodes
            ),
            "same_rotation_bins_rate": mean(
                full_by_episode[episode]["pred_rotation_bins"]
                == permuted_by_episode[episode]["pred_rotation_bins"]
                for episode in episodes
            ),
            "median_rotation_change_deg": median(
                rotation_distance(
                    full_by_episode[episode]["pred_rotation_bins"],
                    permuted_by_episode[episode]["pred_rotation_bins"],
                )
                for episode in episodes
            ),
            "mean_rotation_change_deg": mean(
                rotation_distance(
                    full_by_episode[episode]["pred_rotation_bins"],
                    permuted_by_episode[episode]["pred_rotation_bins"],
                )
                for episode in episodes
            ),
            "median_position_change_m": median(
                np.linalg.norm(
                    np.asarray(full_by_episode[episode]["action"][:3])
                    - np.asarray(permuted_by_episode[episode]["action"][:3])
                )
                for episode in episodes
            ),
        }

    output = {
        "num_episodes": 100,
        "final_layer": final_layer,
        "robustness_complete_to_incomplete": robustness,
        "causal_incomplete": causal,
    }
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_metrics = (
        ("state_to_point_mass", "state query → points (mass)"),
        ("action_to_state_mass", "action query → state (mass)"),
        ("action_to_point_mass", "action query → points (mass)"),
        ("action_to_point_entropy", "action → point entropy"),
    )
    stages = np.arange(5)
    for axis, (metric, title) in zip(axes.flat, plot_metrics):
        for checkpoint, marker in (("old", "o"), ("new", "s")):
            subset = [
                row
                for row in stage_rows
                if row["cloud_kind"] == "incomplete" and row["checkpoint"] == checkpoint
            ]
            subset.sort(key=lambda row: row["stage"])
            axis.plot(stages, [row[metric] for row in subset], marker=marker, label=checkpoint)
        axis.set_title(title)
        axis.set_xlabel("PTV3 encoder stage")
        axis.set_xticks(stages)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Water plants: old vs new checkpoint, 25% incomplete clouds (100 episodes)")
    fig.savefig(root / "attention_stage_comparison.png", dpi=180)


if __name__ == "__main__":
    main()
