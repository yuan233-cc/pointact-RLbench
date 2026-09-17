"""Analyze PTV3 feature and action-attention robustness to point dropout."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_incomplete_pointcloud_study_20260916"
CAPTURE_DIR = OUTPUT / "captures"
SOURCE_SCENE = ROOT / "PTV3_reach_target_action_attention_20260916" / "scene"
RENDERED = OUTPUT / "rendered"

OBJECT_NAMES = ("target", "distractor0", "distractor1")
OBJECT_LABELS = ("red target", "green distractor 0", "green distractor 1")
OBJECT_COLORS = ("#e41a1c", "#4daf4a", "#1b7837")
LAST_BLOCK = {0: 2, 1: 2, 2: 2, 3: 11, 4: 2}


def normalize_rows(array: np.ndarray) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def map_input_to_scene(capture, scene):
    points_world = capture["input_coordinates"] + capture["scene_center"]
    raw = scene["world_points"].reshape(-1, 3)
    valid = np.isfinite(raw).all(axis=1)
    distance, compact_index = cKDTree(raw[valid]).query(points_world)
    flat_index = np.flatnonzero(valid)[compact_index]
    rows, cols = np.unravel_index(flat_index, scene["instance_mask"].shape)
    instance_ids = scene["instance_mask"].reshape(-1)[flat_index]
    return points_world, rows, cols, instance_ids, distance


def memberships(instance_ids: np.ndarray, metadata: dict) -> np.ndarray:
    result = np.zeros((len(instance_ids), 3), dtype=np.float64)
    objects = {item["name"]: item for item in metadata["objects"]}
    for index, name in enumerate(OBJECT_NAMES):
        result[:, index] = np.isin(instance_ids, objects[name]["handles"])
    return result


def stage_assignment(capture, stage: int) -> np.ndarray:
    if stage == 0:
        return np.arange(len(capture["input_coordinates"]), dtype=np.int64)
    return capture[f"input_to_stage{stage}"].astype(np.int64)


def object_attention(capture, membership: np.ndarray, stage: int, block: int) -> dict:
    prefix = f"action_attention_stage{stage}_block{block}"
    weight = capture[f"{prefix}_point_weights"].astype(np.float64)
    assignment = stage_assignment(capture, stage)
    sizes = np.bincount(assignment, minlength=len(weight)).astype(np.float64)
    fraction = np.zeros((len(weight), 3), dtype=np.float64)
    for column in range(3):
        fraction[:, column] = np.bincount(
            assignment, weights=membership[:, column], minlength=len(weight)
        ) / np.maximum(sizes, 1.0)
    mass = weight @ fraction
    total = float(mass.sum())
    return {
        "point_attention_mass": float(weight.sum()),
        "ball_attention_mass": float(total),
        "object_attention_mass": {
            name: float(value) for name, value in zip(OBJECT_NAMES, mass)
        },
        "share_among_visible_balls": {
            name: float(value / total) if total else 0.0
            for name, value in zip(OBJECT_NAMES, mass)
        },
    }


def matched_feature_metrics(clean, current) -> dict:
    clean_xyz = clean["coordinates"] + clean["scene_center"]
    current_xyz = current["coordinates"] + current["scene_center"]
    distance = np.linalg.norm(
        clean_xyz[:, None, :] - current_xyz[None, :, :], axis=-1
    )
    clean_index, current_index = linear_sum_assignment(distance)
    clean_feature = normalize_rows(clean["features"].astype(np.float32))[clean_index]
    current_feature = normalize_rows(current["features"].astype(np.float32))[current_index]
    cosine = np.sum(clean_feature * current_feature, axis=-1)

    clean_action = normalize_rows(clean["action_features"].astype(np.float32)[0])
    current_action = normalize_rows(current["action_features"].astype(np.float32)[0])
    action_cosine = clean_action @ current_action.T

    clean_attention = clean["action_attention_stage4_block2_point_weights"]
    current_attention = current["action_attention_stage4_block2_point_weights"]
    matched_clean_attention = clean_attention[clean_index]
    matched_current_attention = current_attention[current_index]
    attention_cosine = float(
        normalize_rows(matched_clean_attention[None])[0]
        @ normalize_rows(matched_current_attention[None])[0]
    )
    clean_world_position = clean["predicted_position"] + clean["scene_center"]
    current_world_position = current["predicted_position"] + current["scene_center"]
    return {
        "matched_final_feature_cosine_mean": float(cosine.mean()),
        "matched_final_feature_cosine_median": float(np.median(cosine)),
        "matched_final_feature_cosine_min": float(cosine.min()),
        "matched_final_token_distance_m_mean": float(distance[clean_index, current_index].mean()),
        "matched_final_token_distance_m_max": float(distance[clean_index, current_index].max()),
        "robot_state_token_cosine": float(action_cosine[0, 0]),
        "action_query_token_cosine": float(action_cosine[1, 1]),
        "final_block_attention_cosine": attention_cosine,
        "predicted_position_shift_m": float(
            np.linalg.norm(current_world_position - clean_world_position)
        ),
        "predicted_position_world": current_world_position.astype(float).tolist(),
        "predicted_position_centered": current["predicted_position"].astype(float).tolist(),
    }


def aggregate(trial_metrics: list[dict]) -> dict:
    grouped = defaultdict(list)
    for trial in trial_metrics:
        grouped[trial["missing_rate_requested"]].append(trial)
    fields = (
        "actual_input_keep_fraction",
        "matched_final_feature_cosine_mean",
        "action_query_token_cosine",
        "final_block_attention_cosine",
        "predicted_position_shift_m",
    )
    output = {}
    for rate, trials in sorted(grouped.items()):
        item = {"num_trials": len(trials)}
        for field in fields:
            values = np.asarray([x[field] for x in trials], dtype=np.float64)
            item[field] = {"mean": float(values.mean()), "std": float(values.std())}
        for name in OBJECT_NAMES:
            for quantity in ("stage4_object_attention_mass", "stage4_ball_share"):
                values = np.asarray([x[quantity][name] for x in trials])
                item[f"{quantity}_{name}"] = {
                    "mean": float(values.mean()), "std": float(values.std())
                }
        output[str(rate)] = item
    return output


def add_boxes(ax, rows, cols, membership):
    for index, (label, color) in enumerate(zip(OBJECT_LABELS, OBJECT_COLORS)):
        selected = membership[:, index] > 0
        if not selected.any():
            continue
        x0, x1 = cols[selected].min(), cols[selected].max()
        y0, y1 = rows[selected].min(), rows[selected].max()
        ax.add_patch(Rectangle(
            (x0 - 4, y0 - 4), x1 - x0 + 8, y1 - y0 + 8,
            fill=False, edgecolor=color, linewidth=1.7,
        ))
        ax.text(x0 - 4, y0 - 6, label, fontsize=6, color="white",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2})


def representative_trials(trials):
    # Clean plus repeat 0 at each nonzero rate.
    return [
        trial for trial in trials
        if trial["missing_rate_requested"] == 0.0 or trial["repeat"] == 0
    ]


def render_feature_attention(captures, trials, scene, metadata):
    selected = representative_trials(trials)
    selected_captures = [captures[x["capture_index"]] for x in selected]
    feature_matrix = np.concatenate(
        [capture["features"].astype(np.float32) for capture in captures], axis=0
    )
    pca = PCA(n_components=3, random_state=7).fit(feature_matrix)
    all_pca = pca.transform(feature_matrix)
    low, high = np.percentile(all_pca, [1, 99], axis=0)
    scale = np.maximum(high - low, 1e-8)

    propagated_attention = []
    maps = []
    for capture in selected_captures:
        _xyz, rows, cols, ids, _distance = map_input_to_scene(capture, scene)
        membership = memberships(ids, metadata)
        weight = capture["action_attention_stage4_block2_point_weights"]
        propagated_attention.append(weight[capture["input_to_final"]])
        maps.append((rows, cols, membership))
    attention_high = float(np.percentile(np.concatenate(propagated_attention), 99))

    fig, axes = plt.subplots(2, len(selected), figsize=(4 * len(selected), 8),
                             constrained_layout=True)
    for column, (trial, capture, mapping, attention) in enumerate(
        zip(selected, selected_captures, maps, propagated_attention)
    ):
        rows, cols, membership = mapping
        transformed = pca.transform(capture["features"].astype(np.float32))
        colors = np.clip((transformed - low) / scale, 0, 1)
        dense_colors = colors[capture["input_to_final"]]

        axes[0, column].imshow(scene["rgb"], alpha=0.18)
        axes[0, column].scatter(cols, rows, c=dense_colors, s=7, linewidths=0)
        add_boxes(axes[0, column], rows, cols, membership)
        axes[0, column].set_title(
            f"{100*trial['missing_rate_requested']:.0f}% missing\n"
            f"shared PCA of final 768-D features"
        )
        axes[0, column].axis("off")

        axes[1, column].imshow(scene["rgb"], alpha=0.18)
        scatter = axes[1, column].scatter(
            cols, rows, c=attention, cmap="turbo", vmin=0,
            vmax=max(attention_high, 1e-12), s=7, linewidths=0,
        )
        add_boxes(axes[1, column], rows, cols, membership)
        axes[1, column].set_title(
            f"Stage 4 Block 2 direct attention\n"
            f"actual PTV3 input kept: {100*len(capture['input_coordinates'])/len(captures[0]['input_coordinates']):.1f}%"
        )
        axes[1, column].axis("off")
    fig.colorbar(scatter, ax=axes[1, :], fraction=0.015, pad=0.01,
                 label="absolute action→point attention (shared scale, p99 clip)")
    fig.suptitle("Point-cloud incompleteness: PTV3 feature and action attention")
    fig.savefig(RENDERED / "feature_and_attention_by_missing_rate.png", dpi=220)
    plt.close(fig)


def render_robustness_curves(aggregated):
    rates = np.asarray(sorted(float(key) for key in aggregated))
    items = [aggregated[str(rate)] for rate in rates]
    panels = (
        ("actual_input_keep_fraction", "Actual PTV3 input retained", "%", 100),
        ("matched_final_feature_cosine_mean", "Matched final-feature cosine", "cosine", 1),
        ("action_query_token_cosine", "Action-query token cosine", "cosine", 1),
        ("predicted_position_shift_m", "Predicted position shift", "cm", 100),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (field, title, ylabel, factor) in zip(axes.ravel(), panels):
        means = factor * np.asarray([x[field]["mean"] for x in items])
        stds = factor * np.asarray([x[field]["std"] for x in items])
        ax.errorbar(100 * rates, means, yerr=stds, marker="o", linewidth=2,
                    capsize=4, color="#4c72b0")
        ax.set_xlabel("Requested missing image-point pixels (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("Robustness metrics (mean ± std over 3 fixed masks; clean n=1)")
    fig.savefig(RENDERED / "robustness_metrics.png", dpi=220)
    plt.close(fig)


def render_ball_attention(aggregated):
    rates = np.asarray(sorted(float(key) for key in aggregated))
    items = [aggregated[str(rate)] for rate in rates]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for name, label, color in zip(OBJECT_NAMES, OBJECT_LABELS, OBJECT_COLORS):
        mass = np.asarray([x[f"stage4_object_attention_mass_{name}"]["mean"] for x in items])
        mass_std = np.asarray([x[f"stage4_object_attention_mass_{name}"]["std"] for x in items])
        share = np.asarray([x[f"stage4_ball_share_{name}"]["mean"] for x in items])
        share_std = np.asarray([x[f"stage4_ball_share_{name}"]["std"] for x in items])
        axes[0].errorbar(100*rates, 100*mass, yerr=100*mass_std, marker="o",
                         capsize=4, label=label, color=color)
        axes[1].errorbar(100*rates, 100*share, yerr=100*share_std, marker="o",
                         capsize=4, label=label, color=color)
    axes[0].set_title("Absolute Stage-4 Block-2 attention")
    axes[0].set_ylabel("% of all attention keys")
    axes[1].set_title("Share among the three visible balls")
    axes[1].set_ylabel("% among balls (renormalized)")
    for ax in axes:
        ax.set_xlabel("Requested missing image-point pixels (%)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Effect of point-cloud dropout on action attention (mean ± std)")
    fig.savefig(RENDERED / "ball_attention_vs_missing_rate.png", dpi=220)
    plt.close(fig)


def render_global_attention_all_stages(captures, trials, scene):
    """Full-scene direct attention at every encoder stage, with no object filtering."""
    selected = representative_trials(trials)
    selected_captures = [captures[item["capture_index"]] for item in selected]
    mappings = [map_input_to_scene(capture, scene) for capture in selected_captures]
    figure, axes = plt.subplots(
        5, len(selected), figsize=(4 * len(selected), 16), constrained_layout=True
    )
    for stage, block in LAST_BLOCK.items():
        propagated = []
        for capture in selected_captures:
            values = capture[f"action_attention_stage{stage}_block{block}_point_weights"]
            propagated.append(values[stage_assignment(capture, stage)])
        upper = float(np.percentile(np.concatenate(propagated), 99))
        lower = 0.0
        row_scatter = None
        for column, (trial, capture, mapping, values) in enumerate(
            zip(selected, selected_captures, mappings, propagated)
        ):
            _xyz, rows, cols, _ids, _distance = mapping
            ax = axes[stage, column]
            ax.imshow(scene["rgb"], alpha=0.12)
            row_scatter = ax.scatter(
                cols, rows, c=values, cmap="turbo", vmin=lower,
                vmax=max(upper, 1e-12), s=6, linewidths=0,
            )
            if stage == 0:
                ax.set_title(
                    f"requested missing {100*trial['missing_rate_requested']:.0f}%\n"
                    f"actual PTV3 input kept "
                    f"{100*len(capture['input_coordinates'])/len(captures[0]['input_coordinates']):.1f}%"
                )
            if column == 0:
                ax.set_ylabel(f"Stage {stage} Block {block}\nfull scene")
            ax.set_xticks([])
            ax.set_yticks([])
        figure.colorbar(
            row_scatter, ax=axes[stage, :], fraction=0.012, pad=0.006,
            label=f"S{stage} absolute direct attention (row-shared p99 clip)",
        )
    figure.suptitle(
        "Global action-query → point-key attention under point-cloud dropout\n"
        "All robot, table, background, and object points are shown; no ball filtering",
        fontsize=16,
    )
    figure.savefig(RENDERED / "global_attention_all_stages.png", dpi=220)
    plt.close(figure)


def main() -> None:
    RENDERED.mkdir(parents=True, exist_ok=True)
    study = json.loads((OUTPUT / "trials.json").read_text(encoding="utf-8"))
    metadata = json.loads((SOURCE_SCENE / "metadata.json").read_text(encoding="utf-8"))
    scene = np.load(SOURCE_SCENE / "observation.npz")
    captures = [
        np.load(CAPTURE_DIR / f"capture_{index:06d}.npz")
        for index in range(len(study["trials"]))
    ]
    clean = captures[0]
    clean_count = len(clean["input_coordinates"])
    trial_metrics = []
    for trial, capture in zip(study["trials"], captures):
        _xyz, _rows, _cols, ids, nearest_distance = map_input_to_scene(capture, scene)
        membership = memberships(ids, metadata)
        metric = {
            **trial,
            "actual_input_points": int(len(capture["input_coordinates"])),
            "actual_input_keep_fraction": len(capture["input_coordinates"]) / clean_count,
            "final_token_count": int(len(capture["coordinates"])),
            "object_input_points": {
                name: int(membership[:, index].sum())
                for index, name in enumerate(OBJECT_NAMES)
            },
            "scene_match_distance_m_max": float(nearest_distance.max()),
            **matched_feature_metrics(clean, capture),
        }
        attention_by_stage = {
            str(stage): object_attention(capture, membership, stage, block)
            for stage, block in LAST_BLOCK.items()
        }
        metric["last_block_attention_by_stage"] = attention_by_stage
        metric["stage4_object_attention_mass"] = attention_by_stage["4"]["object_attention_mass"]
        metric["stage4_ball_share"] = attention_by_stage["4"]["share_among_visible_balls"]
        trial_metrics.append(metric)

    aggregated = aggregate(trial_metrics)
    render_feature_attention(captures, study["trials"], scene, metadata)
    render_robustness_curves(aggregated)
    render_ball_attention(aggregated)
    render_global_attention_all_stages(captures, study["trials"], scene)

    validation_cosines = []
    for capture in captures:
        validation_cosines.extend([
            float(capture[key]) for key in capture.files
            if key.endswith("_reconstruction_cosine")
        ])
    report = {
        "design": {
            "missing_rates": study["missing_rates"],
            "repeat_seeds": study["repeat_seeds"],
            "rgb_instruction_robot_state_unchanged": True,
            "actions_executed": 0,
            "dropout_masks_nested_within_repeat": True,
            "note": study["corruption"],
        },
        "metric_definitions": {
            "matched_final_feature_cosine": (
                "Hungarian one-to-one spatial matching of final PTV3 tokens to the "
                "clean run, followed by cosine similarity of the matched 768-D features."
            ),
            "action_query_token_cosine": (
                "Cosine between the final PTV3 action-query embedding and the clean run."
            ),
            "predicted_position_shift_m": (
                "Euclidean shift of the policy position after restoring each run's "
                "point-cloud scene center; reported in world-coordinate metres."
            ),
            "object_attention": (
                "Direct action-query to point-key softmax attention. At pooled stages, "
                "token mass is apportioned by its input-point descendant fractions."
            ),
        },
        "attention_reconstruction_minimum_cosine": float(min(validation_cosines)),
        "trials": trial_metrics,
        "aggregate_by_missing_rate": aggregated,
        "caveats": [
            "Pixel dropout is random sparsity, not structured object occlusion or depth bias.",
            "Voxel downsampling makes actual PTV3 point loss smaller than requested pixel loss.",
            "Only one scene is used; three masks measure mask sensitivity, not task-level success rate.",
            "reach_target is outside this checkpoint's hybridvla_10tasks training set.",
            "Attention is a direct layer diagnostic, not causal importance or attention rollout.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
