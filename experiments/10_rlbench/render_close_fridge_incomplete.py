"""Render close_fridge PTV3 feature/attention robustness diagnostics."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_close_fridge_incomplete_study_20260916"
CAPTURES = OUTPUT / "captures"
RENDERED = OUTPUT / "rendered"
CAMERAS = ("left_shoulder", "right_shoulder", "wrist", "front")
LAST_BLOCK = {0: 2, 1: 2, 2: 2, 3: 11, 4: 2}


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def stage_assignment(capture, stage):
    if stage == 0:
        return np.arange(len(capture["input_coordinates"]), dtype=np.int64)
    return capture[f"input_to_stage{stage}"].astype(np.int64)


def representative(trials):
    return [x for x in trials if x["missing_rate_requested"] == 0 or x["repeat"] == 0]


def match_metrics(clean, current):
    a = clean["coordinates"] + clean["scene_center"]
    b = current["coordinates"] + current["scene_center"]
    distances = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
    ia, ib = linear_sum_assignment(distances)
    cosine = np.sum(unit(clean["features"].astype(np.float32))[ia] *
                    unit(current["features"].astype(np.float32))[ib], axis=-1)
    aq0 = unit(clean["action_features"].astype(np.float32)[0])[1]
    aq1 = unit(current["action_features"].astype(np.float32)[0])[1]
    p0 = clean["predicted_position"] + clean["scene_center"]
    p1 = current["predicted_position"] + current["scene_center"]
    return {
        "matched_final_feature_cosine_mean": float(cosine.mean()),
        "matched_final_feature_cosine_std": float(cosine.std()),
        "matched_final_token_distance_m_mean": float(distances[ia, ib].mean()),
        "action_query_token_cosine": float(aq0 @ aq1),
        "predicted_position_shift_m": float(np.linalg.norm(p1 - p0)),
        "predicted_position_world": p1.astype(float).tolist(),
    }


def aggregate(metrics):
    groups = defaultdict(list)
    for item in metrics:
        groups[item["missing_rate_requested"]].append(item)
    fields = ("actual_input_keep_fraction", "matched_final_feature_cosine_mean",
              "action_query_token_cosine", "final_block_attention_cosine",
              "predicted_position_shift_m")
    result = {}
    for rate, items in sorted(groups.items()):
        result[str(rate)] = {"num_trials": len(items)}
        for field in fields:
            values = np.asarray([x[field] for x in items])
            result[str(rate)][field] = {"mean": float(values.mean()), "std": float(values.std())}
    return result


def plot_curves(summary):
    rates = np.asarray(sorted(float(x) for x in summary))
    items = [summary[str(x)] for x in rates]
    panels = (
        ("actual_input_keep_fraction", "Actual PTV3 input retained", "%", 100),
        ("matched_final_feature_cosine_mean", "Matched final-feature cosine", "cosine", 1),
        ("action_query_token_cosine", "Action-query feature cosine", "cosine", 1),
        ("predicted_position_shift_m", "First predicted position drift", "cm", 100),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (key, title, ylabel, factor) in zip(axes.ravel(), panels):
        mean = factor * np.asarray([x[key]["mean"] for x in items])
        std = factor * np.asarray([x[key]["std"] for x in items])
        ax.errorbar(100*rates, mean, yerr=std, marker="o", capsize=4, lw=2)
        ax.set(title=title, xlabel="Requested missing XYZ pixels (%)", ylabel=ylabel)
        ax.grid(alpha=.25)
    fig.suptitle("close_fridge: PTV3 robustness to incomplete point clouds (mean ± std)")
    fig.savefig(RENDERED / "robustness_metrics.png", dpi=220)
    plt.close(fig)


def plot_global(captures, trials):
    chosen = representative(trials)
    selected = [captures[x["capture_index"]] for x in chosen]
    fig, axes = plt.subplots(5, 4, figsize=(17, 18), constrained_layout=True)
    for stage, block in LAST_BLOCK.items():
        all_values = []
        for capture in selected:
            weight = capture[f"action_attention_stage{stage}_block{block}_point_weights"]
            all_values.append(weight[stage_assignment(capture, stage)])
        upper = max(float(np.percentile(np.concatenate(all_values), 99)), 1e-12)
        scatter = None
        for column, (trial, capture, values) in enumerate(zip(chosen, selected, all_values)):
            xyz = capture["input_coordinates"] + capture["scene_center"]
            scatter = axes[stage, column].scatter(
                xyz[:, 0], xyz[:, 1], c=values, s=2.2, cmap="turbo", vmin=0, vmax=upper,
                linewidths=0, rasterized=True,
            )
            pred = capture["predicted_position"] + capture["scene_center"]
            axes[stage, column].scatter(pred[0], pred[1], marker="*", s=90,
                                        c="white", edgecolors="black", linewidths=.7)
            axes[stage, column].set_aspect("equal", adjustable="box")
            axes[stage, column].set_xticks([]); axes[stage, column].set_yticks([])
            if stage == 0:
                keep = len(capture["input_coordinates"]) / len(selected[0]["input_coordinates"])
                axes[stage, column].set_title(
                    f"{100*trial['missing_rate_requested']:.0f}% missing\nactual input kept {100*keep:.1f}%"
                )
            if column == 0:
                axes[stage, column].set_ylabel(f"Stage {stage}, block {block}\nworld top view")
        fig.colorbar(scatter, ax=axes[stage, :], fraction=.012, pad=.005,
                     label=f"S{stage} direct attention (row-shared p99)")
    fig.suptitle("close_fridge: global action-query → point-key attention\nAll scene/robot/fridge points; white star = first predicted position")
    fig.savefig(RENDERED / "global_attention_all_stages.png", dpi=220)
    plt.close(fig)


def map_to_camera(capture, scene, camera):
    """Project by nearest observed XYZ separately in each camera.

    Doing this per camera is important: the same surface is visible in several
    views, so a single joint nearest-neighbour search would arbitrarily assign
    almost every voxel to just one camera.
    """
    points = scene[f"{camera}_points"]
    valid = np.isfinite(points).all(axis=-1)
    rows, cols = np.nonzero(valid)
    raw = points[valid]
    world = capture["input_coordinates"] + capture["scene_center"]
    distance, index = cKDTree(raw).query(world)
    return rows[index], cols[index], distance


def plot_camera_views(captures, trials, scene):
    chosen = representative(trials)
    selected = [captures[x["capture_index"]] for x in chosen]
    propagated = []
    for capture in selected:
        weight = capture["action_attention_stage4_block2_point_weights"]
        propagated.append(weight[capture["input_to_stage4"]])
    upper = max(float(np.percentile(np.concatenate(propagated), 99)), 1e-12)
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), constrained_layout=True)
    scatter = None
    for row, camera in enumerate(CAMERAS):
        for col, (trial, capture, values) in enumerate(zip(chosen, selected, propagated)):
            pixel_rows, pixel_cols, distance = map_to_camera(capture, scene, camera)
            # Only show model points that are actually on a surface visible in
            # this view. One centimetre matches the PTV3 voxel scale.
            select = distance <= 0.01
            axes[row, col].imshow(scene[f"{camera}_rgb"], alpha=.18)
            scatter = axes[row, col].scatter(pixel_cols[select], pixel_rows[select],
                                              c=values[select], s=5, cmap="turbo",
                                              vmin=0, vmax=upper, linewidths=0)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"{100*trial['missing_rate_requested']:.0f}% missing")
            if col == 0:
                axes[row, col].text(-.03, .5, camera, rotation=90, va="center", ha="right",
                                    transform=axes[row, col].transAxes, fontsize=11)
    fig.colorbar(scatter, ax=axes, fraction=.012, pad=.006,
                 label="Stage-4 Block-2 direct attention (shared p99)")
    fig.suptitle("close_fridge: front-input point attention reprojected into four camera views")
    fig.savefig(RENDERED / "global_attention_camera_views.png", dpi=220)
    plt.close(fig)


def plot_front_attention(captures, trials, scene):
    chosen = representative(trials)
    selected = [captures[x["capture_index"]] for x in chosen]
    values_all = []
    mappings = []
    for capture in selected:
        weight = capture["action_attention_stage4_block2_point_weights"]
        values_all.append(weight[capture["input_to_stage4"]])
        mappings.append(map_to_camera(capture, scene, "front"))
    upper = max(float(np.percentile(np.concatenate(values_all), 99)), 1e-12)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    scatter = None
    for ax, trial, mapping, values in zip(axes, chosen, mappings, values_all):
        rows, cols, distance = mapping
        visible = distance <= .01
        ax.imshow(scene["front_rgb"], alpha=.18)
        scatter = ax.scatter(cols[visible], rows[visible], c=values[visible], s=7,
                             cmap="turbo", vmin=0, vmax=upper, linewidths=0)
        ax.set_title(f"{100*trial['missing_rate_requested']:.0f}% missing")
        ax.axis("off")
    fig.colorbar(scatter, ax=axes, fraction=.018, pad=.008,
                 label="Stage-4 Block-2 direct attention (shared p99)")
    fig.suptitle("close_fridge: global attention over every retained front-view scene point")
    fig.savefig(RENDERED / "global_attention_front_view.png", dpi=220)
    plt.close(fig)


def plot_front_features(captures, trials, scene):
    chosen = representative(trials)
    selected = [captures[x["capture_index"]] for x in chosen]
    matrix = np.concatenate([x["features"].astype(np.float32) for x in captures])
    pca = PCA(n_components=3, random_state=7).fit(matrix)
    transformed = [pca.transform(x["features"].astype(np.float32)) for x in selected]
    pooled = np.concatenate(transformed)
    low, high = np.percentile(pooled, [1, 99], axis=0)
    scale = np.maximum(high-low, 1e-8)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    for ax, trial, capture, feature in zip(axes, chosen, selected, transformed):
        rows, cols, distance = map_to_camera(capture, scene, "front")
        visible = distance <= .01
        colors = np.clip((feature-low)/scale, 0, 1)[capture["input_to_final"]]
        ax.imshow(scene["front_rgb"], alpha=.18)
        ax.scatter(cols[visible], rows[visible], c=colors[visible], s=7, linewidths=0)
        ax.set_title(f"{100*trial['missing_rate_requested']:.0f}% missing")
        ax.axis("off")
    fig.suptitle(
        "close_fridge: final 768-D PTV3 features (shared PCA→RGB; color similarity is qualitative)"
    )
    fig.savefig(RENDERED / "final_feature_pca_front_view.png", dpi=220)
    plt.close(fig)


def main():
    RENDERED.mkdir(parents=True, exist_ok=True)
    study = json.loads((OUTPUT / "trials.json").read_text())
    scene = np.load(OUTPUT / "scene" / "observation.npz")
    captures = [np.load(CAPTURES / f"capture_{i:06d}.npz") for i in range(len(study["trials"]))]
    clean = captures[0]
    clean_attention = clean["action_attention_stage4_block2_point_weights"]
    metrics = []
    for trial, capture in zip(study["trials"], captures):
        current_attention = capture["action_attention_stage4_block2_point_weights"]
        # Attention vectors differ in length; compare their sorted distributions.
        size = min(len(clean_attention), len(current_attention))
        ac = np.sort(clean_attention)[-size:]
        bc = np.sort(current_attention)[-size:]
        attention_cos = float(unit(ac[None])[0] @ unit(bc[None])[0])
        metrics.append({
            **trial,
            "actual_input_points": int(len(capture["input_coordinates"])),
            "actual_input_keep_fraction": len(capture["input_coordinates"]) / len(clean["input_coordinates"]),
            "final_token_count": int(len(capture["coordinates"])),
            "final_block_attention_cosine": attention_cos,
            **match_metrics(clean, capture),
        })
    grouped = aggregate(metrics)
    plot_curves(grouped)
    plot_global(captures, study["trials"])
    plot_camera_views(captures, study["trials"], scene)
    plot_front_attention(captures, study["trials"], scene)
    plot_front_features(captures, study["trials"], scene)
    report = {
        "task": "close_fridge", "checkpoint_training_task": True,
        "design": {"missing_rates": study["missing_rates"], "repeat_seeds": study["repeat_seeds"],
                   "rgb_instruction_robot_state_unchanged": True,
                   "checkpoint_selected_camera": "front",
                   "front_camera_point_cloud_corrupted": True,
                   "other_camera_images_used_only_for_offline_reprojection": True},
        "trials": metrics, "aggregate_by_missing_rate": grouped,
        "caveats": [
            "Random XYZ pixel dropout models unstructured missing depth, not contiguous occlusion.",
            "Static diagnostics use one reset scene and three fixed nested masks per nonzero rate.",
            "Attention is a direct layer diagnostic, not causal importance.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
