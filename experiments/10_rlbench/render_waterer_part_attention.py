"""Detailed, part-level action-head and direct-attention analysis of the waterer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_waterer_part_attention_20260916"
NEW_CAPTURE = OUTPUT / "captures" / "capture_000000.npz"
OLD_CAPTURE = (
    ROOT / "PTV3_feature_visualizations_corrected_20260916"
    / "captures" / "capture_000004.npz"
)
SCENE = OUTPUT / "scene" / "observation.npz"
METADATA = OUTPUT / "scene" / "metadata.json"
RENDERED = OUTPUT / "rendered"

PART_NAMES = ("spout", "handle", "top opening/rim", "upper body", "lower body")
PART_COLORS = ("#fdae61", "#d7191c", "#2c7bb6", "#66bd63", "#1a9641")
LAST_BLOCK = {0: 2, 1: 2, 2: 2, 3: 11, 4: 2}


def match_input_to_front(capture, scene):
    world = capture["input_coordinates"] + capture["scene_center"]
    raw = scene["front_points"].reshape(-1, 3)
    valid = np.isfinite(raw).all(axis=1)
    distance, compact = cKDTree(raw[valid]).query(world)
    flat = np.flatnonzero(valid)[compact]
    rows, cols = np.unravel_index(flat, scene["front_mask"].shape)
    instance = scene["front_mask"].reshape(-1)[flat]
    return rows, cols, instance, distance


def classify_parts(rows: np.ndarray, cols: np.ndarray, waterer: np.ndarray):
    """Mutually exclusive front-view geometry partition for the visible waterer."""
    labels = np.full(len(rows), -1, dtype=np.int64)
    labels[waterer & (cols < 147)] = 0
    labels[waterer & (cols > 194)] = 1
    center = waterer & (cols >= 147) & (cols <= 194)
    labels[center & (rows < 158)] = 2
    labels[center & (rows >= 158) & (rows < 185)] = 3
    labels[center & (rows >= 185)] = 4
    if np.any(waterer & (labels < 0)):
        raise RuntimeError("Waterer part partition left visible points unassigned")
    return labels


def raw_part_image(scene, visual_handle: int):
    mask = scene["front_mask"] == visual_handle
    rows, cols = np.indices(mask.shape)
    labels = np.full(mask.shape, -1, dtype=np.int64)
    labels[mask & (cols < 147)] = 0
    labels[mask & (cols > 194)] = 1
    center = mask & (cols >= 147) & (cols <= 194)
    labels[center & (rows < 158)] = 2
    labels[center & (rows >= 158) & (rows < 185)] = 3
    labels[center & (rows >= 185)] = 4
    return labels


def stage_assignment(capture, stage: int):
    if stage == 0:
        return np.arange(len(capture["input_coordinates"]), dtype=np.int64)
    return capture[f"input_to_stage{stage}"].astype(np.int64)


def allocate_token_values(values, assignment, part_labels, num_parts=5):
    """Allocate token mass by descendant fractions and compute surface intensity."""
    count = len(values)
    descendants = np.bincount(assignment, minlength=count).astype(np.float64)
    fractions = np.zeros((count, num_parts), dtype=np.float64)
    for part in range(num_parts):
        fractions[:, part] = np.bincount(
            assignment,
            weights=(part_labels == part).astype(np.float64),
            minlength=count,
        ) / np.maximum(descendants, 1.0)
    mass = np.asarray(values, dtype=np.float64) @ fractions
    inherited = np.asarray(values, dtype=np.float64)[assignment]
    mean_intensity = np.asarray([
        inherited[part_labels == part].mean() for part in range(num_parts)
    ])
    return mass, mean_intensity, inherited, fractions


def metric_record(values, assignment, part_labels):
    mass, intensity, inherited, _ = allocate_token_values(
        values, assignment, part_labels
    )
    waterer_mass = float(mass.sum())
    waterer_mean = float(inherited[part_labels >= 0].mean())
    return {
        "waterer_absolute_mass": waterer_mass,
        "parts": {
            name: {
                "visible_input_points": int((part_labels == index).sum()),
                "absolute_mass": float(mass[index]),
                "share_within_waterer": float(mass[index] / max(waterer_mass, 1e-12)),
                "mean_inherited_intensity": float(intensity[index]),
                "intensity_vs_waterer_mean": float(intensity[index] / max(waterer_mean, 1e-12)),
            }
            for index, name in enumerate(PART_NAMES)
        },
    }


def crop_limits(part_image):
    rows, cols = np.where(part_image >= 0)
    return cols.min() - 12, cols.max() + 12, rows.min() - 12, rows.max() + 12


def setup_crop(ax, limits):
    x0, x1, y0, y1 = limits
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.axis("off")


def render_detail(scene, rows, cols, part_labels, action_values, direct_values, part_image):
    limits = crop_limits(part_image)
    selected = part_labels >= 0
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    axes[0, 0].imshow(scene["front_rgb"])
    overlay = np.zeros((*part_image.shape, 4), dtype=np.float32)
    for index, color in enumerate(PART_COLORS):
        from matplotlib.colors import to_rgba
        overlay[part_image == index] = to_rgba(color, alpha=0.58)
    axes[0, 0].imshow(overlay)
    axes[0, 0].legend(
        handles=[Patch(facecolor=c, label=n) for n, c in zip(PART_NAMES, PART_COLORS)],
        fontsize=8, loc="upper right",
    )
    axes[0, 0].set_title("Visible waterer part partition")
    setup_crop(axes[0, 0], limits)

    waterer_action = action_values[selected]
    shared_action_min, shared_action_max = np.percentile(waterer_action, [1, 99])
    scatter = axes[0, 1].scatter(
        cols[selected], rows[selected], c=waterer_action, cmap="magma",
        vmin=shared_action_min, vmax=shared_action_max, s=18, linewidths=0,
    )
    axes[0, 1].imshow(scene["front_rgb"], alpha=0.18)
    axes[0, 1].set_title("Original figure quantity\naction-head position-anchor probability")
    setup_crop(axes[0, 1], limits)
    figure.colorbar(scatter, ax=axes[0, 1], fraction=0.045, label="probability mass")

    unique_action, unique_inverse = np.unique(waterer_action, return_inverse=True)
    # Rank unique anchor values so all descendants of one anchor keep one level.
    level = np.minimum(
        4, (5 * unique_inverse / max(len(unique_action), 1)).astype(int)
    )
    level_cmap = ListedColormap(["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"])
    level_scatter = axes[0, 2].scatter(
        cols[selected], rows[selected], c=level, cmap=level_cmap,
        norm=BoundaryNorm(np.arange(-.5, 5.5), level_cmap.N),
        s=18, linewidths=0,
    )
    axes[0, 2].imshow(scene["front_rgb"], alpha=0.18)
    axes[0, 2].set_title(
        f"Within-waterer rank over {len(unique_action)} unique anchors\n"
        "very low → very high"
    )
    setup_crop(axes[0, 2], limits)
    cb = figure.colorbar(level_scatter, ax=axes[0, 2], fraction=0.045, ticks=range(5))
    cb.ax.set_yticklabels(["very low", "low", "medium", "high", "very high"])

    direct_selected = direct_values[selected]
    direct_min, direct_max = np.percentile(direct_selected, [1, 99])
    direct_scatter = axes[1, 0].scatter(
        cols[selected], rows[selected], c=direct_selected, cmap="turbo",
        vmin=direct_min, vmax=direct_max, s=18, linewidths=0,
    )
    axes[1, 0].imshow(scene["front_rgb"], alpha=0.18)
    axes[1, 0].set_title("True direct action-query attention\nStage 4 Block 2")
    setup_crop(axes[1, 0], limits)
    figure.colorbar(direct_scatter, ax=axes[1, 0], fraction=0.045,
                    label="action→point softmax mass")

    # Two orthographic point plots make overlapping body/handle regions readable.
    capture = np.load(NEW_CAPTURE)
    xyz = capture["input_coordinates"][selected]
    for ax, pair, title in (
        (axes[1, 1], (1, 2), "Waterer side view (y-z)"),
        (axes[1, 2], (0, 2), "Waterer side view (x-z)"),
    ):
        plot = ax.scatter(xyz[:, pair[0]], xyz[:, pair[1]], c=waterer_action,
                          cmap="magma", vmin=shared_action_min,
                          vmax=shared_action_max, s=13, linewidths=0)
        ax.set_xlabel("xyz"[pair[0]])
        ax.set_ylabel("xyz"[pair[1]])
        ax.set_title(title + "\naction-head probability")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
    figure.colorbar(plot, ax=axes[1, 1:], fraction=0.025, label="probability mass")

    figure.suptitle("Detailed waterer-only probability / attention", fontsize=16)
    figure.savefig(RENDERED / "waterer_attention_detail.png", dpi=240)
    plt.close(figure)
    return unique_action.tolist()


def render_part_summary(action_metrics, direct_metrics):
    action_parts = action_metrics["mean"]["parts"]
    names = list(PART_NAMES)
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    ratios = np.asarray([action_parts[name]["intensity_vs_waterer_mean"] for name in names])
    bars = axes[0].bar(x, ratios, color=PART_COLORS)
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0].bar_label(bars, fmt="%.2fx", padding=3)
    axes[0].set_xticks(x, names, rotation=25, ha="right")
    axes[0].set_ylabel("mean intensity / waterer mean")
    axes[0].set_title("Action-head intensity per visible surface point")

    shares = 100 * np.asarray([action_parts[name]["share_within_waterer"] for name in names])
    bars = axes[1].bar(x, shares, color=PART_COLORS)
    axes[1].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[1].set_xticks(x, names, rotation=25, ha="right")
    axes[1].set_ylabel("% of waterer probability mass")
    axes[1].set_title("Action-head mass allocation (part size matters)")

    matrix = np.asarray([
        [direct_metrics[str(stage)]["parts"][name]["intensity_vs_waterer_mean"]
         for name in names]
        for stage in range(5)
    ])
    image = axes[2].imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=.5, vmax=1.5)
    axes[2].set_xticks(x, names, rotation=25, ha="right")
    axes[2].set_yticks(np.arange(5), [f"S{s} B{LAST_BLOCK[s]}" for s in range(5)])
    axes[2].set_title("True direct-attention intensity\n(relative to waterer mean per stage)")
    for row in range(5):
        for column in range(len(names)):
            axes[2].text(column, row, f"{matrix[row,column]:.2f}x",
                         ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axes[2], fraction=.045, label="relative intensity")
    figure.suptitle("Which part of the waterer is high or low?", fontsize=16)
    figure.savefig(RENDERED / "waterer_part_quantitative.png", dpi=240)
    plt.close(figure)


def main() -> None:
    RENDERED.mkdir(parents=True, exist_ok=True)
    capture = np.load(NEW_CAPTURE)
    old = np.load(OLD_CAPTURE)
    scene = np.load(SCENE)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    visual_handle = next(
        item["handle"] for item in metadata["waterer"]["objects"]
        if item["name"] == "waterer_visual"
    )
    rows, cols, instance, distance = match_input_to_front(capture, scene)
    waterer = instance == visual_handle
    part_labels = classify_parts(rows, cols, waterer)
    part_image = raw_part_image(scene, visual_handle)

    # Use the exact action-head values from the user-referenced old capture.
    final_assignment = capture["input_to_final"].astype(np.int64)
    action_metrics = {}
    for axis, label in enumerate(("x", "y", "z")):
        action_metrics[label] = metric_record(
            old["action_point_probability_xyz"][axis], final_assignment, part_labels
        )
    action_metrics["mean"] = metric_record(
        old["action_point_probability"], final_assignment, part_labels
    )

    direct_metrics = {}
    direct_inherited = None
    for stage, block in LAST_BLOCK.items():
        assignment = stage_assignment(capture, stage)
        values = capture[f"action_attention_stage{stage}_block{block}_point_weights"]
        direct_metrics[str(stage)] = metric_record(values, assignment, part_labels)
        if stage == 4:
            direct_inherited = values[assignment]

    action_inherited = old["action_point_probability"][final_assignment]
    unique_action_values = render_detail(
        scene, rows, cols, part_labels, action_inherited, direct_inherited, part_image
    )
    render_part_summary(action_metrics, direct_metrics)

    validation = [
        float(capture[key]) for key in capture.files
        if key.endswith("_reconstruction_cosine")
    ]
    report = {
        "task": metadata["task"],
        "instruction": metadata["instruction"],
        "referenced_old_capture": str(OLD_CAPTURE),
        "exact_scene_reproduction": {
            "input_point_count_old_new": [
                int(len(old["input_coordinates"])), int(len(capture["input_coordinates"]))
            ],
            "final_token_count_old_new": [
                int(len(old["coordinates"])), int(len(capture["coordinates"]))
            ],
            "maximum_input_coordinate_difference": float(
                np.max(np.abs(old["input_coordinates"] - capture["input_coordinates"]))
            ),
            "maximum_action_probability_difference_on_rerun": float(
                np.max(np.abs(old["action_point_probability"] - capture["action_point_probability"]))
            ),
        },
        "segmentation": {
            "method": "Mutually exclusive front-view geometric partition inside the exact RLBench waterer_visual instance mask.",
            "boundaries_pixels": {
                "spout": "column < 147",
                "handle": "column > 194",
                "top opening/rim": "147 <= column <= 194 and row < 158",
                "upper body": "147 <= column <= 194 and 158 <= row < 185",
                "lower body": "147 <= column <= 194 and row >= 185",
            },
            "model_input_waterer_points": int(waterer.sum()),
            "nearest_raw_point_distance_m_max": float(distance.max()),
        },
        "original_figure_quantity": {
            "definition": "Mean over x/y/z of the action head's position-anchor probability; this is not transformer attention.",
            "within_waterer_unique_anchor_values": unique_action_values,
            "by_axis_and_mean": action_metrics,
        },
        "true_direct_action_attention": {
            "definition": "Action-query to point-key softmax attention, averaged over heads; robot-state query excluded.",
            "minimum_reconstruction_cosine_all_24_blocks": min(validation),
            "last_block_per_stage": direct_metrics,
        },
        "interpretation_notes": [
            "Mean inherited intensity compares local surface strength independent of part area.",
            "Mass share measures how much of the waterer's total probability is allocated to a part and therefore depends on visible area.",
            "Pooled-token mass is apportioned using exact input-point descendants.",
            "The simulator exposes the visible waterer as one instance, so named parts use explicit front-view geometry boundaries shown in the plot.",
            "Direct attention is not attention rollout or causal importance.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(action_metrics["mean"], indent=2))
    print(json.dumps(direct_metrics["4"], indent=2))


if __name__ == "__main__":
    main()
