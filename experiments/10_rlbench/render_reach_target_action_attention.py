"""Render PointACT action-query attention on the three reach_target balls.

This consumes captures made by run_ptv3_action_attention_server.py.  It never
imports or modifies the model: all plots are offline diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_reach_target_action_attention_20260916"
CAPTURE = OUTPUT / "captures" / "capture_000000.npz"
SCENE = OUTPUT / "scene" / "observation.npz"
METADATA = OUTPUT / "scene" / "metadata.json"
RENDERED = OUTPUT / "rendered"

OBJECT_NAMES = ("target", "distractor0", "distractor1")
OBJECT_LABELS = ("red target", "green distractor 0", "green distractor 1")
OBJECT_COLORS = ("#e41a1c", "#4daf4a", "#1b7837")
LAST_BLOCK = {0: 2, 1: 2, 2: 2, 3: 11, 4: 2}


def point_to_scene(capture: np.lib.npyio.NpzFile, scene: np.lib.npyio.NpzFile):
    """Match each model input point to the nearest camera pixel and mask id."""
    points_world = capture["input_coordinates"] + capture["scene_center"]
    raw = scene["world_points"].reshape(-1, 3)
    valid = np.isfinite(raw).all(axis=1)
    distance, compact_index = cKDTree(raw[valid]).query(points_world)
    flat_index = np.flatnonzero(valid)[compact_index]
    height, width = scene["instance_mask"].shape
    rows, cols = np.unravel_index(flat_index, (height, width))
    instance_ids = scene["instance_mask"].reshape(-1)[flat_index]
    return points_world, rows, cols, instance_ids, distance


def object_membership(instance_ids: np.ndarray, metadata: dict) -> np.ndarray:
    membership = np.zeros((len(instance_ids), len(OBJECT_NAMES)), dtype=np.float64)
    by_name = {item["name"]: item for item in metadata["objects"]}
    for column, name in enumerate(OBJECT_NAMES):
        membership[:, column] = np.isin(
            instance_ids, np.asarray(by_name[name]["handles"])
        )
    return membership


def assignment_for_stage(capture: np.lib.npyio.NpzFile, stage: int) -> np.ndarray:
    if stage == 0:
        return np.arange(len(capture["input_coordinates"]), dtype=np.int64)
    return capture[f"input_to_stage{stage}"].astype(np.int64)


def fractions_per_token(membership: np.ndarray, assignment: np.ndarray, count: int):
    sizes = np.bincount(assignment, minlength=count).astype(np.float64)
    fractions = np.zeros((count, membership.shape[1]), dtype=np.float64)
    for column in range(membership.shape[1]):
        fractions[:, column] = np.bincount(
            assignment, weights=membership[:, column], minlength=count
        )
    fractions /= np.maximum(sizes[:, None], 1.0)
    return fractions, sizes


def summarize_block(
    capture: np.lib.npyio.NpzFile,
    prefix: str,
    stage: int,
    membership: np.ndarray,
) -> dict:
    weight = capture[f"{prefix}_point_weights"].astype(np.float64)
    assignment = assignment_for_stage(capture, stage)
    fractions, sizes = fractions_per_token(membership, assignment, len(weight))
    object_mass = weight @ fractions
    point_mass = float(weight.sum())
    background_mass = max(0.0, point_mass - float(object_mass.sum()))
    internal_token_mass = max(0.0, 1.0 - point_mass)
    ball_total = float(object_mass.sum())
    ball_share = object_mass / ball_total if ball_total > 0 else np.zeros(3)
    return {
        "stage": stage,
        "block": int(prefix.rsplit("block", 1)[1]),
        "num_stage_points": int(len(weight)),
        "num_action_queries": int(capture[f"{prefix}_num_action_queries"]),
        "num_heads": int(capture[f"{prefix}_num_heads"]),
        "point_attention_mass": point_mass,
        "internal_action_state_token_mass": internal_token_mass,
        "object_attention_mass": {
            name: float(value) for name, value in zip(OBJECT_NAMES, object_mass)
        },
        "background_attention_mass": background_mass,
        "share_among_three_balls": {
            name: float(value) for name, value in zip(OBJECT_NAMES, ball_share)
        },
        "reconstruction_max_abs_error": float(
            capture[f"{prefix}_reconstruction_max_abs_error"]
        ),
        "reconstruction_mean_abs_error": float(
            capture[f"{prefix}_reconstruction_mean_abs_error"]
        ),
        "reconstruction_cosine": float(
            capture[f"{prefix}_reconstruction_cosine"]
        ),
        "descendants_per_token_min_median_max": [
            float(sizes.min()), float(np.median(sizes)), float(sizes.max())
        ],
    }


def add_ball_boxes(ax, rows: np.ndarray, cols: np.ndarray, membership: np.ndarray):
    for index, (label, color) in enumerate(zip(OBJECT_LABELS, OBJECT_COLORS)):
        selected = membership[:, index] > 0
        if not selected.any():
            continue
        x0, x1 = cols[selected].min(), cols[selected].max()
        y0, y1 = rows[selected].min(), rows[selected].max()
        pad = 4
        ax.add_patch(
            Rectangle(
                (x0 - pad, y0 - pad), x1 - x0 + 2 * pad, y1 - y0 + 2 * pad,
                fill=False, edgecolor=color, linewidth=2.0,
            )
        )
        ax.text(
            x0 - pad, y0 - pad - 3, label, color="white", fontsize=7,
            bbox={"facecolor": color, "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
        )


def render_multiscale(
    capture, scene, rows, cols, membership, summaries: dict[str, dict]
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(scene["rgb"])
    add_ball_boxes(axes[0], rows, cols, membership)
    axes[0].set_title("RLBench input: reach the red target")
    axes[0].axis("off")

    for panel, stage in enumerate(range(5), start=1):
        block = LAST_BLOCK[stage]
        prefix = f"action_attention_stage{stage}_block{block}"
        weights = capture[f"{prefix}_point_weights"].astype(np.float64)
        assignment = assignment_for_stage(capture, stage)
        per_input = weights[assignment]
        upper = float(np.percentile(per_input, 99))
        if upper <= 0:
            upper = float(per_input.max()) or 1.0
        ax = axes[panel]
        ax.imshow(scene["rgb"], alpha=0.22)
        scatter = ax.scatter(
            cols, rows, c=np.clip(per_input / upper, 0, 1), cmap="turbo",
            vmin=0, vmax=1, s=7, linewidths=0, alpha=0.92,
        )
        add_ball_boxes(ax, rows, cols, membership)
        info = summaries[prefix]
        share = info["share_among_three_balls"]
        ax.set_title(
            f"Stage {stage}, last block {block} ({len(weights)} tokens)\n"
            f"within-ball share T/D0/D1: "
            f"{100*share['target']:.1f}/{100*share['distractor0']:.1f}/"
            f"{100*share['distractor1']:.1f}%"
        )
        ax.axis("off")
        cb = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.01)
        cb.set_label("relative direct attention (clipped at input-point p99)", fontsize=7)

    fig.suptitle(
        "Action-query → point-key attention (mean over heads; one action query token)",
        fontsize=15,
    )
    fig.savefig(RENDERED / "action_attention_multiscale.png", dpi=220)
    plt.close(fig)


def render_summary(last_summaries: list[dict]):
    stage_labels = [f"S{x['stage']} B{x['block']}" for x in last_summaries]
    object_values = np.asarray(
        [[s["object_attention_mass"][name] for name in OBJECT_NAMES] for s in last_summaries]
    )
    background = np.asarray([s["background_attention_mass"] for s in last_summaries])
    internal = np.asarray(
        [s["internal_action_state_token_mass"] for s in last_summaries]
    )
    shares = np.asarray(
        [[s["share_among_three_balls"][name] for name in OBJECT_NAMES] for s in last_summaries]
    )
    x = np.arange(len(stage_labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    bottom = np.zeros(len(x))
    for column, (label, color) in enumerate(zip(OBJECT_LABELS, OBJECT_COLORS)):
        axes[0].bar(x, 100 * object_values[:, column], bottom=100 * bottom,
                    label=label, color=color)
        bottom += object_values[:, column]
    axes[0].bar(x, 100 * background, bottom=100 * bottom,
                label="other scene points", color="#9e9e9e")
    bottom += background
    axes[0].bar(x, 100 * internal, bottom=100 * bottom,
                label="action/state token keys", color="#377eb8")
    axes[0].set_xticks(x, stage_labels)
    axes[0].set_ylabel("% of all direct attention")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Where the action query attends")
    axes[0].legend(fontsize=8, loc="upper left")

    width = 0.24
    for column, (label, color) in enumerate(zip(OBJECT_LABELS, OBJECT_COLORS)):
        bars = axes[1].bar(x + (column - 1) * width, 100 * shares[:, column],
                           width=width, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    axes[1].set_xticks(x, stage_labels)
    axes[1].set_ylabel("% among the three balls only")
    axes[1].set_ylim(0, max(100, 100 * shares.max() + 10))
    axes[1].set_title("Three-ball comparison (renormalized; not total attention)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Direct action attention to target and distractor balls")
    fig.savefig(RENDERED / "action_attention_ball_summary.png", dpi=220)
    plt.close(fig)


def render_all_blocks(all_summaries: list[dict]):
    labels = [f"S{s['stage']}B{s['block']}" for s in all_summaries]
    shares = np.asarray(
        [[s["share_among_three_balls"][name] for name in OBJECT_NAMES] for s in all_summaries]
    )
    masses = np.asarray(
        [[s["object_attention_mass"][name] for name in OBJECT_NAMES] for s in all_summaries]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), constrained_layout=True)
    im0 = axes[0].imshow(100 * shares, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    axes[0].set_title("Share among balls (%)")
    im1 = axes[1].imshow(100 * masses, aspect="auto", cmap="magma")
    axes[1].set_title("Absolute attention mass (% of all keys)")
    for ax in axes:
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.set_xticks(np.arange(3), OBJECT_LABELS, rotation=20, ha="right")
    for row in range(len(labels)):
        for col in range(3):
            axes[0].text(col, row, f"{100*shares[row,col]:.1f}", ha="center", va="center",
                         color="white" if shares[row,col] < 0.75 else "black", fontsize=7)
            axes[1].text(col, row, f"{100*masses[row,col]:.3f}", ha="center", va="center",
                         color="white", fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.04, label="%")
    fig.colorbar(im1, ax=axes[1], fraction=0.04, label="% of total direct attention")
    fig.suptitle("All 24 encoder action-attention blocks")
    fig.savefig(RENDERED / "action_attention_all_blocks.png", dpi=220)
    plt.close(fig)


def main() -> None:
    RENDERED.mkdir(parents=True, exist_ok=True)
    capture = np.load(CAPTURE)
    scene = np.load(SCENE)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    _, rows, cols, instance_ids, distance = point_to_scene(capture, scene)
    membership = object_membership(instance_ids, metadata)

    prefixes = []
    pattern = re.compile(r"(action_attention_stage(\d+)_block(\d+))_point_weights$")
    for key in capture.files:
        match = pattern.fullmatch(key)
        if match:
            prefixes.append((int(match.group(2)), int(match.group(3)), match.group(1)))
    prefixes.sort()
    summaries = {
        prefix: summarize_block(capture, prefix, stage, membership)
        for stage, _block, prefix in prefixes
    }
    last_summaries = [
        summaries[f"action_attention_stage{stage}_block{block}"]
        for stage, block in LAST_BLOCK.items()
    ]

    render_multiscale(capture, scene, rows, cols, membership, summaries)
    render_summary(last_summaries)
    render_all_blocks(list(summaries.values()))

    reconstruction = list(summaries.values())
    report = {
        "task": metadata["task"],
        "instruction": metadata["instruction"],
        "checkpoint": "checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000",
        "actions_executed": metadata["actions_executed"],
        "input_point_count": int(len(instance_ids)),
        "input_points_per_object": {
            name: int(membership[:, index].sum())
            for index, name in enumerate(OBJECT_NAMES)
        },
        "nearest_scene_pixel_distance_m": {
            "max": float(distance.max()),
            "mean": float(distance.mean()),
            "p99": float(np.percentile(distance, 99)),
        },
        "attention_definition": (
            "Direct softmax weight from the learned action query to PTV3 point keys, "
            "averaged over attention heads. The leading robot-state query is excluded."
        ),
        "num_action_query_tokens": int(reconstruction[0]["num_action_queries"]),
        "validation": {
            "num_attention_blocks": len(reconstruction),
            "minimum_reconstruction_cosine": min(
                x["reconstruction_cosine"] for x in reconstruction
            ),
            "maximum_mean_abs_error": max(
                x["reconstruction_mean_abs_error"] for x in reconstruction
            ),
            "maximum_abs_error": max(
                x["reconstruction_max_abs_error"] for x in reconstruction
            ),
        },
        "last_block_per_stage": last_summaries,
        "all_blocks": list(summaries.values()),
        "caveats": [
            "This is direct attention, not attention rollout and not a causal importance score.",
            "At pooled stages, a token's mass is apportioned by the fractions of its input-point descendants belonging to each object.",
            "The heatmaps repeat a pooled token's weight on its descendants for visualization; color is relative within each panel.",
            "Point attention plus action/state-token-key attention sums to approximately one.",
            "reach_target is not one of the checkpoint's hybridvla_10tasks training tasks, so this is an out-of-distribution diagnostic.",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["validation"], indent=2))
    for item in last_summaries:
        print(
            f"stage {item['stage']} block {item['block']}: ",
            item["share_among_three_balls"],
            "absolute=", item["object_attention_mass"],
        )


if __name__ == "__main__":
    main()
