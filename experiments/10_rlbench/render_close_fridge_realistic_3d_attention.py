"""Render intuitive full 3-D point clouds and true action-attention heatmaps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from ptv3_feature_viz import write_ply

CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(CJK_FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=CJK_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_close_fridge_realistic_missing_20260917"
INPUTS = OUTPUT / "attention_inputs"
CAPTURES = OUTPUT / "attention_captures"
RENDERED = OUTPUT / "rendered"
PLY_DIR = OUTPUT / "pointclouds_3d"
SEVERITIES = (0.0, 0.25, 0.50, 0.75)
X_LIM = (-0.274, 0.774)
Y_LIM = (-0.655, 0.655)
Z_LIM = (0.752, 1.751)
ATTENTION_KEY = "action_attention_stage4_block2_point_weights"
ATTENTION_COORD_KEY = "action_attention_stage4_block2_point_coordinates"


def as_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    if rgb.size and rgb.max() > 1.5:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def in_workspace(xyz: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(xyz).all(axis=-1)
        & (xyz[..., 0] >= X_LIM[0]) & (xyz[..., 0] <= X_LIM[1])
        & (xyz[..., 1] >= Y_LIM[0]) & (xyz[..., 1] <= Y_LIM[1])
        & (xyz[..., 2] >= Z_LIM[0]) & (xyz[..., 2] <= Z_LIM[1])
    )


def configure_3d(ax) -> None:
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_zlim(*Z_LIM)
    ax.set_box_aspect((X_LIM[1]-X_LIM[0], Y_LIM[1]-Y_LIM[0], Z_LIM[1]-Z_LIM[0]))
    ax.view_init(elev=27, azim=-58)
    ax.set_xlabel("x", labelpad=-7)
    ax.set_ylabel("y", labelpad=-7)
    ax.set_zlabel("z", labelpad=-7)
    ax.tick_params(labelsize=6, pad=-2)
    ax.grid(alpha=0.16)


def full_scene_bounds(raw) -> tuple[np.ndarray, np.ndarray]:
    all_points = []
    for sample in raw:
        points = sample["corrupted_points"]
        valid = np.isfinite(points).all(axis=-1)
        all_points.append(points[valid])
    points = np.concatenate(all_points)
    low, high = np.percentile(points, [0.2, 99.8], axis=0)
    padding = np.maximum(0.035 * (high-low), 0.015)
    return low-padding, high+padding


def configure_full_scene_3d(ax, bounds) -> None:
    low, high = bounds
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_box_aspect(np.maximum(high-low, 1e-3))
    ax.view_init(elev=25, azim=-58)
    ax.set_xlabel("x", labelpad=-7)
    ax.set_ylabel("y", labelpad=-7)
    ax.set_zlabel("z", labelpad=-7)
    ax.tick_params(labelsize=6, pad=-2)
    ax.grid(alpha=0.16)


def world_input(capture) -> np.ndarray:
    center = np.asarray(capture["scene_center"], dtype=np.float32).reshape(-1, 3)[0]
    return capture["input_coordinates"].astype(np.float32) + center


def world_stage4(capture) -> np.ndarray:
    center = np.asarray(capture["scene_center"], dtype=np.float32).reshape(-1, 3)[0]
    return capture[ATTENTION_COORD_KEY].astype(np.float32) + center


def input_attention(capture) -> np.ndarray:
    weights = capture[ATTENTION_KEY].astype(np.float32)
    return weights[capture["input_to_stage4"].astype(np.int64)]


def predicted_world(capture) -> np.ndarray:
    center = np.asarray(capture["scene_center"], dtype=np.float32).reshape(-1, 3)[0]
    return capture["predicted_position"].astype(np.float32) + center


def load_all():
    trials = json.loads((OUTPUT / "attention_trials.json").read_text(encoding="utf-8"))
    raw = [np.load(INPUTS / f"severity_{s:.2f}.npz") for s in SEVERITIES]
    captures = [np.load(CAPTURES / f"capture_{i:06d}.npz") for i in range(4)]
    return trials, raw, captures


def render_main(raw, captures, norm, cmap) -> None:
    clean_pred = predicted_world(captures[0])
    fig = plt.figure(figsize=(20, 10.7), constrained_layout=True)
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for col in range(4):
            axes[row, col] = fig.add_subplot(2, 4, row*4+col+1, projection="3d")

    for col, (severity, sample, capture) in enumerate(zip(SEVERITIES, raw, captures)):
        corrupted = sample["corrupted_points"]
        rgb = as_rgb(sample["rgb"])
        valid = in_workspace(corrupted)
        xyz = corrupted[valid]
        colors = rgb[valid]
        axes[0, col].scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=0.75,
            linewidths=0, depthshade=False, rasterized=True,
        )
        axes[0, col].set_title(
            f"{severity:.0%} failure\nactual valid workspace points: {len(xyz):,}",
            fontsize=11,
        )

        model_xyz = world_input(capture)
        attention = input_attention(capture)
        # A faint raw cloud gives geometric context; colored points are the
        # exact voxelized points received by PTV3.
        context = xyz[::4]
        axes[1, col].scatter(
            context[:, 0], context[:, 1], context[:, 2], c="#b9b9b9",
            s=0.35, alpha=0.10, linewidths=0, depthshade=False, rasterized=True,
        )
        axes[1, col].scatter(
            model_xyz[:, 0], model_xyz[:, 1], model_xyz[:, 2],
            c=attention, cmap=cmap, norm=norm, s=4.0,
            linewidths=0, depthshade=False, rasterized=True,
        )
        pred = predicted_world(capture)
        axes[1, col].scatter(
            pred[0], pred[1], pred[2], marker="*", s=180,
            c="white", edgecolors="black", linewidths=1.0,
        )
        drift = 100.0 * float(np.linalg.norm(pred-clean_pred))
        axes[1, col].set_title(
            f"true direct attention on {len(model_xyz):,} PTV3 input points\n"
            f"first action drift: {drift:.1f} cm",
            fontsize=10,
        )
        configure_3d(axes[0, col])
        configure_3d(axes[1, col])

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[1]),
        location="bottom", shrink=0.72, pad=0.025, aspect=45,
    )
    colorbar.set_label(
        "Stage-4 Block-2 direct action-query → point-key attention "
        "(shared absolute scale; p99.5 clipped)"
    )
    fig.text(0.008, 0.73, "Actual corrupted RGB point cloud", rotation=90,
             va="center", fontsize=13, weight="bold")
    fig.text(0.008, 0.27, "Action attention on the same 3-D scene", rotation=90,
             va="center", fontsize=13, weight="bold")
    fig.suptitle(
        "close_fridge — full 3-D point cloud and PointACT action attention\n"
        "Same scene/view/bounds/color scale; white star = first predicted action position",
        fontsize=16,
    )
    fig.savefig(RENDERED / "workspace_3d_pointcloud_and_attention.png", dpi=220)
    plt.close(fig)


def render_full_scene(raw, captures, norm, cmap) -> None:
    bounds = full_scene_bounds(raw)
    clean_pred = predicted_world(captures[0])
    fig = plt.figure(figsize=(20, 10.7), constrained_layout=True)
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for col in range(4):
            axes[row, col] = fig.add_subplot(2, 4, row*4+col+1, projection="3d")

    for col, (severity, sample, capture) in enumerate(zip(SEVERITIES, raw, captures)):
        corrupted = sample["corrupted_points"]
        rgb = as_rgb(sample["rgb"])
        valid_all = np.isfinite(corrupted).all(axis=-1)
        valid_workspace = in_workspace(corrupted)
        xyz = corrupted[valid_all]
        colors = rgb[valid_all]
        axes[0, col].scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=0.40,
            linewidths=0, depthshade=False, rasterized=True,
        )
        axes[0, col].set_title(
            f"{severity:.0%} 缺失强度\n全部有效点 {len(xyz):,}；工作空间内 {valid_workspace.sum():,}",
            fontsize=11,
        )

        # Keep the complete scene as faint context and overlay the exact PTV3
        # input points with true direct attention.
        context = xyz[::5]
        axes[1, col].scatter(
            context[:, 0], context[:, 1], context[:, 2], c="#a8a8a8",
            s=0.28, alpha=0.12, linewidths=0, depthshade=False, rasterized=True,
        )
        model_xyz = world_input(capture)
        attention = input_attention(capture)
        axes[1, col].scatter(
            model_xyz[:, 0], model_xyz[:, 1], model_xyz[:, 2],
            c=attention, cmap=cmap, norm=norm, s=7.0,
            linewidths=0, depthshade=False, rasterized=True,
        )
        pred = predicted_world(capture)
        axes[1, col].scatter(
            pred[0], pred[1], pred[2], marker="*", s=190,
            c="white", edgecolors="black", linewidths=1.0,
        )
        drift = 100.0 * float(np.linalg.norm(pred-clean_pred))
        axes[1, col].set_title(
            f"真实 Stage-4/Block-2 action attention\n"
            f"首个动作位置漂移 {drift:.1f} cm",
            fontsize=10,
        )
        configure_full_scene_3d(axes[0, col], bounds)
        configure_full_scene_3d(axes[1, col], bounds)

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[1]),
        location="bottom", shrink=0.72, pad=0.025, aspect=45,
    )
    colorbar.set_label(
        "Stage-4 Block-2 直接 action-query → point-key attention "
        "（四列共用绝对色标，截断于 99.5 百分位）"
    )
    fig.text(0.008, 0.73, "前相机完整 3D RGB 点云", rotation=90,
             va="center", fontsize=13, weight="bold")
    fig.text(0.008, 0.27, "完整场景背景上的 action attention", rotation=90,
             va="center", fontsize=13, weight="bold")
    fig.suptitle(
        "close_fridge：完整残缺 3D 点云及 PointACT action attention\n"
        "四列使用同一场景、视角、坐标范围和 attention 色标；白色星号为首个预测动作位置",
        fontsize=16,
    )
    fig.savefig(RENDERED / "full_3d_pointcloud_and_attention.png", dpi=220)
    plt.close(fig)


def render_corruption_and_change(raw, captures) -> None:
    clean_xyz = world_input(captures[0])
    clean_attention = input_attention(captures[0])
    clean_tree = cKDTree(clean_xyz)
    positive = clean_attention[clean_attention > 0]
    epsilon = max(float(np.percentile(positive, 5)) * 0.1, 1e-12)

    fig = plt.figure(figsize=(20, 10.7), constrained_layout=True)
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for col in range(4):
            axes[row, col] = fig.add_subplot(2, 4, row*4+col+1, projection="3d")

    type_colors = {0: "#bcbcbc", 2: "#377eb8", 3: "#ffad4d"}
    for col, (severity, sample, capture) in enumerate(zip(SEVERITIES, raw, captures)):
        clean = sample["clean_points"]
        corrupted = sample["corrupted_points"]
        labels = sample["corruption_labels"]
        for label, color in type_colors.items():
            select = (labels == label) & in_workspace(corrupted)
            points = corrupted[select]
            axes[0, col].scatter(
                points[:, 0], points[:, 1], points[:, 2], c=color,
                s=0.65 if label == 0 else 2.2, alpha=0.34 if label == 0 else 0.90,
                linewidths=0, depthshade=False, rasterized=True,
            )
        # Missing samples have no corrupted XYZ. Plot their original locations
        # as crosses so the 3-D holes remain visible and interpretable.
        for label, color in ((1, "#e41a1c"), (4, "#984ea3")):
            select = (labels == label) & in_workspace(clean)
            points = clean[select]
            axes[0, col].scatter(
                points[:, 0], points[:, 1], points[:, 2], c=color,
                s=3.0, marker="x", alpha=0.85, linewidths=0.35,
                depthshade=False, rasterized=True,
            )
        axes[0, col].set_title(f"{severity:.0%} failure: where geometry was damaged")

        current_xyz = world_input(capture)
        current_attention = input_attention(capture)
        distance, nearest = clean_tree.query(current_xyz)
        matched = distance <= 0.015
        log2_ratio = np.zeros(len(current_xyz), dtype=np.float32)
        log2_ratio[matched] = np.log2(
            (current_attention[matched] + epsilon)
            / (clean_attention[nearest[matched]] + epsilon)
        )
        unmatched = ~matched
        axes[1, col].scatter(
            current_xyz[unmatched, 0], current_xyz[unmatched, 1], current_xyz[unmatched, 2],
            c="#a8a8a8", s=2.0, alpha=0.25, linewidths=0,
            depthshade=False, rasterized=True,
        )
        axes[1, col].scatter(
            current_xyz[matched, 0], current_xyz[matched, 1], current_xyz[matched, 2],
            c=np.clip(log2_ratio[matched], -2, 2), cmap="coolwarm",
            norm=Normalize(-2, 2), s=4.0, linewidths=0,
            depthshade=False, rasterized=True,
        )
        axes[1, col].set_title(
            f"attention change vs clean at matched locations\n"
            f"spatially matched: {100*matched.mean():.1f}%"
        )
        configure_3d(axes[0, col])
        configure_3d(axes[1, col])

    legend = [
        Line2D([0], [0], marker="o", linestyle="", color="#bcbcbc", label="unchanged valid"),
        Line2D([0], [0], marker="x", linestyle="", color="#e41a1c", label="target missing (ghost location)"),
        Line2D([0], [0], marker="o", linestyle="", color="#377eb8", label="background-depth leakage"),
        Line2D([0], [0], marker="o", linestyle="", color="#ffad4d", label="range distortion"),
        Line2D([0], [0], marker="x", linestyle="", color="#984ea3", label="other scene hole (ghost location)"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.945))
    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=Normalize(-2, 2), cmap="coolwarm"),
        ax=list(axes[1]), location="bottom", shrink=0.72, pad=0.025, aspect=45,
    )
    cb.set_label("log2 attention ratio vs clean: blue = lower, red = higher; grey = no ≤1.5 cm match")
    fig.suptitle(
        "close_fridge — 3-D corruption anatomy and attention redistribution",
        fontsize=16,
    )
    fig.savefig(RENDERED / "full_3d_corruption_and_attention_change.png", dpi=220)
    plt.close(fig)


def configure_target_3d(ax, xyz: np.ndarray, azim: float) -> None:
    low, high = np.percentile(xyz, [0.5, 99.5], axis=0)
    padding = np.maximum(0.04 * (high-low), 0.008)
    low -= padding
    high += padding
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_box_aspect(np.maximum(high-low, 1e-3))
    ax.view_init(elev=12, azim=azim)
    ax.set_xlabel("x", labelpad=-6)
    ax.set_ylabel("y", labelpad=-6)
    ax.set_zlabel("z", labelpad=-6)
    ax.tick_params(labelsize=6, pad=-2)
    ax.grid(alpha=0.18)


def render_target_mask_explanation(sample) -> None:
    PLY_DIR.mkdir(parents=True, exist_ok=True)
    rgb = as_rgb(sample["rgb"])
    mask = sample["target_mask"].astype(bool)
    points = sample["clean_points"]
    target_valid = mask & np.isfinite(points).all(axis=-1)
    target_workspace = target_valid & in_workspace(points)
    target_xyz = points[target_valid]
    target_rgb = rgb[target_valid]
    workspace_xyz = points[target_workspace]
    workspace_rgb = rgb[target_workspace]
    boundary = mask & ~binary_erosion(mask, iterations=1, border_value=0)

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    ax_rgb = fig.add_subplot(2, 2, 1)
    ax_mask = fig.add_subplot(2, 2, 2)
    ax_all = fig.add_subplot(2, 2, 3, projection="3d")
    ax_workspace = fig.add_subplot(2, 2, 4, projection="3d")

    ax_rgb.imshow(rgb)
    ax_rgb.set_title("原始 front RGB：冰箱位于画面右侧并占据较大区域")
    ax_rgb.axis("off")

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = (0.0, 0.9, 0.95, 0.35)
    overlay[boundary] = (1.0, 0.1, 0.05, 0.95)
    ax_mask.imshow(rgb)
    ax_mask.imshow(overlay)
    ax_mask.set_title(
        f"target mask：青色内部/红色边界\n"
        f"fridge_root 及子物体的可见像素：{mask.sum():,}/{mask.size:,} ({100*mask.mean():.1f}%)"
    )
    ax_mask.axis("off")

    ax_all.scatter(
        target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2],
        c=target_rgb, s=0.8, linewidths=0, depthshade=False, rasterized=True,
    )
    ax_all.set_title(
        f"target mask 对应的全部可见 3D 表面：{len(target_xyz):,} 点\n"
        "正对主要门板方向观察；仍只是单相机可见表面"
    )
    configure_target_3d(ax_all, target_xyz, azim=0)

    ax_workspace.scatter(
        workspace_xyz[:, 0], workspace_xyz[:, 1], workspace_xyz[:, 2],
        c=workspace_rgb, s=1.3, linewidths=0, depthshade=False, rasterized=True,
    )
    ax_workspace.set_title(
        f"PointACT 工作空间裁剪后：{len(workspace_xyz):,} 点\n"
        "这才是干净场景中模型实际能够看到的冰箱几何"
    )
    configure_target_3d(ax_workspace, target_xyz, azim=-52)

    fig.suptitle(
        "close_fridge：target mask 到底是什么，以及为什么 3D 中像一块薄板\n"
        "它是 RLBench instance handle 生成的实验损坏区域，不是模型预测、特征或 attention",
        fontsize=16,
    )
    fig.savefig(RENDERED / "target_mask_and_fridge_shape_explained.png", dpi=220)
    plt.close(fig)

    write_ply(PLY_DIR / "target_mask_all_visible_fridge.ply", target_xyz, target_rgb)
    write_ply(
        PLY_DIR / "target_mask_workspace_visible_fridge.ply",
        workspace_xyz,
        workspace_rgb,
    )


def metrics_and_ply(trials, raw, captures, norm, cmap) -> dict:
    PLY_DIR.mkdir(parents=True, exist_ok=True)
    clean_pred = predicted_world(captures[0])
    clean_input = world_input(captures[0])
    clean_input_attention = input_attention(captures[0])
    clean_tree = cKDTree(clean_input)
    rows = []

    for severity, sample, capture in zip(SEVERITIES, raw, captures):
        corrupted = sample["corrupted_points"]
        rgb = as_rgb(sample["rgb"])
        visible_all = np.isfinite(corrupted).all(axis=-1)
        visible_workspace = in_workspace(corrupted)
        raw_xyz = corrupted[visible_all]
        raw_rgb = rgb[visible_all]

        model_xyz = world_input(capture)
        attention = input_attention(capture)
        attention_rgb = cmap(norm(attention))[:, :3]
        write_ply(PLY_DIR / f"severity_{severity:.2f}_full_rgb.ply", raw_xyz, raw_rgb)
        write_ply(
            PLY_DIR / f"severity_{severity:.2f}_attention_stage4_block2.ply",
            model_xyz,
            attention_rgb,
        )

        stage_xyz = world_stage4(capture)
        stage_weight = capture[ATTENTION_KEY].astype(np.float64)
        point_mass = float(stage_weight.sum())
        probability = stage_weight / max(point_mass, 1e-15)
        centroid = np.sum(stage_xyz * probability[:, None], axis=0)
        entropy = float(
            -np.sum(probability * np.log(np.maximum(probability, 1e-15)))
            / max(np.log(len(probability)), 1e-15)
        )
        distance, nearest = clean_tree.query(model_xyz)
        matched = distance <= 0.015
        if matched.any():
            a = clean_input_attention[nearest[matched]].astype(np.float64)
            b = attention[matched].astype(np.float64)
            cosine = float(np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-15))
        else:
            cosine = float("nan")
        pred = predicted_world(capture)
        rows.append(
            {
                "severity": severity,
                "raw_valid_all_camera_points": int(len(raw_xyz)),
                "raw_valid_workspace_points": int(visible_workspace.sum()),
                "ptv3_input_points": int(len(model_xyz)),
                "stage4_tokens": int(len(stage_weight)),
                "point_key_attention_mass": point_mass,
                "attention_entropy_among_point_keys": entropy,
                "attention_centroid_world": centroid.astype(float).tolist(),
                "matched_attention_field_cosine_vs_clean": cosine,
                "spatial_match_fraction_vs_clean": float(matched.mean()),
                "predicted_position_world": pred.astype(float).tolist(),
                "predicted_position_drift_m": float(np.linalg.norm(pred-clean_pred)),
                "minimum_attention_reconstruction_cosine": float(min(
                    float(capture[key])
                    for key in capture.files
                    if key.endswith("_reconstruction_cosine")
                )),
            }
        )

    clean_centroid = np.asarray(rows[0]["attention_centroid_world"])
    for row in rows:
        row["attention_centroid_drift_m"] = float(
            np.linalg.norm(np.asarray(row["attention_centroid_world"])-clean_centroid)
        )

    rates = 100*np.asarray(SEVERITIES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].plot(rates, 100*np.asarray([x["predicted_position_drift_m"] for x in rows]), marker="o")
    axes[0, 0].set(title="First action position drift", ylabel="cm")
    axes[0, 1].plot(rates, 100*np.asarray([x["attention_centroid_drift_m"] for x in rows]), marker="o")
    axes[0, 1].set(title="Stage-4 attention centroid drift", ylabel="cm")
    axes[1, 0].plot(rates, [x["matched_attention_field_cosine_vs_clean"] for x in rows], marker="o")
    axes[1, 0].set(title="Spatially matched attention-field cosine", ylabel="cosine", ylim=(0, 1.04))
    axes[1, 1].plot(rates, [x["attention_entropy_among_point_keys"] for x in rows], marker="o")
    axes[1, 1].set(title="Attention entropy among point keys", ylabel="normalized entropy", ylim=(0, 1.04))
    for ax in axes.ravel():
        ax.set_xlabel("simulated target failure severity (%)")
        ax.set_xticks(rates)
        ax.grid(alpha=0.25)
    fig.suptitle("How realistic point-cloud damage changes PointACT action attention")
    fig.savefig(RENDERED / "attention_impact_metrics.png", dpi=220)
    plt.close(fig)

    report = {
        "task": "close_fridge",
        "attention_layer": "PTV3 encoder stage 4 block 2",
        "attention_definition": (
            "Direct action-query to point-key softmax attention averaged over heads "
            "and non-state action queries; this is not attention rollout or causal importance."
        ),
        "visualization": {
            "workspace": {"x": X_LIM, "y": Y_LIM, "z": Z_LIM},
            "shared_attention_scale": True,
            "attention_scale_p99_5_clip": float(norm.vmax),
            "attention_power_norm_gamma": float(norm.gamma),
            "same_3d_view_and_bounds": True,
        },
        "trials": rows,
        "capture_metadata": trials,
    }
    (OUTPUT / "attention_3d_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    RENDERED.mkdir(parents=True, exist_ok=True)
    trials, raw, captures = load_all()
    all_attention = np.concatenate([input_attention(x) for x in captures])
    upper = max(float(np.percentile(all_attention, 99.5)), 1e-12)
    norm = PowerNorm(gamma=0.48, vmin=0.0, vmax=upper, clip=True)
    cmap = plt.get_cmap("turbo")
    render_main(raw, captures, norm, cmap)
    render_full_scene(raw, captures, norm, cmap)
    render_corruption_and_change(raw, captures)
    render_target_mask_explanation(raw[0])
    report = metrics_and_ply(trials, raw, captures, norm, cmap)
    print(json.dumps(report["trials"], indent=2))


if __name__ == "__main__":
    main()
