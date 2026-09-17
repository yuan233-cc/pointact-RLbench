"""Render full-scene and target-local 3-D PointACT action attention."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.spatial import cKDTree

from ptv3_feature_viz import write_ply


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_wine_umbrella_realistic_missing_20260917"
INPUTS = OUTPUT / "attention_inputs"
CAPTURES = OUTPUT / "attention_captures"
RENDERED = OUTPUT / "rendered"
PLY_DIR = OUTPUT / "pointclouds_3d"
SEVERITIES = (0.0, 0.25, 0.50, 0.75)
TASKS = {
    "stack_wine": {"title": "Stack wine", "target": "wine bottle"},
    "take_umbrella_out_of_umbrella_stand": {
        "title": "Umbrella out",
        "target": "umbrella",
    },
}
ATTENTION_KEY = "action_attention_stage4_block2_point_weights"
ATTENTION_COORD_KEY = "action_attention_stage4_block2_point_coordinates"
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(CJK_FONT).exists():
    font_manager.fontManager.addfont(CJK_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=CJK_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


def as_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    if rgb.size and rgb.max() > 1.5:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def world_input(capture) -> np.ndarray:
    center = np.asarray(capture["scene_center"], dtype=np.float32).reshape(-1, 3)[0]
    return capture["input_coordinates"].astype(np.float32) + center


def world_stage4(capture) -> np.ndarray:
    center = np.asarray(capture["scene_center"], dtype=np.float32).reshape(-1, 3)[0]
    return capture[ATTENTION_COORD_KEY].astype(np.float32) + center


def input_attention(capture) -> np.ndarray:
    return capture[ATTENTION_KEY].astype(np.float32)[
        capture["input_to_stage4"].astype(np.int64)
    ]


def predicted_world(capture) -> np.ndarray:
    center = np.asarray(capture["scene_center"], dtype=np.float32).reshape(-1, 3)[0]
    return capture["predicted_position"].astype(np.float32) + center


def bounds(points: np.ndarray, q: tuple[float, float] = (.2, 99.8), minimum=.015):
    low, high = np.percentile(points, q, axis=0)
    pad = np.maximum(.05*(high-low), minimum)
    return low-pad, high+pad


def configure_3d(ax, limits, elev=25, azim=-58) -> None:
    low, high = limits
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_box_aspect(np.maximum(high-low, 1e-3))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x", labelpad=-7)
    ax.set_ylabel("y", labelpad=-7)
    ax.set_zlabel("z", labelpad=-7)
    ax.tick_params(labelsize=6, pad=-2)
    ax.grid(alpha=.16)


def inside_bounds(xyz: np.ndarray, limits) -> np.ndarray:
    low, high = limits
    return np.isfinite(xyz).all(axis=-1) & np.all((xyz >= low) & (xyz <= high), axis=-1)


def draw_box(ax, limits, color="#00e5ff", linewidth=1.35) -> None:
    low, high = limits
    corners = np.asarray([
        [x, y, z]
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ])
    for first in range(8):
        for axis in range(3):
            second = first ^ (1 << (2-axis))
            if first < second:
                segment = corners[[first, second]]
                ax.plot(
                    segment[:, 0], segment[:, 1], segment[:, 2],
                    color=color, linewidth=linewidth, alpha=.95,
                )


def load_task(task_name: str, trial_map: dict):
    raw, captures, trials = [], [], []
    for severity in SEVERITIES:
        trial = trial_map[(task_name, severity)]
        raw.append(np.load(INPUTS / trial["file"]))
        captures.append(np.load(CAPTURES / f"capture_{trial['capture_index']:06d}.npz"))
        trials.append(trial)
    return raw, captures, trials


def render_global(task_name: str, raw, captures, norm, cmap) -> None:
    all_xyz = np.concatenate([
        sample["corrupted_points"][np.isfinite(sample["corrupted_points"]).all(axis=-1)]
        for sample in raw
    ])
    scene_limits = bounds(all_xyz)
    clean_pred = predicted_world(captures[0])
    fig = plt.figure(figsize=(20, 10.7), constrained_layout=True)
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for col in range(4):
            axes[row, col] = fig.add_subplot(2, 4, row*4+col+1, projection="3d")

    for col, (severity, sample, capture) in enumerate(zip(SEVERITIES, raw, captures)):
        corrupted = sample["corrupted_points"]
        valid = np.isfinite(corrupted).all(axis=-1)
        xyz = corrupted[valid]
        rgb = as_rgb(sample["rgb"])[valid]
        axes[0, col].scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=.40,
            linewidths=0, depthshade=False, rasterized=True,
        )
        axes[0, col].set_title(
            f"缺失强度 {severity:.0%}\n完整场景有效点 {len(xyz):,}", fontsize=11
        )

        context = xyz[::5]
        axes[1, col].scatter(
            context[:, 0], context[:, 1], context[:, 2], c="#999999",
            s=.25, alpha=.10, linewidths=0, depthshade=False, rasterized=True,
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
        drift = 100*float(np.linalg.norm(pred-clean_pred))
        axes[1, col].set_title(
            f"PTV3 全局 action attention\n首动作位置漂移 {drift:.1f} cm", fontsize=10
        )
        configure_3d(axes[0, col], scene_limits)
        configure_3d(axes[1, col], scene_limits)

    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[1]),
        location="bottom", shrink=.72, pad=.025, aspect=45,
    )
    cb.set_label(
        "Stage-4 Block-2 直接 action-query → point-key attention（四列共用绝对色标）"
    )
    fig.text(.008, .73, "完整前相机 3D RGB 点云", rotation=90, va="center", fontsize=13, weight="bold")
    fig.text(.008, .27, "完整场景上的全局 action attention", rotation=90, va="center", fontsize=13, weight="bold")
    fig.suptitle(
        f"{TASKS[task_name]['title']}：点云缺失对完整 3D 场景及 action-expert attention 的影响\n"
        "同一场景、视角和色标；白色星号为第一个预测动作位置",
        fontsize=16,
    )
    fig.savefig(RENDERED / f"{task_name}_full_3d_pointcloud_and_attention.png", dpi=220)
    plt.close(fig)


def render_workspace_closeup(task_name: str, raw, captures, norm, cmap) -> None:
    """Crop away walls/background and enlarge the actual model task workspace."""
    clean_model_xyz = world_input(captures[0])
    low = clean_model_xyz.min(axis=0)
    high = clean_model_xyz.max(axis=0)
    task_pad = np.maximum(.06*(high-low), np.asarray([.025, .025, .020]))
    task_limits = (low-task_pad, high+task_pad)

    clean_points = raw[0]["clean_points"]
    target = raw[0]["target_mask"].astype(bool)
    target_valid = target & np.isfinite(clean_points).all(axis=-1)
    target_xyz = clean_points[target_valid]
    target_low = target_xyz.min(axis=0)
    target_high = target_xyz.max(axis=0)
    target_pad = np.maximum(.025*(target_high-target_low), .004)
    target_limits = (target_low-target_pad, target_high+target_pad)

    clean_pred = predicted_world(captures[0])
    fig = plt.figure(figsize=(20, 10.7), constrained_layout=True)
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for col in range(4):
            axes[row, col] = fig.add_subplot(2, 4, row*4+col+1, projection="3d")

    for col, (severity, sample, capture) in enumerate(zip(SEVERITIES, raw, captures)):
        corrupted = sample["corrupted_points"]
        select = inside_bounds(corrupted, task_limits)
        xyz = corrupted[select]
        rgb = as_rgb(sample["rgb"])[select]
        axes[0, col].scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1.25,
            linewidths=0, depthshade=False, rasterized=True,
        )
        draw_box(axes[0, col], target_limits)
        axes[0, col].set_title(
            f"缺失强度 {severity:.0%}\n只显示任务工作区：{len(xyz):,} 点",
            fontsize=11,
        )

        model_xyz = world_input(capture)
        attention = input_attention(capture)
        model_select = inside_bounds(model_xyz, task_limits)
        context = xyz[::3]
        axes[1, col].scatter(
            context[:, 0], context[:, 1], context[:, 2], c="#a8a8a8",
            s=.35, alpha=.07, linewidths=0, depthshade=False, rasterized=True,
        )
        axes[1, col].scatter(
            model_xyz[model_select, 0], model_xyz[model_select, 1], model_xyz[model_select, 2],
            c=attention[model_select], cmap=cmap, norm=norm, s=13,
            linewidths=0, depthshade=False, rasterized=True,
        )
        draw_box(axes[1, col], target_limits)
        pred = predicted_world(capture)
        axes[1, col].scatter(
            pred[0], pred[1], pred[2], marker="*", s=210,
            c="white", edgecolors="black", linewidths=1.1,
        )
        drift = 100*float(np.linalg.norm(pred-clean_pred))
        axes[1, col].set_title(
            f"放大的工作区 action attention\n首动作位置漂移 {drift:.1f} cm",
            fontsize=10,
        )
        configure_3d(axes[0, col], task_limits, elev=24, azim=-58)
        configure_3d(axes[1, col], task_limits, elev=24, azim=-58)

    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[1]),
        location="bottom", shrink=.72, pad=.025, aspect=45,
    )
    cb.set_label(
        "Stage-4 Block-2 直接 action-query → point-key attention（四列共用绝对色标）"
    )
    fig.text(.008, .73, "裁去墙面后的任务工作区", rotation=90,
             va="center", fontsize=13, weight="bold")
    fig.text(.008, .27, "放大的全局 action attention", rotation=90,
             va="center", fontsize=13, weight="bold")
    fig.suptitle(
        f"{TASKS[task_name]['title']}：工作区域近景（青色框 = {TASKS[task_name]['target']}）\n"
        "已移除远处墙面和工作区外点；四列使用相同视角、坐标范围及 attention 色标",
        fontsize=16,
    )
    fig.savefig(RENDERED / f"{task_name}_workspace_closeup_pointcloud_and_attention.png", dpi=220)
    plt.close(fig)


def render_target(task_name: str, raw, captures, norm, cmap) -> None:
    clean = raw[0]["clean_points"]
    target = raw[0]["target_mask"].astype(bool)
    valid_target = target & np.isfinite(clean).all(axis=-1)
    target_xyz = clean[valid_target]
    target_limits = bounds(target_xyz, (.5, 99.5), minimum=.008)

    fig = plt.figure(figsize=(20, 10.7), constrained_layout=True)
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for col in range(4):
            axes[row, col] = fig.add_subplot(2, 4, row*4+col+1, projection="3d")
    for col, (severity, sample, capture) in enumerate(zip(SEVERITIES, raw, captures)):
        corrupted = sample["corrupted_points"]
        labels = sample["corruption_labels"]
        surviving = target & np.isfinite(corrupted).all(axis=-1)
        missing = target & ~np.isfinite(corrupted).all(axis=-1)
        colors = as_rgb(sample["rgb"])
        xyz = corrupted[surviving]
        axes[0, col].scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors[surviving], s=4,
            linewidths=0, depthshade=False, rasterized=True,
        )
        ghost = clean[missing]
        if len(ghost):
            axes[0, col].scatter(
                ghost[:, 0], ghost[:, 1], ghost[:, 2], c="#e31a1c", marker="x",
                s=7, linewidths=.45, depthshade=False, rasterized=True,
            )
        axes[0, col].set_title(
            f"{severity:.0%}: {TASKS[task_name]['target']} 可见表面\n"
            f"有效 {surviving.sum():,}；红叉缺失 {missing.sum():,}", fontsize=10
        )

        model_xyz = world_input(capture)
        attention = input_attention(capture)
        low, high = target_limits
        in_box = np.all((model_xyz >= low) & (model_xyz <= high), axis=1)
        axes[1, col].scatter(
            model_xyz[in_box, 0], model_xyz[in_box, 1], model_xyz[in_box, 2],
            c=attention[in_box], cmap=cmap, norm=norm, s=18,
            linewidths=0, depthshade=False, rasterized=True,
        )
        axes[1, col].set_title(
            f"目标局部 attention（全局同色标）\n框内 PTV3 输入点 {in_box.sum():,}", fontsize=10
        )
        configure_3d(axes[0, col], target_limits, elev=18, azim=-55)
        configure_3d(axes[1, col], target_limits, elev=18, azim=-55)

    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[1]),
        location="bottom", shrink=.72, pad=.025, aspect=45,
    )
    cb.set_label("与全局图相同的绝对 action-attention 色标")
    fig.suptitle(
        f"{TASKS[task_name]['title']}：目标物体局部点云与 attention\n"
        "上排红叉表示被置为无效的原始 3D 位置；下排不是重新归一化后的局部热图",
        fontsize=16,
    )
    fig.savefig(RENDERED / f"{task_name}_target_3d_attention_zoom.png", dpi=220)
    plt.close(fig)


def render_mask(task_name: str, sample) -> None:
    rgb = as_rgb(sample["rgb"])
    mask = sample["target_mask"].astype(bool)
    points = sample["clean_points"]
    valid = mask & np.isfinite(points).all(axis=-1)
    xyz = points[valid]
    colors = rgb[valid]
    target_limits = bounds(xyz, (.5, 99.5), minimum=.008)
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask] = (0, .95, .95, .48)
    ax0.imshow(rgb)
    ax0.imshow(overlay)
    ax0.set_title(
        f"实验 target mask（青色）\n可见像素 {mask.sum():,}/{mask.size:,} ({100*mask.mean():.2f}%)"
    )
    ax0.axis("off")
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=4,
                linewidths=0, depthshade=False, rasterized=True)
    ax1.set_title(f"mask 对应的单前相机可见 3D 表面：{len(xyz):,} 点")
    configure_3d(ax1, target_limits, elev=18, azim=-55)
    fig.suptitle(
        f"{TASKS[task_name]['title']}：target mask 是施加模拟损坏的物体区域，"
        "不是模型预测或 attention",
        fontsize=14,
    )
    fig.savefig(RENDERED / f"{task_name}_target_mask_explained.png", dpi=220)
    plt.close(fig)


def task_metrics(task_name: str, raw, captures, norm, cmap) -> list[dict]:
    task_ply = PLY_DIR / task_name
    task_ply.mkdir(parents=True, exist_ok=True)
    clean_pred = predicted_world(captures[0])
    clean_input = world_input(captures[0])
    clean_attention = input_attention(captures[0])
    tree = cKDTree(clean_input)
    rows = []
    for severity, sample, capture in zip(SEVERITIES, raw, captures):
        corrupted = sample["corrupted_points"]
        valid = np.isfinite(corrupted).all(axis=-1)
        raw_xyz = corrupted[valid]
        raw_rgb = as_rgb(sample["rgb"])[valid]
        model_xyz = world_input(capture)
        attention = input_attention(capture)
        attention_rgb = cmap(norm(attention))[:, :3]
        write_ply(task_ply / f"severity_{severity:.2f}_full_rgb.ply", raw_xyz, raw_rgb)
        write_ply(
            task_ply / f"severity_{severity:.2f}_attention_stage4_block2.ply",
            model_xyz,
            attention_rgb,
        )
        distance, nearest = tree.query(model_xyz)
        matched = distance <= .015
        if matched.any():
            a = clean_attention[nearest[matched]].astype(float)
            b = attention[matched].astype(float)
            cosine = float(np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-15))
        else:
            cosine = float("nan")
        stage_xyz = world_stage4(capture)
        stage_attention = capture[ATTENTION_KEY].astype(float)
        probability = stage_attention/max(stage_attention.sum(), 1e-15)
        centroid = np.sum(stage_xyz*probability[:, None], axis=0)
        entropy = float(
            -np.sum(probability*np.log(np.maximum(probability, 1e-15)))
            / max(np.log(len(probability)), 1e-15)
        )
        reconstruction = [
            float(capture[key]) for key in capture.files if key.endswith("_reconstruction_cosine")
        ]
        rows.append({
            "severity": severity,
            "raw_valid_camera_points": int(valid.sum()),
            "target_visible_pixels": int(sample["target_mask"].sum()),
            "ptv3_input_points": int(len(model_xyz)),
            "stage4_tokens": int(len(stage_attention)),
            "predicted_position_world": predicted_world(capture).astype(float).tolist(),
            "predicted_position_drift_m": float(np.linalg.norm(predicted_world(capture)-clean_pred)),
            "attention_centroid_world": centroid.astype(float).tolist(),
            "matched_attention_field_cosine_vs_clean": cosine,
            "spatial_match_fraction_vs_clean": float(matched.mean()),
            "attention_entropy_among_point_keys": entropy,
            "minimum_attention_reconstruction_cosine": float(min(reconstruction)),
        })
    clean_centroid = np.asarray(rows[0]["attention_centroid_world"])
    for row in rows:
        row["attention_centroid_drift_m"] = float(
            np.linalg.norm(np.asarray(row["attention_centroid_world"])-clean_centroid)
        )
    return rows


def render_metric_comparison(report: dict) -> None:
    rates = 100*np.asarray(SEVERITIES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for task_name, meta in TASKS.items():
        rows = report["tasks"][task_name]["trials"]
        axes[0, 0].plot(rates, 100*np.asarray([x["predicted_position_drift_m"] for x in rows]), marker="o", label=meta["title"])
        axes[0, 1].plot(rates, 100*np.asarray([x["attention_centroid_drift_m"] for x in rows]), marker="o", label=meta["title"])
        axes[1, 0].plot(rates, [x["matched_attention_field_cosine_vs_clean"] for x in rows], marker="o", label=meta["title"])
        axes[1, 1].plot(rates, [x["attention_entropy_among_point_keys"] for x in rows], marker="o", label=meta["title"])
    axes[0, 0].set(title="First action position drift", ylabel="cm")
    axes[0, 1].set(title="Stage-4 attention centroid drift", ylabel="cm")
    axes[1, 0].set(title="Spatially matched attention cosine vs clean", ylabel="cosine", ylim=(0, 1.04))
    axes[1, 1].set(title="Attention entropy among point keys", ylabel="normalized entropy", ylim=(0, 1.04))
    for ax in axes.ravel():
        ax.set_xlabel("simulated target failure severity (%)")
        ax.set_xticks(rates)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Point-cloud damage effects on true PointACT action attention")
    fig.savefig(RENDERED / "attention_impact_metrics_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    RENDERED.mkdir(parents=True, exist_ok=True)
    PLY_DIR.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((OUTPUT / "attention_trials.json").read_text(encoding="utf-8"))
    trial_map = {(item["task"], float(item["severity"])): item for item in metadata["trials"]}
    report = {
        "attention_layer": "PTV3 encoder stage 4 block 2",
        "attention_definition": metadata["attention_definition"],
        "warning": "Direct attention is descriptive, not a causal-importance score.",
        "tasks": {},
    }
    for task_name in TASKS:
        raw, captures, trials = load_task(task_name, trial_map)
        all_attention = np.concatenate([input_attention(capture) for capture in captures])
        upper = max(float(np.percentile(all_attention, 99.5)), 1e-12)
        norm = PowerNorm(gamma=.48, vmin=0, vmax=upper, clip=True)
        cmap = plt.get_cmap("turbo")
        render_global(task_name, raw, captures, norm, cmap)
        render_workspace_closeup(task_name, raw, captures, norm, cmap)
        render_target(task_name, raw, captures, norm, cmap)
        render_mask(task_name, raw[0])
        rows = task_metrics(task_name, raw, captures, norm, cmap)
        report["tasks"][task_name] = {
            "shared_attention_scale_p99_5": upper,
            "trials": rows,
            "capture_metadata": trials,
        }
    render_metric_comparison(report)
    (OUTPUT / "attention_3d_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v["trials"] for k, v in report["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
