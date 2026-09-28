"""Render one RLBench polar depth-fill frame as a PNG and interactive 3D HTML.

Example (from the repository root):
  python experiments/10_rlbench/visualize_polar_depth_fill.py \
    --frame-index 0 --target-label 82 \
    --output phone_polar_depth_fill_frame0_20260921
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import msgpack
import msgpack_numpy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from create_stack_wine_10episode_failure_dataset import CAMERA_EXTRINSICS


msgpack_numpy.patch()
ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / (
    "robot_data/rlbench/lerobot_point_lmdb/"
    "phone_on_base_1episode_polar_housecat_filled9_seed24_v3"
)
FRAMES = ROOT.parent / (
    "rlbench_custom_render/RLBench/output/material_profile_v3_aligned/"
    "phone_on_base_episode1/frames_spp512"
)


def read_cloud(dataset: Path, dirname: str, key: bytes) -> np.ndarray:
    env = lmdb.open(str(dataset / dirname), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as tx:
            packed = tx.get(key)
            if packed is None:
                raise KeyError(f"Missing {key!r} in {dirname}")
            return np.asarray(msgpack.unpackb(packed), dtype=np.float32)
    finally:
        env.close()


def display_coordinates(world_xyz: np.ndarray) -> np.ndarray:
    """Camera right, camera depth, camera up; all three axes are in meters."""
    camera = (world_xyz - CAMERA_EXTRINSICS[:3, 3]) @ CAMERA_EXTRINSICS[:3, :3]
    return np.column_stack((camera[:, 0], camera[:, 2], -camera[:, 1]))


def common_limits(points: np.ndarray, margin: float = 0.06) -> list[tuple[float, float]]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    pad = np.maximum((hi - lo) * margin, 0.005)
    return list(zip(lo - pad, hi + pad))


def draw_cloud(ax, xyz, rgb, limits, title, *, new_xyz=None, point_size=2.0):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=np.clip(rgb, 0, 1), s=point_size, depthshade=False, linewidths=0)
    if new_xyz is not None and len(new_xyz):
        ax.scatter(new_xyz[:, 0], new_xyz[:, 1], new_xyz[:, 2],
                   c="#db1684", s=point_size * 1.5, depthshade=False,
                   linewidths=0, label="filled")
    for setter, bounds in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits):
        setter(*bounds)
    ax.set_box_aspect([b[1] - b[0] for b in limits], zoom=0.88)
    ax.set_xlabel("right (m)", fontsize=9)
    ax.set_ylabel("depth (m)", fontsize=9)
    ax.set_zlabel("up (m)", fontsize=9)
    ax.view_init(elev=22, azim=-62)
    ax.tick_params(labelsize=7, pad=0)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_facecolor("#fbfcff")


def add_plotly_cloud(fig, row, col, xyz, rgb, name, *, size=2.0,
                     color_override=None, show_legend=False):
    colors = color_override or [
        f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
        for r, g, b in np.clip(rgb, 0, 1)
    ]
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker=dict(size=size, color=colors, opacity=0.96),
        name=name, showlegend=show_legend,
        hovertemplate="right=%{x:.3f} m<br>depth=%{y:.3f} m<br>up=%{z:.3f} m<extra></extra>",
    ), row=row, col=col)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--frames", type=Path, default=FRAMES)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--target-label", type=int, default=82,
                        help="Object-mask label to isolate in the lower 3D row")
    parser.add_argument("--output", type=Path, default=ROOT / "polar_depth_fill_visualization")
    args = parser.parse_args()
    key = f"{args.episode_index}-{args.frame_index}".encode("ascii")
    stem = f"{args.frame_index:06d}"
    original = read_cloud(args.dataset, "points_frontview_polar_incomplete9", key)
    filled = read_cloud(args.dataset, "points_frontview_polar_filled9", key)
    original_pixels = np.load(args.dataset / "point_pixel_indices" / f"{stem}.npy")
    filled_pixels = np.load(args.dataset / "point_pixel_indices_filled" / f"{stem}.npy")
    fill_mask = np.load(args.dataset / "point_depth_filled_mask" / f"{stem}.npy")
    with np.load(args.frames / f"{stem}.npz") as frame:
        rgb_image = np.asarray(frame["rgb"], dtype=np.uint8)
        object_mask = np.asarray(frame["object_mask"])
    if not (len(original) == len(original_pixels)
            and len(filled) == len(filled_pixels) == len(fill_mask)
            and np.array_equal(original, filled[:len(original)])
            and np.all(~fill_mask[:len(original)])):
        raise ValueError("The original and filled point rows are not aligned")

    target = object_mask == args.target_label
    if not target.any():
        raise ValueError(f"Object-mask label {args.target_label} is absent")
    ys, xs = np.where(target)
    x0, x1 = max(0, int(xs.min()) - 12), min(rgb_image.shape[1], int(xs.max()) + 13)
    y0, y1 = max(0, int(ys.min()) - 12), min(rgb_image.shape[0], int(ys.max()) + 13)
    target_flat = target.reshape(-1)
    old_target = target_flat[original_pixels]
    filled_target = target_flat[filled_pixels]

    old_xyz = display_coordinates(original[:, :3])
    all_xyz = display_coordinates(filled[:, :3])
    full_limits = common_limits(all_xyz)
    target_limits = common_limits(all_xyz[filled_target], margin=0.10)
    args.output.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 10), dpi=160, facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[0.8, 1, 1],
                           height_ratios=[1, 1], wspace=0.02, hspace=0.08)
    ax_rgb = fig.add_subplot(grid[0, 0])
    ax_rgb.imshow(rgb_image, interpolation="nearest")
    ax_rgb.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, ec="#db1684", lw=1.7))
    ax_rgb.set_title("Original RGB · frame 0", fontsize=12)
    ax_rgb.axis("off")
    ax_crop = fig.add_subplot(grid[1, 0])
    ax_crop.imshow(rgb_image[y0:y1, x0:x1], interpolation="nearest")
    ax_crop.set_title(f"RGB crop · object label {args.target_label}", fontsize=12)
    ax_crop.axis("off")
    draw_cloud(fig.add_subplot(grid[0, 1], projection="3d"),
               old_xyz, original[:, 3:6], full_limits,
               f"Incomplete cloud · {len(original):,} points")
    draw_cloud(fig.add_subplot(grid[0, 2], projection="3d"),
               old_xyz, original[:, 3:6], full_limits,
               f"Filled cloud · {len(filled):,} points",
               new_xyz=all_xyz[fill_mask])
    draw_cloud(fig.add_subplot(grid[1, 1], projection="3d"),
               old_xyz[old_target], original[old_target, 3:6], target_limits,
               f"Incomplete object · {int(old_target.sum()):,} points", point_size=9)
    draw_cloud(fig.add_subplot(grid[1, 2], projection="3d"),
               old_xyz[old_target], original[old_target, 3:6], target_limits,
               f"Filled object · {int(filled_target.sum()):,} points",
               new_xyz=all_xyz[filled_target & fill_mask], point_size=9)
    fig.text(0.5, 0.025,
             "Magenta = depth-estimated points; their RGB and polarization come from the same image pixel. "
             "Both point-cloud panels share the same 3D view and scale.",
             ha="center", va="center", fontsize=10, color="#444")
    png = args.output / f"frame_{args.frame_index:06d}_comparison.png"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

    titles = ["Original RGB", "Incomplete cloud", "Filled cloud · magenta = new",
              "RGB crop", "Incomplete object", "Filled object · magenta = new"]
    interactive = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "scene"}],
               [{"type": "xy"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=titles, column_widths=[0.26, 0.37, 0.37],
        horizontal_spacing=0.03, vertical_spacing=0.09,
    )
    interactive.add_trace(go.Image(z=rgb_image), row=1, col=1)
    interactive.add_trace(go.Image(z=rgb_image[y0:y1, x0:x1]), row=2, col=1)
    add_plotly_cloud(interactive, 1, 2, old_xyz, original[:, 3:6], "original")
    add_plotly_cloud(interactive, 1, 3, old_xyz, original[:, 3:6], "retained")
    add_plotly_cloud(interactive, 1, 3, all_xyz[fill_mask], filled[fill_mask, 3:6],
                     "depth-filled points", color_override="#db1684", show_legend=True)
    add_plotly_cloud(interactive, 2, 2, old_xyz[old_target],
                     original[old_target, 3:6], "original object", size=3.2)
    add_plotly_cloud(interactive, 2, 3, old_xyz[old_target],
                     original[old_target, 3:6], "retained object", size=3.2)
    add_plotly_cloud(interactive, 2, 3, all_xyz[filled_target & fill_mask],
                     filled[filled_target & fill_mask, 3:6], "filled object points",
                     size=3.5, color_override="#db1684", show_legend=True)
    for scene, limits in zip(("scene", "scene2", "scene3", "scene4"),
                             (full_limits, full_limits, target_limits, target_limits)):
        interactive.update_layout(**{scene: dict(
            xaxis=dict(title="camera right (m)", range=list(limits[0])),
            yaxis=dict(title="depth (m)", range=list(limits[1])),
            zaxis=dict(title="camera up (m)", range=list(limits[2])),
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.85)),
        )})
    interactive.update_layout(height=1050, width=1600, margin=dict(l=20, r=20, t=90, b=30),
                              paper_bgcolor="white", showlegend=True,
                              title=f"Phone polar depth fill · episode {args.episode_index}, frame {args.frame_index}")
    interactive.update_xaxes(visible=False)
    interactive.update_yaxes(visible=False)
    html = args.output / f"frame_{args.frame_index:06d}_interactive.html"
    interactive.write_html(html, include_plotlyjs=True, full_html=True)
    summary = {
        "episode_index": args.episode_index, "frame_index": args.frame_index,
        "object_mask_label": args.target_label,
        "incomplete_points": len(original), "filled_points_total": len(filled),
        "new_depth_estimated_points": int(fill_mask.sum()),
        "incomplete_target_points": int(old_target.sum()),
        "filled_target_points_total": int(filled_target.sum()),
        "new_target_points": int(np.sum(filled_target & fill_mask)),
        "rgb_crop_xyxy": [x0, y0, x1, y1],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"png": str(png), "interactive_html": str(html), **summary}, indent=2))


if __name__ == "__main__":
    main()
