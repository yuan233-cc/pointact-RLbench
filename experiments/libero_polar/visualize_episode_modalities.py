#!/usr/bin/env python3
"""Export readable polar and point-cloud PNGs from a replayed LIBERO episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path, help="Episode directory containing frames/*.npz")
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, default=49)
    parser.add_argument("--overview-frames", type=int, nargs="+", default=[0, 24, 49, 73, 97])
    return parser.parse_args()


def pixel_xy(indices: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    return indices % width, indices // width


def aolp_rgb(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "AoLP" in data:
        angle = np.mod(data["AoLP"], np.pi)
    else:
        angle = np.mod(0.5 * np.arctan2(data["sin2AoLP"], data["cos2AoLP"]), np.pi)
    dolp = np.asarray(data["DoLP"], dtype=np.float32)
    valid = np.asarray(data["aolp_valid_mask"], dtype=bool)
    # AoLP is pi-periodic, so [0, pi) maps once around the hue wheel.  Value is
    # contrast-enhanced DoLP: low-confidence orientations remain dark.
    scale = max(float(np.percentile(dolp[valid], 99)) if np.any(valid) else 0.0, 1e-6)
    hsv = np.zeros((*angle.shape, 3), dtype=np.float32)
    hsv[..., 0] = angle / np.pi
    hsv[..., 1] = valid.astype(np.float32)
    hsv[..., 2] = np.where(valid, np.clip(dolp / scale, 0.12, 1.0), 0.0)
    return hsv_to_rgb(hsv)


def projected_cloud(ax, rgb: np.ndarray, pixels: np.ndarray, colors: np.ndarray, title: str) -> None:
    height, width = rgb.shape[:2]
    ax.imshow(rgb, alpha=0.20)
    x, y = pixel_xy(pixels, width)
    ax.scatter(x, y, s=2.0, c=np.clip(colors, 0, 1), linewidths=0, alpha=0.9)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(title)
    ax.axis("off")


def missing_pixels(data: np.lib.npyio.NpzFile) -> np.ndarray:
    clean = np.asarray(data["clean_pixel_index"], dtype=np.int64)
    retained_sources = np.asarray(data["incomplete_source_pixel_index"], dtype=np.int64)
    return clean[~np.isin(clean, retained_sources)]


def save_polar_pngs(frame_path: Path, output: Path) -> None:
    with np.load(frame_path) as data:
        rgb = data["rgb"]
        dolp = np.asarray(data["DoLP"], dtype=np.float32)
        valid = np.asarray(data["polar_valid_mask"], dtype=bool)
        valid_values = dolp[valid]
        p99 = float(np.percentile(valid_values, 99)) if valid_values.size else 1.0

        plt.imsave(output / "rgb.png", rgb)
        plt.imsave(output / "dolp_physical_0_to_1.png", dolp, cmap="magma", vmin=0, vmax=1)
        plt.imsave(output / "dolp_contrast_p99.png", dolp, cmap="magma", vmin=0, vmax=max(p99, 1e-6))
        plt.imsave(output / "aolp_cyclic_dolp_weighted.png", aolp_rgb(data))

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[1].imshow(dolp, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("DoLP physical scale [0, 1]")
        im = axes[2].imshow(dolp, cmap="magma", vmin=0, vmax=max(p99, 1e-6))
        axes[2].set_title(f"DoLP contrast scale [0, P99={p99:.3f}]")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
        axes[3].imshow(aolp_rgb(data))
        axes[3].set_title("AoLP cyclic hue; brightness = DoLP")
        for ax in axes:
            ax.axis("off")
        fig.savefig(output / "polar_modalities_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def save_projected_pointcloud_png(frame_path: Path, output: Path) -> None:
    with np.load(frame_path) as data:
        rgb = data["rgb"]
        height, width = rgb.shape[:2]
        missing = missing_pixels(data)
        synth = np.asarray(data["filled_synthetic_mask"], dtype=bool)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        projected_cloud(axes[0, 0], rgb, data["clean_pixel_index"], data["clean9"][:, 3:6],
                        f"Clean cloud ({len(data['clean9'])} points)")
        projected_cloud(axes[0, 1], rgb, data["incomplete_current_pixel_index"],
                        data["incomplete9"][:, 3:6], f"Incomplete cloud ({len(data['incomplete9'])} points)")
        axes[0, 2].imshow(rgb, alpha=0.28)
        mx, my = pixel_xy(missing, width)
        axes[0, 2].scatter(mx, my, s=8, c="red", linewidths=0)
        axes[0, 2].set_title(f"Removed / missing source points ({len(missing)})")
        axes[0, 2].set_xlim(0, width)
        axes[0, 2].set_ylim(height, 0)
        axes[0, 2].axis("off")

        projected_cloud(axes[1, 0], rgb, data["filled_current_pixel_index"], data["filled9"][:, 3:6],
                        f"Filled cloud ({len(data['filled9'])} points)")
        axes[1, 1].imshow(rgb, alpha=0.28)
        sx, sy = pixel_xy(data["filled_current_pixel_index"][synth], width)
        axes[1, 1].scatter(sx, sy, s=8, c="#00d5ff", linewidths=0)
        axes[1, 1].set_title(f"Synthetic completion points ({int(synth.sum())})")
        axes[1, 1].set_xlim(0, width)
        axes[1, 1].set_ylim(height, 0)
        axes[1, 1].axis("off")

        axes[1, 2].imshow(rgb, alpha=0.28)
        codes = np.asarray(data["incomplete_corruption_code"])
        palette = {1: ("#ff8c00", "support distortion"), 2: ("#38b000", "wrong target depth"),
                   4: ("#9b5de5", "floating points")}
        for code, (color, label) in palette.items():
            mask = codes == code
            x, y = pixel_xy(data["incomplete_current_pixel_index"][mask], width)
            axes[1, 2].scatter(x, y, s=8, c=color, label=f"{label}: {int(mask.sum())}", linewidths=0)
        axes[1, 2].set_title("Retained corrupted points by type")
        axes[1, 2].set_xlim(0, width)
        axes[1, 2].set_ylim(height, 0)
        axes[1, 2].axis("off")
        axes[1, 2].legend(loc="lower left", fontsize=8)
        fig.savefig(output / "pointcloud_projected_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def set_equal_3d(ax, xyz: np.ndarray) -> None:
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    center = (lo + hi) / 2
    radius = max(float(np.max(hi - lo)) / 2, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def save_3d_pointcloud_png(frame_path: Path, output: Path) -> None:
    with np.load(frame_path) as data:
        clouds = [("Clean", data["clean9"]), ("Incomplete", data["incomplete9"]),
                  ("Filled", data["filled9"])]
        bounds = np.concatenate([cloud[:, :3] for _, cloud in clouds], axis=0)
        fig = plt.figure(figsize=(18, 6), constrained_layout=True)
        for index, (name, cloud) in enumerate(clouds, 1):
            ax = fig.add_subplot(1, 3, index, projection="3d")
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=np.clip(cloud[:, 3:6], 0, 1),
                       s=1.3, linewidths=0, depthshade=False)
            set_equal_3d(ax, bounds)
            ax.view_init(elev=55, azim=-90)
            ax.set_title(f"{name} 9-D cloud: {len(cloud)} points")
            ax.set_xlabel("world x")
            ax.set_ylabel("world y")
            ax.set_zlabel("world z")
        fig.savefig(output / "pointcloud_3d_clean_incomplete_filled.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_overview(episode: Path, frame_numbers: list[int], output: Path) -> None:
    fig, axes = plt.subplots(len(frame_numbers), 6, figsize=(21, 3.5 * len(frame_numbers)), constrained_layout=True)
    if len(frame_numbers) == 1:
        axes = axes[None, :]
    for row, number in enumerate(frame_numbers):
        with np.load(episode / "frames" / f"{number:06d}.npz") as data:
            rgb = data["rgb"]
            dolp = data["DoLP"]
            p99 = max(float(np.percentile(dolp[data["polar_valid_mask"]], 99)), 1e-6)
            axes[row, 0].imshow(rgb)
            axes[row, 1].imshow(dolp, cmap="magma", vmin=0, vmax=p99)
            axes[row, 2].imshow(aolp_rgb(data))
            projected_cloud(axes[row, 3], rgb, data["incomplete_current_pixel_index"],
                            data["incomplete9"][:, 3:6], "Incomplete cloud")
            axes[row, 4].imshow(rgb, alpha=0.28)
            x, y = pixel_xy(missing_pixels(data), rgb.shape[1])
            axes[row, 4].scatter(x, y, s=5, c="red", linewidths=0)
            axes[row, 4].set_title("Missing points")
            synth = data["filled_synthetic_mask"].astype(bool)
            axes[row, 5].imshow(rgb, alpha=0.28)
            x, y = pixel_xy(data["filled_current_pixel_index"][synth], rgb.shape[1])
            axes[row, 5].scatter(x, y, s=5, c="#00d5ff", linewidths=0)
            axes[row, 5].set_title("Synthetic filled points")
            for col in (0, 1, 2, 4, 5):
                axes[row, col].axis("off")
            axes[row, 0].set_title(f"frame {number}: RGB")
            axes[row, 1].set_title(f"DoLP, P99={p99:.3f}")
            axes[row, 2].set_title("AoLP cyclic hue")
    fig.savefig(output / "episode_modalities_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_path = args.episode / "frames" / f"{args.frame:06d}.npz"
    if not frame_path.is_file():
        raise FileNotFoundError(frame_path)
    save_polar_pngs(frame_path, args.output)
    save_projected_pointcloud_png(frame_path, args.output)
    save_3d_pointcloud_png(frame_path, args.output)
    save_overview(args.episode, args.overview_frames, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
