"""Render retained and removed points across one incomplete-data episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import msgpack
import msgpack_numpy
import numpy as np


msgpack_numpy.patch()
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(CJK_FONT).exists():
    font_manager.fontManager.addfont(CJK_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=CJK_FONT
    ).get_name()
plt.rcParams["axes.unicode_minus"] = False


def load_episode(env: lmdb.Environment, episode: int) -> list[np.ndarray]:
    clouds = []
    with env.begin(buffers=True) as txn:
        frame = 0
        while True:
            value = txn.get(f"{episode}-{frame}".encode("ascii"))
            if value is None:
                break
            clouds.append(np.asarray(msgpack.unpackb(value), dtype=np.float32))
            frame += 1
    if not clouds:
        raise KeyError(f"No frames found for episode {episode}")
    return clouds


def retained_mask(source: np.ndarray, incomplete: np.ndarray) -> np.ndarray:
    retained_rows = {row.tobytes() for row in np.ascontiguousarray(incomplete)}
    return np.fromiter(
        (row.tobytes() in retained_rows for row in np.ascontiguousarray(source)),
        dtype=bool,
        count=len(source),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--incomplete", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_env = lmdb.open(
        str(args.source / "points_frontview"), readonly=True, lock=False
    )
    incomplete_env = lmdb.open(
        str(args.incomplete / "points_frontview"), readonly=True, lock=False
    )
    try:
        source_frames = load_episode(source_env, args.episode)
        incomplete_frames = load_episode(incomplete_env, args.episode)
    finally:
        source_env.close()
        incomplete_env.close()
    if len(source_frames) != len(incomplete_frames):
        raise ValueError("Source and incomplete episode lengths differ")

    all_xyz = np.concatenate([cloud[:, :3] for cloud in source_frames], axis=0)
    low, high = np.percentile(all_xyz, [0.2, 99.8], axis=0)
    pad = np.maximum(0.045 * (high - low), 0.012)
    low, high = low - pad, high + pad

    columns = min(3, len(source_frames))
    rows = int(np.ceil(len(source_frames) / columns))
    fig = plt.figure(
        figsize=(6.6 * columns, 5.5 * rows), constrained_layout=True
    )
    for index, (source, incomplete) in enumerate(
        zip(source_frames, incomplete_frames), start=1
    ):
        keep = retained_mask(source, incomplete)
        removed = ~keep
        expected_removed = int(round(len(source) * 0.25))
        if removed.sum() != expected_removed:
            raise AssertionError(
                f"frame {index - 1}: recovered {removed.sum()} removed points, "
                f"expected {expected_removed}"
            )

        ax = fig.add_subplot(rows, columns, index, projection="3d")
        xyz = source[:, :3]
        rgb = np.clip(source[:, 3:6], 0, 1)
        ax.scatter(
            xyz[keep, 0], xyz[keep, 1], xyz[keep, 2],
            c=rgb[keep], s=2.0, alpha=0.28, linewidths=0,
            depthshade=False, rasterized=True,
        )
        ax.scatter(
            xyz[removed, 0], xyz[removed, 1], xyz[removed, 2],
            c="#ff00b8", s=4.2, alpha=0.9, linewidths=0,
            depthshade=False, rasterized=True,
        )
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_zlim(low[2], high[2])
        ax.set_box_aspect(np.maximum(high - low, 1e-3))
        ax.view_init(elev=24, azim=-58)
        ax.set_xlabel("x", labelpad=-7)
        ax.set_ylabel("y", labelpad=-7)
        ax.set_zlabel("z", labelpad=-7)
        ax.tick_params(labelsize=6, pad=-2)
        ax.grid(alpha=0.16)
        ax.set_title(
            f"frame {index - 1}: 删除 {removed.sum():,}/{len(source):,} "
            f"({removed.mean():.1%})",
            fontsize=11,
        )

    fig.suptitle(
        f"{args.task} — episode {args.episode} 的连续帧\n"
        "洋红色=删除点，原色半透明=保留点；所有帧共享同一世界坐标空洞场",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
