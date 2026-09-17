"""Visualize one original stored point-cloud frame for every RLBench task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import msgpack
import msgpack_numpy
import numpy as np

from ptv3_feature_viz import write_ply


msgpack_numpy.patch()
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(CJK_FONT).exists():
    font_manager.fontManager.addfont(CJK_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=CJK_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

TASK_SLUGS = {
    0: "close_box",
    1: "close_laptop_lid",
    2: "toilet_seat_down",
    3: "sweep_to_dustpan",
    4: "close_fridge",
    5: "phone_on_base",
    6: "take_umbrella_out_of_umbrella_stand",
    7: "take_frame_off_hanger",
    8: "stack_wine",
    9: "water_plants",
}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_frame(video_path: Path) -> np.ndarray:
    with av.open(str(video_path)) as container:
        return next(container.decode(video=0)).to_ndarray(format="rgb24")


def cloud_bounds(xyz: np.ndarray):
    low, high = np.percentile(xyz, [.2, 99.8], axis=0)
    pad = np.maximum(.045*(high-low), .012)
    return low-pad, high+pad


def configure_3d(ax, limits) -> None:
    low, high = limits
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_box_aspect(np.maximum(high-low, 1e-3))
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("x", labelpad=-7)
    ax.set_ylabel("y", labelpad=-7)
    ax.set_zlabel("z", labelpad=-7)
    ax.tick_params(labelsize=6, pad=-2)
    ax.grid(alpha=.16)


def load_samples(dataset: Path) -> list[dict]:
    tasks = {item["task"]: item["task_index"] for item in jsonl(dataset/"meta"/"tasks.jsonl")}
    episodes = jsonl(dataset/"meta"/"episodes.jsonl")
    first_episode = {}
    for episode in episodes:
        task_index = tasks[episode["tasks"][0]]
        first_episode.setdefault(task_index, episode["episode_index"])

    samples = []
    env = lmdb.open(str(dataset/"points_frontview"), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(buffers=True) as txn:
            for task_index in sorted(first_episode):
                episode_index = first_episode[task_index]
                point_key = f"{episode_index}-0"
                value = txn.get(point_key.encode("ascii"))
                if value is None:
                    raise KeyError(point_key)
                cloud = np.asarray(msgpack.unpackb(value), dtype=np.float32)
                video_path = (
                    dataset/"videos"/"chunk-000"/"observation.images.front_image"
                    /f"episode_{episode_index:06d}.mp4"
                )
                samples.append({
                    "task_index": task_index,
                    "task": TASK_SLUGS[task_index],
                    "episode_index": episode_index,
                    "point_key": point_key,
                    "cloud": cloud,
                    "rgb": first_frame(video_path),
                })
    finally:
        env.close()
    return samples


def render_individual(
    sample: dict, output: Path, variant_label: str, file_prefix: str
) -> dict:
    cloud = sample["cloud"]
    xyz, colors = cloud[:, :3], np.clip(cloud[:, 3:6], 0, 1)
    limits = cloud_bounds(xyz)
    fig = plt.figure(figsize=(13, 6.2), constrained_layout=True)
    ax_rgb = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_rgb.imshow(sample["rgb"])
    ax_rgb.set_title("训练集 front RGB（第一帧）")
    ax_rgb.axis("off")
    ax_3d.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=3.5,
        linewidths=0, depthshade=False, rasterized=True,
    )
    ax_3d.set_title(f"{variant_label}：{len(xyz):,} 个 1 cm voxel 点")
    configure_3d(ax_3d, limits)
    fig.suptitle(
        f"{sample['task']} — episode {sample['episode_index']}, key {sample['point_key']}\n"
        "单前相机可见表面；不是物体 360° CAD 全表面",
        fontsize=14,
    )
    target = output/f"{file_prefix}_{sample['task']}_one_frame.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    write_ply(output/f"{file_prefix}_{sample['task']}_one_frame.ply", xyz, colors)
    return {
        "task_index": sample["task_index"],
        "task": sample["task"],
        "episode_index": sample["episode_index"],
        "frame_index": 0,
        "point_key": sample["point_key"],
        "stored_points": len(xyz),
        "xyz_min": xyz.min(axis=0).astype(float).tolist(),
        "xyz_max": xyz.max(axis=0).astype(float).tolist(),
        "png": target.name,
        "ply": f"{file_prefix}_{sample['task']}_one_frame.ply",
    }


def render_contact_sheet(
    samples: list[dict], output: Path, variant_label: str, file_prefix: str
) -> None:
    fig = plt.figure(figsize=(23, 10), constrained_layout=True)
    for index, sample in enumerate(samples, start=1):
        ax = fig.add_subplot(2, 5, index, projection="3d")
        cloud = sample["cloud"]
        xyz, colors = cloud[:, :3], np.clip(cloud[:, 3:6], 0, 1)
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=2.2,
            linewidths=0, depthshade=False, rasterized=True,
        )
        configure_3d(ax, cloud_bounds(xyz))
        ax.set_title(
            f"{sample['task']}\nepisode {sample['episode_index']}, {len(xyz):,} points",
            fontsize=10,
        )
    fig.suptitle(
        f"PointACT RLBench {variant_label}：10 个任务各取第一条 episode 的第一帧\n"
        "每个子图单独紧致缩放；front 单相机、工作空间裁剪、1 cm voxelization",
        fontsize=16,
    )
    fig.savefig(output/f"{file_prefix}_dataset_one_frame_per_task.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant-label", default="原始训练点云")
    parser.add_argument("--file-prefix", default="original")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.dataset)
    records = [
        render_individual(sample, args.output, args.variant_label, args.file_prefix)
        for sample in samples
    ]
    render_contact_sheet(samples, args.output, args.variant_label, args.file_prefix)
    report = {
        "source_dataset": str(args.dataset.resolve()),
        "selection": "first frame of the first episode belonging to each task",
        "variant_label": args.variant_label,
        "interpretation": (
            "stored front-camera observation after workspace crop and 1 cm voxelization; "
            "not a complete 360-degree object surface"
        ),
        "samples": records,
    }
    (args.output/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
