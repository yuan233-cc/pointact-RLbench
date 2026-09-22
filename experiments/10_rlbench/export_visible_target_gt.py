"""Export training-only visible target point sets for the ten-task filled9 archive.

Each LMDB record contains sampled world XYZ and a target label for every row
of points_frontview_polar_filled9. No target label is needed by inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np


msgpack_numpy.patch()
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps_polar_incomplete9_v1"
RAW = REPO_ROOT.parent / "rlbench_custom_render/RLBench/output/ten_tasks_polar_train_20260921"
TARGET_MESHES = {
    "close_box": ("box_lid",),
    "close_fridge": ("door_top_visual",),
    "close_laptop_lid": ("lid_visual",),
    "phone_on_base": ("phone_visual",),
    "stack_wine": ("wine_bottle_visual",),
    "sweep_to_dustpan": ("sweep_to_dustpan_broom_visual",),
    "take_frame_off_hanger": ("frame_visual", "picture"),
    "take_umbrella_out_of_umbrella_stand": ("umbrella_visual",),
    "toilet_seat_down": ("toilet_seat_up_seat",),
    "water_plants": ("waterer_visual",),
}


def sample_target(points: np.ndarray, limit: int, voxel_size: float, seed: int) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    voxels = np.floor(points / voxel_size).astype(np.int32)
    _, representatives = np.unique(voxels, axis=0, return_index=True)
    points = points[np.sort(representatives)]
    if len(points) > limit:
        indices = np.random.default_rng(seed).choice(len(points), limit, replace=False)
        points = points[indices]
    return np.ascontiguousarray(points, dtype=np.float32)


def target_groups(mesh_spec: tuple[str, ...] | dict) -> list[tuple[str, tuple[str, ...]]]:
    """Resolve a binary target or named interaction groups for balanced GT sampling."""
    if not isinstance(mesh_spec, dict):
        return [("target", tuple(mesh_spec))]
    if "manipulated" not in mesh_spec or "related" not in mesh_spec:
        raise ValueError("Interaction targets require manipulated and related groups")
    groups = []
    for role in ("manipulated", "related"):
        value = mesh_spec[role]
        entries = value.items() if isinstance(value, dict) else [(role, value)]
        for name, mesh_names in entries:
            if not mesh_names or any(not isinstance(mesh, str) or not mesh for mesh in mesh_names):
                raise ValueError(f"Empty or invalid target mesh group: {role}/{name}")
            groups.append((f"{role}/{name}", tuple(mesh_names)))
    if not groups or groups[0][0] != "manipulated/manipulated":
        raise ValueError("The first target group must be the manipulated object")
    names = [mesh for _group, meshes in groups for mesh in meshes]
    if len(names) != len(set(names)):
        raise ValueError("Target mesh names may not belong to multiple groups")
    return groups


def fair_quotas(counts: list[int], budget: int) -> list[int]:
    quotas = [0] * len(counts)
    remaining = budget
    while remaining:
        available = [i for i, count in enumerate(counts) if quotas[i] < count]
        if not available:
            break
        share = max(1, remaining // len(available))
        for i in available:
            take = min(share, counts[i] - quotas[i], remaining)
            quotas[i] += take
            remaining -= take
            if not remaining:
                break
    return quotas


def sample_interaction_groups(group_points: list[np.ndarray], limit: int, seed: int) -> np.ndarray:
    """Reserve half the GT budget for the manipulated object when possible."""
    counts = [len(points) for points in group_points]
    if not counts:
        return np.empty((0, 3), dtype=np.float32)
    manipulated = min(counts[0], limit // 2)
    related = fair_quotas(counts[1:], limit - manipulated)
    manipulated += min(counts[0] - manipulated, limit - manipulated - sum(related))
    quotas = [manipulated, *related]
    sampled = []
    for i, (points, quota) in enumerate(zip(group_points, quotas)):
        if not quota:
            continue
        if len(points) > quota:
            indices = np.random.default_rng(seed + i).choice(len(points), quota, replace=False)
            points = points[indices]
        sampled.append(points)
    return np.ascontiguousarray(np.concatenate(sampled), dtype=np.float32) if sampled else np.empty((0, 3), dtype=np.float32)


def project_current_pixels(points: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    to_world = np.asarray(camera["to_world"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    camera_points = (points - to_world[:3, 3]) @ to_world[:3, :3]
    z = camera_points[:, 2]
    uv = camera_points[:, :2] / np.maximum(z[:, None], 1e-8)
    uv = uv @ intrinsics[:2, :2].T + intrinsics[:2, 2]
    pixels = np.floor(uv).astype(np.int32)
    return pixels, z


def export_record(frame_path: Path, snapshot_path: Path, cloud: np.ndarray,
                  mesh_spec: tuple[str, ...] | dict, limit: int, voxel_size: float,
                  seed: int) -> dict:
    snapshot = json.loads(snapshot_path.read_text())
    groups = target_groups(mesh_spec)
    group_handles = []
    for group_name, mesh_names in groups:
        handles = sorted({int(mesh["handle"]) for mesh in snapshot["meshes"]
                          if mesh["name"].split("/")[0] in mesh_names})
        if not handles:
            raise ValueError(f"No target mesh for {group_name}: {mesh_names} in {snapshot_path}")
        group_handles.append(handles)
    with np.load(frame_path) as frame:
        depth = np.asarray(frame["depth_m"], dtype=np.float32)
        group_masks = [np.isin(frame["object_mask"], handles) for handles in group_handles]
        mask = np.logical_or.reduce(group_masks)
        dense_points = np.asarray(frame["point_cloud"], dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0) & np.isfinite(dense_points).all(axis=-1)
        if isinstance(mesh_spec, dict):
            sampled_groups = [sample_target(dense_points[valid & group_mask],
                                            int((valid & group_mask).sum()), voxel_size, seed + i)
                              for i, group_mask in enumerate(group_masks)]
            target_points = sample_interaction_groups(sampled_groups, limit, seed)
        else:
            target_points = sample_target(dense_points[valid & mask], limit, voxel_size, seed)

        pixels, z = project_current_pixels(cloud[:, :3], snapshot["cameras"]["front"])
        height, width = mask.shape
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
                  (pixels[:, 1] >= 0) & (pixels[:, 1] < height) & (z > 0))
        input_mask = np.zeros(len(cloud), dtype=np.bool_)
        valid_indices = np.flatnonzero(inside)
        py, px = pixels[valid_indices, 1], pixels[valid_indices, 0]
        input_mask[valid_indices] = (
            mask[py, px] & np.isfinite(depth[py, px]) &
            (np.abs(z[valid_indices] - depth[py, px]) < 0.05)
        )
    return {"points": target_points, "input_mask": input_mask}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--task-map", type=Path, help="JSON mapping task names to mesh-name lists")
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--limit-frames", type=int, default=None, help="Smoke validation only")
    args = parser.parse_args()
    if args.max_points < 1 or args.voxel_size <= 0:
        parser.error("max-points and voxel-size must be positive")
    target_meshes = TARGET_MESHES if args.task_map is None else json.loads(args.task_map.read_text())
    source_meta = json.loads((args.dataset / "meta/polar_incomplete_features.json").read_text())
    records = [json.loads(line) for line in (args.dataset / "frame_corruption_stats.jsonl").read_text().splitlines()]
    if args.limit_frames is not None:
        records = records[:args.limit_frames]
    output = args.output or args.dataset / "target_visible_gt"
    if output.exists():
        raise FileExistsError(output)
    stage = output.with_name(output.name + ".building")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    source = lmdb.open(str(args.dataset / "points_frontview_polar_filled9"),
                       readonly=True, lock=False, readahead=False)
    sink = lmdb.open(str(stage), map_size=1024**3)
    tasks = source_meta["tasks"]
    episodes_per_task = source_meta["episodes_per_task"]
    nonempty = 0
    try:
        with source.begin(buffers=True) as read_txn:
            write_txn = sink.begin(write=True)
            for i, record in enumerate(records):
                episode = int(record["episode_index"])
                frame_index = int(record["frame_index"])
                task = tasks[episode // episodes_per_task]
                if record["task"] != task or task not in target_meshes:
                    raise ValueError(f"Missing or mismatched target mapping for {record}")
                key = f"{episode}-{frame_index}".encode("ascii")
                value = read_txn.get(key)
                if value is None:
                    raise KeyError(key)
                cloud = np.asarray(msgpack.unpackb(value), dtype=np.float32)
                raw_episode = args.raw / task / f"episode_{episode % episodes_per_task:06d}"
                label = export_record(
                    raw_episode / "frames_spp512" / f"{frame_index:06d}.npz",
                    raw_episode / "snapshots/frames" / f"{frame_index:06d}.json",
                    cloud, target_meshes[task], args.max_points, args.voxel_size,
                    episode * 10000 + frame_index,
                )
                if len(label["points"]):
                    nonempty += 1
                write_txn.put(key, msgpack.packb(label))
                if (i + 1) % 100 == 0:
                    write_txn.commit()
                    write_txn = sink.begin(write=True)
                    print(f"exported {i + 1}/{len(records)} frames", flush=True)
            write_txn.commit()
        (stage / "manifest.json").write_text(json.dumps({
            "frames": len(records), "nonempty_target_frames": nonempty,
            "complete": len(records) == source_meta["total_frames"],
            "target_meshes": target_meshes,
            "max_points": args.max_points, "voxel_size_m": args.voxel_size,
            "sampling_strategy": (
                "manipulated_half_then_fair_related" if args.task_map is not None and
                isinstance(next(iter(target_meshes.values())), dict) else "uniform_target"
            ),
            "source_raw": str(args.raw),
        }, indent=2) + "\n")
    finally:
        source.close()
        sink.close()
    stage.rename(output)
    print(f"Saved {len(records)} frames, {nonempty} with visible targets: {output}")


if __name__ == "__main__":
    main()
