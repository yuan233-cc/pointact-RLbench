"""Create a 10-episode stack_wine dataset with realistic point-cloud failures.

The source LeRobot dataset stores voxelized, unordered xyzrgb points and no
instance masks.  This builder therefore uses conservative task-specific proxy
masks derived from color, connected 3-D components, and workspace geometry.
All random-looking parameters are fixed per episode so the artifacts remain
temporally coherent across the six key-step frames.

Only point clouds are corrupted.  RGB videos, robot state, actions, language,
and timestamps are preserved.  Episodes 800--809 are renumbered to 0--9 so the
result is a standalone one-task LeRobot v2.1 dataset.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import av
import lmdb
import matplotlib
import msgpack
import msgpack_numpy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


msgpack_numpy.patch()

SOURCE_EPISODES = tuple(range(800, 810))
VIDEO_KEYS = (
    "observation.images.front_image",
    "observation.images.left_shoulder_image",
    "observation.images.right_shoulder_image",
    "observation.images.wrist_image",
)

# Fixed RLBench front-camera calibration.  EXTRINSICS maps camera coordinates
# to world coordinates; points are projected with (world - t) @ R.
CAMERA_FOCAL = 351.6771217
CAMERA_CENTER = np.asarray([128.0, 128.0], dtype=np.float32)
CAMERA_EXTRINSICS = np.asarray(
    [
        [-1.1920929e-7, 0.42261794, -0.90630794, 1.3499992],
        [1.0, 5.9604645e-7, 1.4901161e-7, 3.7154656e-8],
        [5.6624413e-7, -0.90630794, -0.42261791, 1.5799993],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

FAILURE_CODES = {
    0: "unchanged",
    1: "rack apparent shape distortion",
    2: "bottle transparent-like wrong depth",
    3: "weak bottle/background boundary bridge",
    4: "sparse floating points",
}
REMOVAL_CODES = {5: "robot connected holes", 6: "bottle transparent-like dropout"}


@dataclass(frozen=True)
class EpisodeParams:
    robot_centers: np.ndarray
    robot_radii: np.ndarray
    bottle_centers: np.ndarray
    bottle_radii: np.ndarray
    rack_phase: float
    rack_amplitude: float
    refraction_sign: float
    floating_centers: np.ndarray
    floating_jitter: np.ndarray


@dataclass
class CorruptionResult:
    cloud: np.ndarray
    codes: np.ndarray
    removed_points: np.ndarray
    removed_codes: np.ndarray
    stats: dict


def deterministic_seed(base_seed: int, source_episode: int) -> int:
    token = f"stack-wine-{source_episode}".encode("ascii")
    digest = hashlib.blake2b(token, digest_size=8, person=b"pcfailv1").digest()
    return (base_seed + int.from_bytes(digest, "little")) % (2**63 - 1)


def make_episode_params(base_seed: int, source_episode: int) -> EpisodeParams:
    rng = np.random.default_rng(deterministic_seed(base_seed, source_episode))
    robot_centers = rng.uniform(0.20, 0.80, size=(2, 3)).astype(np.float32)
    robot_radii = rng.uniform(0.22, 0.38, size=(2, 3)).astype(np.float32)
    bottle_centers = rng.uniform(0.18, 0.82, size=(2, 3)).astype(np.float32)
    bottle_radii = rng.uniform(0.30, 0.52, size=(2, 3)).astype(np.float32)
    floating_centers = np.column_stack(
        (
            rng.uniform(-0.05, 0.55, 3),
            rng.uniform(-0.28, 0.24, 3),
            rng.uniform(0.91, 1.24, 3),
        )
    ).astype(np.float32)
    return EpisodeParams(
        robot_centers=robot_centers,
        robot_radii=robot_radii,
        bottle_centers=bottle_centers,
        bottle_radii=bottle_radii,
        rack_phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        rack_amplitude=float(rng.uniform(0.010, 0.016)),
        refraction_sign=float(rng.choice((-1.0, 1.0))),
        floating_centers=floating_centers,
        floating_jitter=rng.normal(0.0, [0.020, 0.020, 0.025], (24, 3)).astype(np.float32),
    )


def connected_components(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Return radius-connected components as local integer index arrays."""
    if not len(points):
        return []
    neighbors = cKDTree(points).query_ball_point(points, radius)
    unseen = set(range(len(points)))
    components: list[np.ndarray] = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        component = [root]
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components


def detect_bottle(cloud: np.ndarray) -> np.ndarray:
    """Track the black bottle as the largest plausible dark 3-D component."""
    xyz, rgb = cloud[:, :3], cloud[:, 3:6]
    workspace = (
        (xyz[:, 0] > -0.05)
        & (xyz[:, 0] < 0.75)
        & (xyz[:, 1] > -0.58)
        & (xyz[:, 1] < 0.50)
        & (xyz[:, 2] > 0.74)
        & (xyz[:, 2] < 1.25)
    )
    dark = workspace & (rgb.mean(axis=1) < 0.24) & (rgb.max(axis=1) < 0.58)
    dark_indices = np.flatnonzero(dark)
    components = connected_components(xyz[dark_indices], radius=0.025)
    plausible: list[tuple[float, np.ndarray]] = []
    for component in components:
        if len(component) < 20:
            continue
        points = xyz[dark_indices[component]]
        extent = points.max(axis=0) - points.min(axis=0)
        longest = float(extent.max())
        center = points.mean(axis=0)
        if not (0.12 < longest < 0.32 and 0.78 < center[2] < 0.98):
            continue
        # Bottle components have one long axis and two narrow axes.  Favor size,
        # then penalize components centered near the robot base.
        sorted_extent = np.sort(extent)
        slenderness = longest / max(float(sorted_extent[1]), 1e-4)
        score = len(component) + 25.0 * slenderness + 40.0 * max(center[0], 0.0)
        plausible.append((score, dark_indices[component]))
    if not plausible:
        raise RuntimeError("Could not locate the stack_wine bottle component")
    core = max(plausible, key=lambda item: item[0])[1]

    # Recover slightly brighter bottle surface voxels around the dark core.
    distances, _ = cKDTree(xyz[core]).query(xyz, k=1)
    shell = (
        workspace
        & (distances <= 0.012)
        & (rgb.mean(axis=1) < 0.43)
        & (rgb.max(axis=1) < 0.72)
    )
    bottle = np.zeros(len(cloud), dtype=bool)
    bottle[core] = True
    bottle |= shell
    return bottle


def detect_rack(cloud: np.ndarray, bottle: np.ndarray) -> np.ndarray:
    xyz, rgb = cloud[:, :3], cloud[:, 3:6]
    warm = (rgb[:, 0] > 0.34) & (rgb[:, 1] > 0.30) & (rgb[:, 2] > 0.22)
    rack = (
        (xyz[:, 0] > 0.10)
        & (xyz[:, 0] < 0.62)
        & (xyz[:, 1] > 0.00)
        & (xyz[:, 1] < 0.38)
        & (xyz[:, 2] > 0.762)
        & (xyz[:, 2] < 0.93)
        & warm
        & ~bottle
    )
    if rack.sum() < 80:
        raise RuntimeError(f"Rack proxy mask is unexpectedly small: {rack.sum()}")
    return rack


def detect_robot(cloud: np.ndarray, bottle: np.ndarray, rack: np.ndarray) -> np.ndarray:
    xyz, rgb = cloud[:, :3], cloud[:, 3:6]
    chroma = rgb.max(axis=1) - rgb.min(axis=1)
    gray_or_dark = (chroma < 0.13) | (rgb.max(axis=1) < 0.24)
    robot_region = (
        (xyz[:, 0] > -0.48)
        & (xyz[:, 0] < 0.38)
        & (xyz[:, 1] > -0.52)
        & (xyz[:, 1] < 0.32)
        & (xyz[:, 2] > 0.76)
        & (xyz[:, 2] < 1.55)
        & ((xyz[:, 0] < 0.10) | (xyz[:, 2] > 0.94))
    )
    robot = robot_region & gray_or_dark & ~bottle & ~rack
    if robot.sum() < 120:
        raise RuntimeError(f"Robot proxy mask is unexpectedly small: {robot.sum()}")
    return robot


def detect_table(cloud: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    xyz = cloud[:, :3]
    return (
        (xyz[:, 0] > -0.48)
        & (xyz[:, 0] < 0.60)
        & (xyz[:, 1] > -0.55)
        & (xyz[:, 1] < 0.55)
        & (xyz[:, 2] > 0.748)
        & (xyz[:, 2] < 0.7625)
        & ~excluded
    )


def normalized_coordinates(xyz: np.ndarray) -> np.ndarray:
    lower = np.quantile(xyz, 0.02, axis=0)
    upper = np.quantile(xyz, 0.98, axis=0)
    return np.clip((xyz - lower) / np.maximum(upper - lower, 1e-4), 0.0, 1.0)


def lowest_score_mask(indices: np.ndarray, score: np.ndarray, count: int, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=bool)
    count = min(max(int(count), 0), len(indices))
    if count:
        chosen = indices[np.argpartition(score, count - 1)[:count]]
        result[chosen] = True
    return result


def project_world(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = (points - CAMERA_EXTRINSICS[:3, 3]) @ CAMERA_EXTRINSICS[:3, :3]
    depth = camera[:, 2]
    uv = CAMERA_FOCAL * camera[:, :2] / np.maximum(depth[:, None], 1e-6)
    uv += CAMERA_CENTER
    return uv, depth


def apply_corruption(cloud: np.ndarray, params: EpisodeParams) -> CorruptionResult:
    original = np.asarray(cloud, dtype=np.float32)
    output = original.copy()
    xyz = original[:, :3]
    bottle = detect_bottle(original)
    rack = detect_rack(original, bottle)
    robot = detect_robot(original, bottle, rack)
    table = detect_table(original, bottle | rack | robot)
    codes = np.zeros(len(original), dtype=np.uint8)
    keep = np.ones(len(original), dtype=bool)
    removal_codes = np.zeros(len(original), dtype=np.uint8)

    # 1) Connected robot holes.  Centers/radii are episode-fixed in a robust
    # normalized robot frame, so holes move coherently with the articulated arm.
    robot_indices = np.flatnonzero(robot)
    robot_local = normalized_coordinates(xyz[robot_indices])
    robot_distance = np.square(
        (robot_local[:, None, :] - params.robot_centers[None, :, :])
        / params.robot_radii[None, :, :]
    ).sum(axis=-1).min(axis=1)
    robot_remove = lowest_score_mask(
        robot_indices, robot_distance, round(0.13 * len(robot_indices)), len(original)
    )
    keep[robot_remove] = False
    removal_codes[robot_remove] = 5

    # 2) Transparent-like bottle failure: a coherent surface patch disappears,
    # while the rest of the affected patch receives a viewing-ray range error.
    bottle_indices = np.flatnonzero(bottle)
    bottle_local = normalized_coordinates(xyz[bottle_indices])
    bottle_distance = np.square(
        (bottle_local[:, None, :] - params.bottle_centers[None, :, :])
        / params.bottle_radii[None, :, :]
    ).sum(axis=-1).min(axis=1)
    affected_count = max(1, round(0.55 * len(bottle_indices)))
    affected_order = np.argpartition(bottle_distance, affected_count - 1)[:affected_count]
    affected_order = affected_order[np.argsort(bottle_distance[affected_order])]
    dropout_count = round(0.65 * affected_count)
    bottle_remove_indices = bottle_indices[affected_order[:dropout_count]]
    bottle_shift_indices = bottle_indices[affected_order[dropout_count:]]
    keep[bottle_remove_indices] = False
    removal_codes[bottle_remove_indices] = 6

    camera_origin = CAMERA_EXTRINSICS[:3, 3]
    shift_vectors = output[bottle_shift_indices, :3] - camera_origin
    shift_ranges = np.linalg.norm(shift_vectors, axis=1)
    shift_rays = shift_vectors / np.maximum(shift_ranges[:, None], 1e-6)
    local = bottle_local[affected_order[dropout_count:]]
    range_error = params.refraction_sign * (
        0.012 + 0.018 * (0.5 + 0.5 * np.sin(7.0 * local[:, 0] + 5.0 * local[:, 2]))
    )
    output[bottle_shift_indices, :3] = camera_origin + shift_rays * (
        shift_ranges + range_error
    )[:, None]
    codes[bottle_shift_indices] = 2

    # 3) Coherent apparent rack deformation.  This bends the reconstructed
    # points by roughly 1--2 cm without changing the simulator geometry.
    rack_indices = np.flatnonzero(rack)
    rack_xyz = output[rack_indices, :3]
    rack_height = np.clip((rack_xyz[:, 2] - 0.762) / (0.93 - 0.762), 0.0, 1.0)
    output[rack_indices, 0] += params.rack_amplitude * np.sin(
        np.pi * rack_height + params.rack_phase
    )
    output[rack_indices, 1] += 0.006 * np.sin(
        2.0 * np.pi * rack_height + 0.5 * params.rack_phase
    )
    codes[rack_indices] = 1

    # 4) Weak bottle/background boundary bridge in the image plane.  Only a
    # narrow band of nearby non-object points is pulled by at most 12 mm toward
    # the bottle range, preserving the deliberately reduced severity.
    uv, ranges = project_world(xyz)
    non_object = ~(bottle | rack | robot)
    candidate_indices = np.flatnonzero(non_object & keep)
    bottle_tree = cKDTree(uv[bottle_indices])
    pixel_distance, nearest = bottle_tree.query(uv[candidate_indices], k=1)
    nearest_bottle = bottle_indices[nearest]
    range_gap = np.abs(ranges[candidate_indices] - ranges[nearest_bottle])
    valid_bridge = (pixel_distance >= 1.0) & (pixel_distance <= 6.5) & (range_gap > 0.010)
    bridge_candidates = candidate_indices[valid_bridge]
    bridge_nearest = nearest_bottle[valid_bridge]
    bridge_distance = pixel_distance[valid_bridge]
    bridge_limit = min(24, max(8, round(0.10 * len(bottle_indices))))
    if len(bridge_candidates):
        order = np.argsort(bridge_distance)[:bridge_limit]
        bridge_indices = bridge_candidates[order]
        bridge_targets = bridge_nearest[order]
        bridge_strength = 0.18 * np.square(
            np.clip((6.5 - bridge_distance[order]) / 5.5, 0.0, 1.0)
        )
        desired_delta = bridge_strength * (
            ranges[bridge_targets] - ranges[bridge_indices]
        )
        desired_delta = np.clip(desired_delta, -0.012, 0.012)
        bridge_vectors = xyz[bridge_indices] - camera_origin
        bridge_rays = bridge_vectors / np.maximum(
            np.linalg.norm(bridge_vectors, axis=1, keepdims=True), 1e-6
        )
        output[bridge_indices, :3] = camera_origin + bridge_rays * (
            ranges[bridge_indices] + desired_delta
        )[:, None]
        codes[bridge_indices] = 3
    else:
        bridge_indices = np.empty(0, dtype=np.int64)
        desired_delta = np.empty(0, dtype=np.float32)

    # 5) A sparse set of table returns is relocated into episode-fixed free-air
    # clusters.  Twenty-four points is visible in diagnostics but remains sparse.
    table_indices = np.flatnonzero(table & keep)
    floating_count = min(24, len(table_indices))
    if floating_count:
        table_xyz = xyz[table_indices]
        phase_score = (
            np.sin(31.0 * table_xyz[:, 0] + params.rack_phase)
            + np.cos(29.0 * table_xyz[:, 1] - params.rack_phase)
        )
        selected = np.argpartition(phase_score, floating_count - 1)[:floating_count]
        floating_indices = table_indices[selected]
        cluster_ids = np.arange(floating_count) % len(params.floating_centers)
        output[floating_indices, :3] = (
            params.floating_centers[cluster_ids]
            + params.floating_jitter[:floating_count]
        )
        codes[floating_indices] = 4
    else:
        floating_indices = np.empty(0, dtype=np.int64)

    removed = ~keep
    final_cloud = np.ascontiguousarray(output[keep], dtype=np.float32)
    final_codes = np.ascontiguousarray(codes[keep])
    stats = {
        "input_points": int(len(original)),
        "output_points": int(len(final_cloud)),
        "proxy_masks": {
            "bottle": int(bottle.sum()),
            "rack": int(rack.sum()),
            "robot": int(robot.sum()),
            "table": int(table.sum()),
        },
        "failures": {
            "robot_hole_removed": int(robot_remove.sum()),
            "bottle_transparent_removed": int(len(bottle_remove_indices)),
            "bottle_wrong_depth": int(len(bottle_shift_indices)),
            "rack_shape_distorted": int(len(rack_indices)),
            "boundary_bridge": int(len(bridge_indices)),
            "boundary_mean_abs_shift_m": (
                float(np.abs(desired_delta).mean()) if len(desired_delta) else 0.0
            ),
            "floating_points": int(len(floating_indices)),
        },
    }
    return CorruptionResult(
        cloud=final_cloud,
        codes=final_codes,
        removed_points=np.ascontiguousarray(original[removed, :3]),
        removed_codes=np.ascontiguousarray(removal_codes[removed]),
        stats=stats,
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def replace_parquet_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing parquet column {name!r}")
    return table.set_column(index, name, pa.array(values, type=pa.int64()))


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def subset_tabular_and_metadata(source: Path, output: Path) -> dict:
    data_out = output / "data" / "chunk-000"
    meta_out = output / "meta"
    data_out.mkdir(parents=True)
    meta_out.mkdir(parents=True)

    source_episodes = load_jsonl(source / "meta" / "episodes.jsonl")
    source_stats = load_jsonl(source / "meta" / "episodes_stats.jsonl")
    source_tasks = load_jsonl(source / "meta" / "tasks.jsonl")
    source_info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))

    episodes_out: list[dict] = []
    stats_out: list[dict] = []
    global_index = 0
    total_frames = 0
    for new_episode, source_episode in enumerate(SOURCE_EPISODES):
        source_path = source / "data" / "chunk-000" / f"episode_{source_episode:06d}.parquet"
        table = pq.read_table(source_path)
        length = table.num_rows
        table = replace_parquet_column(
            table, "episode_index", np.full(length, new_episode, dtype=np.int64)
        )
        table = replace_parquet_column(
            table, "task_index", np.zeros(length, dtype=np.int64)
        )
        table = replace_parquet_column(
            table, "index", np.arange(global_index, global_index + length, dtype=np.int64)
        )
        pq.write_table(
            table,
            data_out / f"episode_{new_episode:06d}.parquet",
            compression="zstd",
        )

        episode_record = copy.deepcopy(source_episodes[source_episode])
        episode_record["episode_index"] = new_episode
        episodes_out.append(episode_record)

        stats_record = copy.deepcopy(source_stats[source_episode])
        stats_record["episode_index"] = new_episode
        count = [length]
        stats_record["stats"]["episode_index"] = {
            "min": [new_episode], "max": [new_episode], "mean": [float(new_episode)],
            "std": [0.0], "count": count,
        }
        stats_record["stats"]["task_index"] = {
            "min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": count,
        }
        frame_indices = np.arange(global_index, global_index + length, dtype=np.float64)
        stats_record["stats"]["index"] = {
            "min": [int(frame_indices.min())],
            "max": [int(frame_indices.max())],
            "mean": [float(frame_indices.mean())],
            "std": [float(frame_indices.std())],
            "count": count,
        }
        stats_out.append(stats_record)
        global_index += length
        total_frames += length

    write_jsonl(meta_out / "episodes.jsonl", episodes_out)
    write_jsonl(meta_out / "episodes_stats.jsonl", stats_out)
    task = copy.deepcopy(source_tasks[8])
    task["task_index"] = 0
    write_jsonl(meta_out / "tasks.jsonl", [task])

    info = copy.deepcopy(source_info)
    info.update(
        {
            "total_episodes": len(SOURCE_EPISODES),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": len(SOURCE_EPISODES) * len(VIDEO_KEYS),
            "total_chunks": 1,
            "splits": {"train": f"0:{len(SOURCE_EPISODES)}"},
        }
    )
    (meta_out / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    for filename in ("state_action_norm_rot6d_2d.json", "state_action_norm_rot6d_3d_frontview.json"):
        shutil.copy2(source / "meta" / filename, meta_out / filename)
    shutil.copytree(source / "robot_state_action_stats", output / "robot_state_action_stats")
    return {"total_frames": total_frames, "task": task["task"]}


def subset_videos(source: Path, output: Path) -> dict:
    modes: dict[str, int] = {}
    for video_key in VIDEO_KEYS:
        for new_episode, source_episode in enumerate(SOURCE_EPISODES):
            source_video = (
                source / "videos" / "chunk-000" / video_key
                / f"episode_{source_episode:06d}.mp4"
            )
            destination = (
                output / "videos" / "chunk-000" / video_key
                / f"episode_{new_episode:06d}.mp4"
            )
            mode = hardlink_or_copy(source_video, destination)
            modes[mode] = modes.get(mode, 0) + 1
    return modes


def load_cloud(transaction: lmdb.Transaction, episode: int, frame: int) -> np.ndarray:
    key = f"{episode}-{frame}".encode("ascii")
    value = transaction.get(key)
    if value is None:
        raise KeyError(key)
    cloud = np.asarray(msgpack.unpackb(value), dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 6:
        raise ValueError(f"{key!r}: expected Nx6 cloud, got {cloud.shape}")
    return cloud


def create_point_lmdb(
    source: Path, output: Path, base_seed: int, frame_lengths: list[int]
) -> tuple[list[dict], dict[int, list[tuple[np.ndarray, CorruptionResult]]]]:
    source_env = lmdb.open(
        str(source / "points_frontview"), readonly=True, lock=False,
        readahead=False, max_readers=2,
    )
    output_env = lmdb.open(
        str(output / "points_frontview"), map_size=512 * 1024**2, subdir=True
    )
    frame_stats: list[dict] = []
    diagnostics: dict[int, list[tuple[np.ndarray, CorruptionResult]]] = {}
    try:
        with source_env.begin(buffers=False) as source_txn:
            output_txn = output_env.begin(write=True)
            try:
                for new_episode, source_episode in enumerate(SOURCE_EPISODES):
                    params = make_episode_params(base_seed, source_episode)
                    episode_diagnostics = []
                    for frame in range(frame_lengths[new_episode]):
                        clean = load_cloud(source_txn, source_episode, frame)
                        result = apply_corruption(clean, params)
                        key = f"{new_episode}-{frame}".encode("ascii")
                        output_txn.put(key, msgpack.packb(result.cloud))
                        record = {
                            "episode_index": new_episode,
                            "source_episode_index": source_episode,
                            "frame_index": frame,
                            **result.stats,
                        }
                        frame_stats.append(record)
                        episode_diagnostics.append((clean, result))
                    diagnostics[new_episode] = episode_diagnostics
                output_txn.commit()
                output_txn = None
            finally:
                if output_txn is not None:
                    output_txn.abort()
        output_env.sync()
    finally:
        source_env.close()
        output_env.close()
    return frame_stats, diagnostics


def decode_video(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def scatter_projected(ax, cloud: np.ndarray, codes: np.ndarray | None = None) -> None:
    uv, depth = project_world(cloud[:, :3])
    valid = (
        np.isfinite(uv).all(axis=1)
        & (depth > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < 256.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < 256.0)
    )
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(depth[indices])[::-1]]
    ax.set_facecolor((0.035, 0.035, 0.045))
    ax.scatter(
        uv[indices, 0], uv[indices, 1], c=np.clip(cloud[indices, 3:6], 0.0, 1.0),
        s=2.2, linewidths=0,
    )
    if codes is not None:
        palette = {1: "#ff8c42", 2: "#00d4ff", 3: "#ffe66d", 4: "#ff4fb3"}
        for code, color in palette.items():
            selected = indices[codes[indices] == code]
            if len(selected):
                ax.scatter(
                    uv[selected, 0], uv[selected, 1], s=8.0,
                    facecolors="none", edgecolors=color, linewidths=0.45,
                )
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def render_diagnostics(
    output: Path, diagnostics: dict[int, list[tuple[np.ndarray, CorruptionResult]]]
) -> None:
    diagnostic_dir = output / "diagnostics"
    diagnostic_dir.mkdir()
    for episode, frames in diagnostics.items():
        video_path = (
            output / "videos" / "chunk-000" / "observation.images.front_image"
            / f"episode_{episode:06d}.mp4"
        )
        rgb_frames = decode_video(video_path)
        if len(rgb_frames) != len(frames):
            raise RuntimeError(
                f"Episode {episode}: {len(rgb_frames)} RGB frames but {len(frames)} clouds"
            )
        fig, axes = plt.subplots(3, len(frames), figsize=(3.05 * len(frames), 9.0))
        for frame, ((clean, result), rgb) in enumerate(zip(frames, rgb_frames)):
            axes[0, frame].imshow(rgb)
            axes[0, frame].axis("off")
            axes[0, frame].set_title(f"frame {frame}")
            scatter_projected(axes[1, frame], clean)
            scatter_projected(axes[2, frame], result.cloud, result.codes)
            if len(result.removed_points):
                removed_uv, removed_depth = project_world(result.removed_points)
                visible = (
                    (removed_depth > 0.0)
                    & (removed_uv[:, 0] >= 0.0) & (removed_uv[:, 0] < 256.0)
                    & (removed_uv[:, 1] >= 0.0) & (removed_uv[:, 1] < 256.0)
                )
                axes[2, frame].scatter(
                    removed_uv[visible, 0], removed_uv[visible, 1],
                    s=5.0, marker="x", c="#ff3b30", linewidths=0.45,
                )
        axes[0, 0].set_ylabel("RGB", fontsize=11)
        axes[1, 0].set_ylabel("clean cloud\nRGB view", fontsize=11)
        axes[2, 0].set_ylabel("corrupted cloud\nRGB view", fontsize=11)
        fig.suptitle(
            f"stack_wine episode {episode:02d} — RGB-aligned point-cloud corruption\n"
            "orange rack | cyan bottle depth | yellow boundary | magenta floating | red x removed",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(
            diagnostic_dir / f"episode_{episode:06d}_rgb_aligned.png",
            dpi=160, bbox_inches="tight",
        )
        plt.close(fig)


def validate_dataset(output: Path, expected_frames: int) -> dict:
    failures: list[str] = []
    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    if info["total_episodes"] != 10 or info["total_frames"] != expected_frames:
        failures.append("info.json episode/frame totals are incorrect")

    point_env = lmdb.open(str(output / "points_frontview"), readonly=True, lock=False)
    checked_points = 0
    point_count_min = None
    point_count_max = 0
    try:
        with point_env.begin(buffers=False) as txn:
            if txn.stat()["entries"] != expected_frames:
                failures.append(
                    f"point LMDB has {txn.stat()['entries']} entries, expected {expected_frames}"
                )
            for episode in range(10):
                table_path = output / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
                table = pq.read_table(table_path)
                if not all(value == episode for value in table["episode_index"].to_pylist()):
                    failures.append(f"episode {episode}: parquet episode_index mismatch")
                if not all(value == 0 for value in table["task_index"].to_pylist()):
                    failures.append(f"episode {episode}: parquet task_index mismatch")
                for frame in range(table.num_rows):
                    cloud = load_cloud(txn, episode, frame)
                    if not np.isfinite(cloud).all():
                        failures.append(f"{episode}-{frame}: non-finite point cloud")
                    if (cloud[:, 3:6] < 0.0).any() or (cloud[:, 3:6] > 1.0).any():
                        failures.append(f"{episode}-{frame}: RGB outside [0, 1]")
                    checked_points += len(cloud)
                    point_count_min = len(cloud) if point_count_min is None else min(point_count_min, len(cloud))
                    point_count_max = max(point_count_max, len(cloud))
    finally:
        point_env.close()

    checked_videos = 0
    for video_key in VIDEO_KEYS:
        for episode in range(10):
            path = output / "videos" / "chunk-000" / video_key / f"episode_{episode:06d}.mp4"
            if not path.is_file():
                failures.append(f"missing video {path.relative_to(output)}")
            checked_videos += 1

    diagnostic_count = len(list((output / "diagnostics").glob("*.png")))
    if diagnostic_count != 10:
        failures.append(f"found {diagnostic_count} diagnostics, expected 10")
    result = {
        "passed": not failures,
        "checked_frames": expected_frames,
        "checked_point_values": checked_points,
        "output_point_count_range": [point_count_min, point_count_max],
        "checked_videos": checked_videos,
        "rgb_aligned_diagnostics": diagnostic_count,
        "failures": failures[:30],
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Dataset validation failed: {failures[:3]}")
    return result


def summarize_corruption(frame_stats: list[dict]) -> dict:
    keys = [
        key for key in next(iter(frame_stats))["failures"]
        if key != "boundary_mean_abs_shift_m"
    ]
    totals = {
        key: sum(record["failures"][key] for record in frame_stats)
        for key in keys
    }
    boundary_points = totals["boundary_bridge"]
    totals["boundary_mean_abs_shift_m"] = (
        sum(
            record["failures"]["boundary_mean_abs_shift_m"]
            * record["failures"]["boundary_bridge"]
            for record in frame_stats
        )
        / boundary_points
        if boundary_points
        else 0.0
    )
    totals["input_points"] = sum(record["input_points"] for record in frame_stats)
    totals["output_points"] = sum(record["output_points"] for record in frame_stats)
    return totals


def write_documentation(
    source: Path,
    output: Path,
    seed: int,
    tabular: dict,
    video_modes: dict,
    frame_stats: list[dict],
) -> dict:
    write_jsonl(output / "frame_corruption_stats.jsonl", frame_stats)
    summary = summarize_corruption(frame_stats)
    metadata = {
        "dataset_name": output.name,
        "task": "stack_wine",
        "source_dataset": str(source.resolve()),
        "source_episodes": list(SOURCE_EPISODES),
        "output_episodes": list(range(10)),
        "episode_mapping": {
            str(new): source_episode
            for new, source_episode in enumerate(SOURCE_EPISODES)
        },
        "total_episodes": 10,
        "total_frames": tabular["total_frames"],
        "base_seed": seed,
        "temporal_consistency": (
            "Artifact centers, deformation phase, refraction direction, and floating "
            "clusters are fixed per episode; proxy masks follow the bottle and robot."
        ),
        "failure_models": {
            "robot": "13% connected local holes in an episode-consistent normalized robot frame",
            "bottle": "transparent-like connected dropout plus 12--30 mm viewing-ray error",
            "bottle_boundary": "weak <=12 mm image-space range bridge into nearby background",
            "rack": "episode-consistent 10--16 mm bend plus 6 mm lateral twist",
            "background": "24 table returns relocated into three sparse free-air clusters",
        },
        "front_camera": {
            "resolution": [256, 256],
            "focal_pixels": CAMERA_FOCAL,
            "principal_point": CAMERA_CENTER.tolist(),
            "extrinsics_camera_to_world": CAMERA_EXTRINSICS.tolist(),
        },
        "unchanged": [
            "four RGB video streams", "robot state", "actions", "timestamps", "language"
        ],
        "normalization": (
            "Normalization JSON files are copied from the original 10-task dataset so this "
            "pilot remains compatible with the existing PointACT training/evaluation setup."
        ),
        "video_storage": video_modes,
        "aggregate_corruption": summary,
        "limitations": [
            "The source LMDB is unordered and contains no simulator instance masks or depth grid.",
            "Bottle/rack/robot/table assignment therefore uses conservative stack_wine-specific proxy masks.",
            "The RGB videos remain clean; corruptions are applied only to points_frontview.",
        ],
    }
    (output / "corruption_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# stack_wine: 10 episodes with realistic point-cloud failures\n\n"
        "This is a directly loadable one-task LeRobot v2.1 dataset derived from source "
        "episodes 800--809 and renumbered to 0--9. It has 60 key-step frames. RGB, state, "
        "actions, language, and timestamps are unchanged; only `points_frontview` is corrupted.\n\n"
        "Each episode combines connected robot holes, transparent-like bottle dropout and "
        "wrong depth, a deliberately weak bottle/background boundary bridge, coherent rack "
        "shape distortion, and 24 sparse floating points per frame. Parameters are fixed for "
        "the episode so artifacts remain temporally coherent.\n\n"
        "See `corruption_metadata.json`, `frame_corruption_stats.jsonl`, `validation.json`, "
        "and the ten RGB-aligned images in `diagnostics/`. Proxy semantic masks are used "
        "because the stored point clouds do not include simulator instance IDs.\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    dataset_parent = root / "robot_data" / "rlbench" / "lerobot_point_lmdb"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=dataset_parent / "hybridvla_10tasks_train_keysteps",
    )
    parser.add_argument(
        "--output", type=Path,
        default=dataset_parent / "hybridvla_stack_wine_10episodes_realistic_failures_v1",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not (source / "points_frontview").is_dir():
        raise FileNotFoundError(source / "points_frontview")

    output.mkdir(parents=True)
    try:
        tabular = subset_tabular_and_metadata(source, output)
        frame_lengths = [
            record["length"] for record in load_jsonl(output / "meta" / "episodes.jsonl")
        ]
        video_modes = subset_videos(source, output)
        frame_stats, diagnostics = create_point_lmdb(
            source, output, args.seed, frame_lengths
        )
        render_diagnostics(output, diagnostics)
        metadata = write_documentation(
            source, output, args.seed, tabular, video_modes, frame_stats
        )
        validation = validate_dataset(output, tabular["total_frames"])
        metadata["validation"] = validation
        (output / "corruption_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    except Exception:
        # Preserve the partial directory for debugging; the output path is never
        # silently reused or overwritten on a later run.
        raise
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
