"""Create the full 10-task RLBench dataset with realistic point-cloud failures.

The source contains 1000 episodes / 5056 key-step frames as unordered voxelized
xyzrgb clouds.  It has no simulator instance IDs, organized depth grid, or
per-point semantic labels.  This builder therefore uses conservative connected
foreground components and task-aware geometric rules.  The random-looking
parameters are fixed per episode, while masks are re-detected per frame.

The original episode/task numbering, RGB videos, robot states, actions,
language, metadata, and normalization are preserved.  Only points_frontview is
new.  Unchanged files are hard-linked when possible, so do not edit linked MP4
or parquet files in place.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import av
import lmdb
import matplotlib
import msgpack
import msgpack_numpy
import numpy as np
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_stack_wine_10episode_failure_dataset import (
    CAMERA_EXTRINSICS,
    CorruptionResult,
    connected_components,
    decode_video,
    detect_bottle,
    detect_table,
    lowest_score_mask,
    make_episode_params,
    normalized_coordinates,
    project_world,
    scatter_projected,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


msgpack_numpy.patch()

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

FAILURE_CODES = {
    0: "unchanged",
    1: "support/object apparent shape distortion",
    2: "task-object transparent-like wrong depth",
    3: "weak object/background boundary bridge",
    4: "sparse floating points",
}
REMOVAL_CODES = {5: "robot connected holes", 6: "task-object transparent-like dropout"}


@dataclass(frozen=True)
class Component:
    indices: np.ndarray
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mean_rgb: np.ndarray

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def extent(self) -> np.ndarray:
        return self.upper - self.lower


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def load_cloud(transaction: lmdb.Transaction, episode: int, frame: int) -> np.ndarray:
    key = f"{episode}-{frame}".encode("ascii")
    value = transaction.get(key)
    if value is None:
        raise KeyError(key)
    cloud = np.asarray(msgpack.unpackb(value), dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 6:
        raise ValueError(f"{key!r}: expected Nx6 xyzrgb, got {cloud.shape}")
    return cloud


def foreground_components(cloud: np.ndarray) -> list[Component]:
    xyz, rgb = cloud[:, :3], cloud[:, 3:6]
    foreground = (
        (xyz[:, 0] > -0.55)
        & (xyz[:, 0] < 1.25)
        & (xyz[:, 1] > -1.0)
        & (xyz[:, 1] < 1.0)
        & (xyz[:, 2] > 0.763)
        & (xyz[:, 2] < 2.0)
    )
    indices = np.flatnonzero(foreground)
    components = []
    for local in connected_components(xyz[indices], radius=0.038):
        selected = indices[local]
        if len(selected) < 12:
            continue
        points = xyz[selected]
        components.append(
            Component(
                indices=selected,
                center=points.mean(axis=0),
                lower=points.min(axis=0),
                upper=points.max(axis=0),
                mean_rgb=rgb[selected].mean(axis=0),
            )
        )
    return components


def is_robot_component(component: Component) -> bool:
    chroma = float(component.mean_rgb.max() - component.mean_rgb.min())
    gray = chroma < 0.11
    anchored_left = component.lower[0] < -0.055 and component.upper[2] > 1.18
    base_region = (
        component.lower[0] < -0.08
        and component.upper[0] > -0.34
        and component.lower[1] < 0.13
        and component.upper[1] > -0.14
        and component.lower[2] < 0.82
    )
    return gray and (anchored_left or base_region)


def scene_masks(cloud: np.ndarray, task_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return robot, task-object, and support-object proxy masks."""
    xyz = cloud[:, :3]
    components = foreground_components(cloud)
    robot_components = [component for component in components if is_robot_component(component)]
    object_components = [component for component in components if not is_robot_component(component)]

    robot = np.zeros(len(cloud), dtype=bool)
    for component in robot_components:
        robot[component.indices] = True

    # Picture frames occasionally touch the gripper and merge with the gray
    # robot component.  Recover only their colored surface point-wise.  A
    # generic x-axis split is intentionally avoided because it would mislabel
    # a robot arm reaching into positive world x as a deformable task object.
    if task_index == 7 and sum(item.size for item in object_components) < 100:
        rgb = cloud[:, 3:6]
        chroma = rgb.max(axis=1) - rgb.min(axis=1)
        colored_indices = np.flatnonzero((xyz[:, 2] > 0.76) & (chroma > 0.12))
        if len(colored_indices) >= 50:
            robot[colored_indices] = False
            colored_points = xyz[colored_indices]
            object_components.append(
                Component(
                    indices=colored_indices,
                    center=colored_points.mean(axis=0),
                    lower=colored_points.min(axis=0),
                    upper=colored_points.max(axis=0),
                    mean_rgb=cloud[colored_indices, 3:6].mean(axis=0),
                )
            )

    # Ignore tiny disconnected foreground fragments for semantic assignment.
    candidates = [component for component in object_components if component.size >= 25]
    if not candidates:
        candidates = object_components
    if not candidates:
        # Last-resort positive-x colored/dark foreground.  Appearance gating
        # avoids treating a white robot arm as a deformable task object when
        # contact merges all visible geometry into one component.
        rgb = cloud[:, 3:6]
        chroma = rgb.max(axis=1) - rgb.min(axis=1)
        fallback = (
            (xyz[:, 2] > 0.765)
            & (xyz[:, 0] > 0.015)
            & ((chroma > 0.055) | (rgb.max(axis=1) < 0.32))
        )
        robot[fallback] = False
        indices = np.flatnonzero(fallback)
        if len(indices) < 20:
            fallback = (
                (xyz[:, 2] > 0.765)
                & ((chroma > 0.055) | (rgb.max(axis=1) < 0.32))
            )
            robot[fallback] = False
            indices = np.flatnonzero(fallback)
        if len(indices) < 20:
            fallback = (xyz[:, 2] > 0.765) & (xyz[:, 0] > 0.015)
            robot[fallback] = False
            indices = np.flatnonzero(fallback)
        if len(indices) < 20:
            fallback = (xyz[:, 2] > 0.765)
            robot[fallback] = False
            indices = np.flatnonzero(fallback)
        if len(indices) < 20:
            raise RuntimeError("Point cloud contains fewer than 20 foreground points")
        points = xyz[indices]
        candidates = [
            Component(
                indices=indices,
                center=points.mean(axis=0), lower=points.min(axis=0),
                upper=points.max(axis=0), mean_rgb=cloud[indices, 3:6].mean(axis=0),
            )
        ]
        used_fallback = True
    else:
        used_fallback = False

    # stack_wine has a validated dark, slender bottle detector.  Other tasks
    # select the manipulated component using stable task geometry.
    target = np.zeros(len(cloud), dtype=bool)
    selected_component: Component | None = None
    if task_index == 8:
        try:
            target = detect_bottle(cloud)
        except RuntimeError:
            target[:] = False

    if target.sum() < 20:
        if task_index == 2:  # toilet lid: highest object component
            selected_component = max(candidates, key=lambda item: item.center[2])
        elif task_index in (3, 9):  # broom / watering can begin on negative-y side
            selected_component = min(candidates, key=lambda item: item.center[1])
        elif task_index == 5:  # handset: compact, low component separate from orange base
            compact = [
                item for item in candidates
                if item.size >= 45 and item.center[2] < 0.98 and item.extent.max() > 0.07
            ]
            selected_component = min(compact or candidates, key=lambda item: item.size)
        else:
            selected_component = max(candidates, key=lambda item: item.size)
        target[selected_component.indices] = True

    if task_index == 8:
        # A grasped bottle can touch the gripper and enter a robot-connected
        # component.  The validated dark-bottle detector takes precedence.
        robot[target] = False

    object_union = np.zeros(len(cloud), dtype=bool)
    for component in candidates:
        object_union[component.indices] = True

    # Attached objects need an internal target/support split because their lid
    # and base often form one connected component.
    chosen_indices = np.flatnonzero(target)
    if task_index in (0, 1, 4, 6) and len(chosen_indices) >= 40:
        if task_index == 0:  # box lid extends toward negative world y
            values = xyz[chosen_indices, 1]
            keep_target = values <= np.quantile(values, 0.48)
        elif task_index == 1:  # laptop lid is the upper half
            values = xyz[chosen_indices, 2]
            keep_target = values >= np.quantile(values, 0.48)
        elif task_index == 4:  # split the large fridge door into material regions
            values = xyz[chosen_indices, 1]
            keep_target = values <= np.quantile(values, 0.52)
        else:  # umbrella versus lower stand
            values = xyz[chosen_indices, 2]
            keep_target = values >= np.quantile(values, 0.35)
        target[:] = False
        target[chosen_indices[keep_target]] = True

    support = object_union & ~target
    if support.sum() < 25:
        # Split a single object along its largest axis so transparent-like and
        # shape failures still affect distinct surface regions.
        indices = np.flatnonzero(target)
        points = xyz[indices]
        axis = int(np.argmax(np.ptp(points, axis=0)))
        threshold = np.median(points[:, axis])
        move_to_support = indices[points[:, axis] > threshold]
        target[move_to_support] = False
        support[move_to_support] = True

    target &= ~robot
    support &= ~robot & ~target
    return robot, target, support, {
        "foreground_components": len(components),
        "robot_components": len(robot_components),
        "object_components": len(candidates),
        "fallback": used_fallback,
    }


def apply_corruption(
    cloud: np.ndarray,
    task_index: int,
    source_episode: int,
    base_seed: int,
) -> CorruptionResult:
    original = np.asarray(cloud, dtype=np.float32)
    output = original.copy()
    xyz = original[:, :3]
    params = make_episode_params(base_seed, source_episode)
    robot, target, support, segmentation = scene_masks(original, task_index)
    table = detect_table(original, robot | target | support)
    codes = np.zeros(len(original), dtype=np.uint8)
    keep = np.ones(len(original), dtype=bool)
    removal_codes = np.zeros(len(original), dtype=np.uint8)

    robot_indices = np.flatnonzero(robot)
    if len(robot_indices) >= 20:
        local = normalized_coordinates(xyz[robot_indices])
        distance = np.square(
            (local[:, None, :] - params.robot_centers[None, :, :])
            / params.robot_radii[None, :, :]
        ).sum(axis=-1).min(axis=1)
        robot_remove = lowest_score_mask(
            robot_indices, distance, round(0.13 * len(robot_indices)), len(original)
        )
    else:
        robot_remove = np.zeros(len(original), dtype=bool)
    keep[robot_remove] = False
    removal_codes[robot_remove] = 5

    target_indices = np.flatnonzero(target)
    target_local = normalized_coordinates(xyz[target_indices])
    target_distance = np.square(
        (target_local[:, None, :] - params.bottle_centers[None, :, :])
        / params.bottle_radii[None, :, :]
    ).sum(axis=-1).min(axis=1)
    # Cap the patch on large doors/lids: transparent failure should be local,
    # not delete a third of an entire refrigerator or box.
    affected_count = min(max(1, round(0.55 * len(target_indices))), 320)
    affected_order = np.argpartition(target_distance, affected_count - 1)[:affected_count]
    affected_order = affected_order[np.argsort(target_distance[affected_order])]
    dropout_count = round(0.65 * affected_count)
    target_remove_indices = target_indices[affected_order[:dropout_count]]
    target_shift_indices = target_indices[affected_order[dropout_count:]]
    keep[target_remove_indices] = False
    removal_codes[target_remove_indices] = 6

    camera_origin = CAMERA_EXTRINSICS[:3, 3]
    vectors = output[target_shift_indices, :3] - camera_origin
    ranges = np.linalg.norm(vectors, axis=1)
    rays = vectors / np.maximum(ranges[:, None], 1e-6)
    shifted_local = target_local[affected_order[dropout_count:]]
    range_error = params.refraction_sign * (
        0.012 + 0.018 * (0.5 + 0.5 * np.sin(7.0 * shifted_local[:, 0] + 5.0 * shifted_local[:, 2]))
    )
    output[target_shift_indices, :3] = camera_origin + rays * (ranges + range_error)[:, None]
    codes[target_shift_indices] = 2

    support_indices = np.flatnonzero(support)
    support_xyz = output[support_indices, :3]
    support_local = normalized_coordinates(support_xyz)
    output[support_indices, 0] += params.rack_amplitude * np.sin(
        np.pi * support_local[:, 2] + params.rack_phase
    )
    output[support_indices, 1] += 0.006 * np.sin(
        2.0 * np.pi * support_local[:, 2] + 0.5 * params.rack_phase
    )
    codes[support_indices] = 1

    uv, all_ranges = project_world(xyz)
    # Support surfaces are valid boundary backgrounds: mixed depth pixels often
    # bridge a manipulated object into its stand, rack, base, or surrounding lid.
    candidate_indices = np.flatnonzero(~target & ~robot & keep)
    target_tree = cKDTree(uv[target_indices])
    pixel_distance, nearest = target_tree.query(uv[candidate_indices], k=1)
    nearest_target = target_indices[nearest]
    valid = (pixel_distance >= 1.0) & (pixel_distance <= 8.5)
    bridge_candidates = candidate_indices[valid]
    bridge_nearest = nearest_target[valid]
    bridge_distance = pixel_distance[valid]
    bridge_limit = min(24, max(8, round(0.10 * len(target_indices))))
    if len(bridge_candidates):
        order = np.argsort(bridge_distance)[:bridge_limit]
        bridge_indices = bridge_candidates[order]
        bridge_targets = bridge_nearest[order]
        strength = 0.18 * np.square(np.clip((8.5 - bridge_distance[order]) / 7.5, 0.0, 1.0))
        delta = strength * (all_ranges[bridge_targets] - all_ranges[bridge_indices])
        small = np.abs(delta) < 0.0015
        delta[small] = params.refraction_sign * (
            0.0015 + 0.0015 * np.clip(strength[small] / 0.18, 0.0, 1.0)
        )
        delta = np.clip(delta, -0.012, 0.012)
        bridge_vectors = xyz[bridge_indices] - camera_origin
        bridge_rays = bridge_vectors / np.maximum(
            np.linalg.norm(bridge_vectors, axis=1, keepdims=True), 1e-6
        )
        output[bridge_indices, :3] = camera_origin + bridge_rays * (
            all_ranges[bridge_indices] + delta
        )[:, None]
        codes[bridge_indices] = 3
    else:
        bridge_indices = np.empty(0, dtype=np.int64)
        delta = np.empty(0, dtype=np.float32)

    table_indices = np.flatnonzero(table & keep)
    if len(table_indices) < 24:
        # Close-fridge views may contain no table at all.  Relocate a few visible
        # support returns instead so every task still receives sparse flying points.
        table_indices = np.flatnonzero(support & keep)
    floating_count = min(24, len(table_indices))
    if floating_count:
        table_xyz = xyz[table_indices]
        score = (
            np.sin(31.0 * table_xyz[:, 0] + params.rack_phase)
            + np.cos(29.0 * table_xyz[:, 1] - params.rack_phase)
        )
        selected = np.argpartition(score, floating_count - 1)[:floating_count]
        floating_indices = table_indices[selected]
        cluster_ids = np.arange(floating_count) % len(params.floating_centers)
        output[floating_indices, :3] = (
            params.floating_centers[cluster_ids] + params.floating_jitter[:floating_count]
        )
        codes[floating_indices] = 4
    else:
        floating_indices = np.empty(0, dtype=np.int64)

    removed = ~keep
    stats = {
        "input_points": int(len(original)),
        "output_points": int(keep.sum()),
        "segmentation": segmentation,
        "proxy_masks": {
            "robot": int(robot.sum()),
            "target_object": int(target.sum()),
            "support_object": int(support.sum()),
            "table": int(table.sum()),
        },
        "failures": {
            "robot_hole_removed": int(robot_remove.sum()),
            "target_transparent_removed": int(len(target_remove_indices)),
            "target_wrong_depth": int(len(target_shift_indices)),
            "support_shape_distorted": int(len(support_indices)),
            "boundary_bridge": int(len(bridge_indices)),
            "boundary_mean_abs_shift_m": float(np.abs(delta).mean()) if len(delta) else 0.0,
            "floating_points": int(len(floating_indices)),
        },
    }
    return CorruptionResult(
        cloud=np.ascontiguousarray(output[keep], dtype=np.float32),
        codes=np.ascontiguousarray(codes[keep]),
        removed_points=np.ascontiguousarray(original[removed, :3]),
        removed_codes=np.ascontiguousarray(removal_codes[removed]),
        stats=stats,
    )


def hardlink_tree(source: Path, destination: Path) -> dict:
    modes = {"hardlink": 0, "copy": 0}

    def link_or_copy(src: str, dst: str) -> str:
        try:
            os.link(src, dst)
            modes["hardlink"] += 1
        except OSError:
            shutil.copy2(src, dst)
            modes["copy"] += 1
        return dst

    shutil.copytree(source, destination, copy_function=link_or_copy)
    return {key: value for key, value in modes.items() if value}


def link_unchanged_dataset_content(source: Path, output: Path) -> dict:
    totals: dict[str, int] = {}
    for name in ("data", "meta", "videos", "robot_state_action_stats"):
        modes = hardlink_tree(source / name, output / name)
        for mode, count in modes.items():
            totals[mode] = totals.get(mode, 0) + count
    return totals


def task_indices(source: Path) -> tuple[list[dict], list[int]]:
    episodes = load_jsonl(source / "meta" / "episodes.jsonl")
    tasks = load_jsonl(source / "meta" / "tasks.jsonl")
    task_by_text = {task["task"]: int(task["task_index"]) for task in tasks}
    indices = [task_by_text[episode["tasks"][0]] for episode in episodes]
    return episodes, indices


def create_point_lmdb(
    source: Path,
    output: Path,
    episodes: list[dict],
    tasks: list[int],
    base_seed: int,
    commit_every: int,
) -> tuple[list[dict], dict[int, list[tuple[np.ndarray, CorruptionResult]]]]:
    source_points = source / "points_frontview"
    map_size = max(2 * (source_points / "data.mdb").stat().st_size, 1024**3)
    source_env = lmdb.open(
        str(source_points), readonly=True, lock=False, readahead=False, max_readers=2
    )
    output_env = lmdb.open(str(output / "points_frontview"), map_size=map_size, subdir=True)
    frame_stats: list[dict] = []
    diagnostics: dict[int, list[tuple[np.ndarray, CorruptionResult]]] = {}
    total_frames = 0
    total_input = 0
    total_output = 0
    try:
        with source_env.begin(buffers=False) as source_txn:
            output_txn = output_env.begin(write=True)
            try:
                for episode, (episode_meta, task_index) in enumerate(zip(episodes, tasks)):
                    representative = episode == task_index * 100
                    diagnostic_frames = []
                    for frame in range(int(episode_meta["length"])):
                        clean = load_cloud(source_txn, episode, frame)
                        result = apply_corruption(clean, task_index, episode, base_seed)
                        key = f"{episode}-{frame}".encode("ascii")
                        output_txn.put(key, msgpack.packb(result.cloud))
                        frame_stats.append(
                            {
                                "episode_index": episode,
                                "frame_index": frame,
                                "task_index": task_index,
                                "task": TASK_SLUGS[task_index],
                                **result.stats,
                            }
                        )
                        if representative:
                            diagnostic_frames.append((clean, result))
                        total_frames += 1
                        total_input += len(clean)
                        total_output += len(result.cloud)
                        if total_frames % commit_every == 0:
                            output_txn.commit()
                            output_txn = output_env.begin(write=True)
                            print(
                                f"processed {total_frames:,}/{sum(item['length'] for item in episodes):,} "
                                f"frames; kept {total_output:,}/{total_input:,} points",
                                flush=True,
                            )
                    if representative:
                        diagnostics[task_index] = diagnostic_frames
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


def render_diagnostics(
    output: Path,
    diagnostics: dict[int, list[tuple[np.ndarray, CorruptionResult]]],
) -> None:
    diagnostic_dir = output / "diagnostics"
    diagnostic_dir.mkdir()
    for task_index, frames in diagnostics.items():
        episode = task_index * 100
        video_path = (
            output / "videos" / "chunk-000" / "observation.images.front_image"
            / f"episode_{episode:06d}.mp4"
        )
        rgb_frames = decode_video(video_path)
        if len(rgb_frames) != len(frames):
            raise RuntimeError(
                f"Episode {episode}: {len(rgb_frames)} RGB frames but {len(frames)} point frames"
            )
        fig, axes = plt.subplots(3, len(frames), figsize=(3.05 * len(frames), 9.0))
        if len(frames) == 1:
            axes = axes[:, None]
        for frame, ((clean, result), rgb) in enumerate(zip(frames, rgb_frames)):
            axes[0, frame].imshow(rgb)
            axes[0, frame].axis("off")
            axes[0, frame].set_title(f"frame {frame}")
            scatter_projected(axes[1, frame], clean)
            scatter_projected(axes[2, frame], result.cloud, result.codes)
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
            f"{TASK_SLUGS[task_index]} representative episode {episode}\n"
            "orange shape | cyan wrong depth | yellow boundary | magenta floating | red x removed",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(
            diagnostic_dir / f"{task_index + 1:02d}_{TASK_SLUGS[task_index]}_rgb_aligned.png",
            dpi=160, bbox_inches="tight",
        )
        plt.close(fig)


def aggregate_records(records: list[dict]) -> dict:
    failure_keys = [
        key for key in records[0]["failures"] if key != "boundary_mean_abs_shift_m"
    ]
    totals = {
        key: sum(record["failures"][key] for record in records)
        for key in failure_keys
    }
    boundary_points = totals["boundary_bridge"]
    totals["boundary_mean_abs_shift_m"] = (
        sum(
            record["failures"]["boundary_mean_abs_shift_m"]
            * record["failures"]["boundary_bridge"]
            for record in records
        ) / boundary_points
        if boundary_points else 0.0
    )
    totals["input_points"] = sum(record["input_points"] for record in records)
    totals["output_points"] = sum(record["output_points"] for record in records)
    totals["global_removal_rate"] = 1.0 - totals["output_points"] / totals["input_points"]
    totals["segmentation_fallback_frames"] = sum(
        bool(record["segmentation"]["fallback"]) for record in records
    )
    return totals


def write_documentation(
    source: Path,
    output: Path,
    base_seed: int,
    storage: dict,
    frame_stats: list[dict],
) -> dict:
    write_jsonl(output / "frame_corruption_stats.jsonl", frame_stats)
    per_task = {
        TASK_SLUGS[task_index]: aggregate_records(
            [record for record in frame_stats if record["task_index"] == task_index]
        )
        for task_index in TASK_SLUGS
    }
    metadata = {
        "dataset_name": output.name,
        "source_dataset": str(source),
        "total_tasks": 10,
        "total_episodes": 1000,
        "total_frames": len(frame_stats),
        "base_seed": base_seed,
        "failure_codes": {str(key): value for key, value in FAILURE_CODES.items()},
        "removal_codes": {str(key): value for key, value in REMOVAL_CODES.items()},
        "failure_models": {
            "robot": "13% connected local holes in an episode-consistent normalized robot proxy",
            "task_object": "capped connected transparent-like dropout plus 12--30 mm viewing-ray error",
            "support_object": "episode-consistent 10--16 mm bend plus 6 mm lateral twist",
            "boundary": "weak <=12 mm image-space range bridge into nearby background",
            "floating": "up to 24 table returns moved into three episode-fixed free-air clusters",
        },
        "temporal_consistency": (
            "Centers, radii, refraction direction, deformation phase, and floating clusters "
            "are fixed per episode; connected foreground masks are re-detected each frame."
        ),
        "unchanged": [
            "episode/task indices", "four RGB streams", "state", "actions",
            "timestamps", "language", "normalization files",
        ],
        "storage": storage,
        "normalization": (
            "The original 10-task 5056-frame euler/rot6d statistics are intentionally retained "
            "for classifier checkpoint compatibility."
        ),
        "aggregate": aggregate_records(frame_stats),
        "per_task": per_task,
        "limitations": [
            "Proxy masks are used because the source point LMDB has no simulator instance labels.",
            "Temporal consistency is episode-level sensor consistency, not exact rigid-link tracking.",
            "RGB remains clean; only points_frontview is corrupted.",
            "Hard-linked unchanged files share their inode with the source and must not be edited in place.",
        ],
    }
    (output / "corruption_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# RLBench 10-task realistic point-cloud failures v2\n\n"
        "Full derivative of `hybridvla_10tasks_train_keysteps`: 10 tasks, 1000 episodes, "
        "and 5056 key-step frames. Original RGB/state/action/language/metadata and episode "
        "indices are unchanged. Only `points_frontview` is regenerated.\n\n"
        "Each episode combines connected robot holes, task-object transparent-like dropout "
        "and wrong depth, support-object apparent distortion, a weak boundary bridge, and "
        "sparse floating points. Parameters are deterministic and episode-consistent.\n\n"
        "The source clouds contain no instance labels, so task-aware connected-component "
        "proxy masks are used. See `corruption_metadata.json`, per-frame statistics, validation, "
        "and ten RGB-aligned representative images under `diagnostics/`.\n",
        encoding="utf-8",
    )
    return metadata


def validate(source: Path, output: Path, expected_frames: int) -> dict:
    failures: list[str] = []
    source_env = lmdb.open(str(source / "points_frontview"), readonly=True, lock=False)
    output_env = lmdb.open(str(output / "points_frontview"), readonly=True, lock=False)
    checked = 0
    point_min = None
    point_max = 0
    try:
        with source_env.begin(buffers=True) as source_txn, output_env.begin(buffers=True) as output_txn:
            if source_txn.stat()["entries"] != expected_frames:
                failures.append("source LMDB entry count differs from metadata")
            if output_txn.stat()["entries"] != expected_frames:
                failures.append("output LMDB entry count differs from metadata")
            for key, _ in source_txn.cursor():
                value = output_txn.get(key)
                if value is None:
                    failures.append(f"missing point key {bytes(key)!r}")
                    continue
                cloud = np.asarray(msgpack.unpackb(value), dtype=np.float32)
                if cloud.ndim != 2 or cloud.shape[1] != 6 or len(cloud) == 0:
                    failures.append(f"invalid cloud {bytes(key)!r}: {cloud.shape}")
                    continue
                if not np.isfinite(cloud).all():
                    failures.append(f"non-finite cloud {bytes(key)!r}")
                if (cloud[:, 3:6] < 0.0).any() or (cloud[:, 3:6] > 1.0).any():
                    failures.append(f"RGB out of range {bytes(key)!r}")
                point_min = len(cloud) if point_min is None else min(point_min, len(cloud))
                point_max = max(point_max, len(cloud))
                checked += 1
    finally:
        source_env.close()
        output_env.close()

    for name in ("data", "meta", "videos", "robot_state_action_stats"):
        if not (output / name).is_dir():
            failures.append(f"missing {name} directory")
    diagnostics = len(list((output / "diagnostics").glob("*.png")))
    if diagnostics != 10:
        failures.append(f"expected 10 diagnostics, found {diagnostics}")
    result = {
        "passed": not failures,
        "checked_frames": checked,
        "checked_tasks": 10,
        "checked_episodes": 1000,
        "output_point_count_range": [point_min, point_max],
        "rgb_aligned_diagnostics": diagnostics,
        "failures": failures[:30],
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Validation failed: {failures[:3]}")
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parent = root / "robot_data" / "rlbench" / "lerobot_point_lmdb"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=parent / "hybridvla_10tasks_train_keysteps"
    )
    parser.add_argument(
        "--output", type=Path,
        default=parent / "hybridvla_10tasks_train_keysteps_realistic_failures_v2",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--commit-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not (source / "points_frontview").is_dir():
        raise FileNotFoundError(source / "points_frontview")
    episodes, tasks = task_indices(source)
    if len(episodes) != 1000 or set(tasks) != set(range(10)):
        raise ValueError("Expected the original 1000-episode / 10-task dataset")

    output.mkdir(parents=True)
    storage = link_unchanged_dataset_content(source, output)
    frame_stats, diagnostics = create_point_lmdb(
        source, output, episodes, tasks, args.seed, args.commit_every
    )
    render_diagnostics(output, diagnostics)
    metadata = write_documentation(source, output, args.seed, storage, frame_stats)
    validation = validate(source, output, len(frame_stats))
    metadata["validation"] = validation
    (output / "corruption_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
