#!/usr/bin/env python3
"""Rebuild the ten-task polar dataset with RLBench geometry and corrected optics.

The RGB/depth/state/action trajectory is preserved.  Point clouds are rebuilt
from archived CoppeliaSim depth and per-frame calibration.  Polarization is
re-rendered from the archived scene after removing the CoppeliaSim wireframe
``workspace`` helper; the real ``diningTable_visible`` meshes remain present.
The builder is episode-resumable and never modifies the source dataset.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT.parent
RLBENCH_ROOT = PROJECT_ROOT / "rlbench_custom_render/RLBench"
for path in (SCRIPT_DIR, RLBENCH_ROOT, RLBENCH_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The offline renderer does not need PyRep or CoppeliaSim.  Register the package
# path directly so importing its standalone CUDA modules does not execute
# rlbench/__init__.py, whose public API intentionally imports the live simulator.
if "rlbench" not in sys.modules:
    package = types.ModuleType("rlbench")
    package.__path__ = [str(RLBENCH_ROOT / "rlbench")]
    sys.modules["rlbench"] = package

from create_rlbench_10task_realistic_failure_dataset import apply_corruption  # noqa: E402
from polar_depth_fill import corruption_hole_pixels, fill_depth_multiscale  # noqa: E402
from episode_snapshot_archive import EpisodeSnapshotArchive  # noqa: E402
from rlbench.native_config import NativePolarizationConfig  # noqa: E402
from rlbench.native_renderer import NativePolarizationRenderer  # noqa: E402


msgpack_numpy.patch()

TASKS = (
    "close_box", "close_laptop_lid", "toilet_seat_down", "sweep_to_dustpan",
    "close_fridge", "phone_on_base", "take_umbrella_out_of_umbrella_stand",
    "take_frame_off_hanger", "stack_wine", "water_plants",
)
SOURCE = REPO_ROOT / (
    "robot_data/rlbench/lerobot_point_lmdb/"
    "hybridvla_10tasks_train_keysteps_polar_incomplete9_v1"
)
RAW = RLBENCH_ROOT / "output/ten_tasks_polar_train_20260921"
OUTPUT = REPO_ROOT / (
    "robot_data/rlbench/lerobot_point_lmdb/"
    "hybridvla_10tasks_train_keysteps_polar_rlbench9_v2"
)
WORKSPACE_LOW = np.array([-0.5, -1.0, 0.7505], dtype=np.float32)
WORKSPACE_HIGH = np.array([1.5, 1.0, 2.0], dtype=np.float32)
LMDB_NAMES = {
    "clean": "points_frontview_polar_clean9",
    "incomplete": "points_frontview_polar_incomplete9",
    "filled": "points_frontview_polar_filled9",
    "pixel": "point_pixel_indices",
    "source_pixel": "point_source_pixel_indices",
    "dense": "polar_frontview_dense",
}


def hardlink_tree(source: Path, destination: Path) -> None:
    def link_or_copy(src: str, dst: str) -> str:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return dst
    shutil.copytree(source, destination, copy_function=link_or_copy)


def camera_pointcloud(depth: np.ndarray, camera: dict) -> np.ndarray:
    """Match PyRep VisionSensor.pointcloud_from_depth_and_camera_params exactly."""
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32))
    pixel_depth = np.stack((u * depth, v * depth, depth), axis=-1)
    to_world = np.asarray(camera["to_world"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    rotation = to_world[:3, :3]
    center = to_world[:3, 3:4]
    world_to_camera = np.concatenate((rotation.T, -rotation.T @ center), axis=1)
    projection = intrinsics @ world_to_camera
    inverse = np.linalg.inv(np.vstack((projection, [0., 0., 0., 1.])))[:3]
    homogeneous = np.concatenate(
        (pixel_depth, np.ones((height, width, 1), dtype=np.float32)), axis=-1)
    return (homogeneous.reshape(-1, 4) @ inverse.T).reshape(height, width, 3).astype(np.float32)


def project_world(xyz: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    to_world = np.asarray(camera["to_world"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    camera_xyz = (np.asarray(xyz, dtype=np.float64) - to_world[:3, 3]) @ to_world[:3, :3]
    depth = camera_xyz[:, 2]
    image = camera_xyz @ intrinsics.T
    uv = image[:, :2] / np.where(np.abs(image[:, 2:3]) > 1e-12,
                                 image[:, 2:3], np.nan)
    return uv, depth


def polar_maps(render: dict) -> dict[str, np.ndarray]:
    dolp = np.asarray(render["DoLP"], dtype=np.float32)
    angle = np.asarray(render["AoLP"], dtype=np.float32)
    valid = (np.asarray(render["valid_mask"], dtype=bool) & np.isfinite(dolp)
             & (dolp >= 0.) & (dolp <= 1.))
    angle_valid = (np.asarray(render["AoLP_valid_mask"], dtype=bool)
                   & valid & np.isfinite(angle))
    return {
        "DoLP": np.where(valid, dolp, 0.).astype(np.float32),
        "AoLP": np.where(angle_valid, angle, 0.).astype(np.float32),
        "valid_mask": valid,
        "AoLP_valid_mask": angle_valid,
    }


def encode_dense(maps: dict[str, np.ndarray]) -> bytes:
    angle = maps["AoLP"]
    angle_valid = maps["AoLP_valid_mask"]
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer, DoLP=maps["DoLP"],
        cos2AoLP=np.where(angle_valid, np.cos(2 * angle), 0.).astype(np.float32),
        sin2AoLP=np.where(angle_valid, np.sin(2 * angle), 0.).astype(np.float32),
        valid_mask=maps["valid_mask"], AoLP_valid_mask=angle_valid,
    )
    return buffer.getvalue()


def clean_cloud(depth: np.ndarray, rgb: np.ndarray, maps: dict[str, np.ndarray],
                camera: dict, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    xyz = camera_pointcloud(depth, camera).reshape(-1, 3)
    pixels = np.arange(len(xyz), dtype=np.int32)
    depth_flat = np.asarray(depth).reshape(-1)
    valid = (np.isfinite(depth_flat) & (depth_flat > 0.01)
             & np.isfinite(xyz).all(axis=1) & maps["valid_mask"].reshape(-1))
    valid &= np.all((xyz >= WORKSPACE_LOW) & (xyz <= WORKSPACE_HIGH), axis=1)
    xyz, pixels = xyz[valid], pixels[valid]
    rgb_values = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)[pixels] / 255.
    dolp = maps["DoLP"].reshape(-1)[pixels]
    angle = maps["AoLP"].reshape(-1)[pixels]
    angle_valid = maps["AoLP_valid_mask"].reshape(-1)[pixels]
    cloud = np.column_stack((
        xyz, rgb_values, dolp,
        np.where(angle_valid, np.cos(2 * angle), 0.),
        np.where(angle_valid, np.sin(2 * angle), 0.),
    )).astype(np.float32)
    voxel = np.floor(cloud[:, :3] / voxel_size).astype(np.int32)
    _, representatives = np.unique(voxel, axis=0, return_index=True)
    representatives.sort()
    return np.ascontiguousarray(cloud[representatives]), pixels[representatives]


def sample_modalities(xyz: np.ndarray, rgb: np.ndarray, maps: dict[str, np.ndarray],
                      camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = maps["DoLP"].shape
    uv, depth = project_world(xyz, camera)
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth)
    pixel_xy = np.floor(np.where(finite[:, None], uv, -1.) + 1e-4).astype(np.int64)
    inside = (finite & (depth > .01) & (pixel_xy[:, 0] >= 0)
              & (pixel_xy[:, 0] < width) & (pixel_xy[:, 1] >= 0)
              & (pixel_xy[:, 1] < height))
    pixel = np.full(len(xyz), -1, dtype=np.int32)
    pixel[inside] = (pixel_xy[inside, 1] * width + pixel_xy[inside, 0]).astype(np.int32)
    valid = inside.copy()
    valid[inside] &= maps["valid_mask"].reshape(-1)[pixel[inside]]
    color = np.zeros((len(xyz), 3), dtype=np.float32)
    polar = np.zeros((len(xyz), 3), dtype=np.float32)
    selected = pixel[valid]
    color[valid] = np.asarray(rgb).reshape(-1, 3)[selected] / 255.
    angle = maps["AoLP"].reshape(-1)[selected]
    angle_valid = maps["AoLP_valid_mask"].reshape(-1)[selected]
    polar[valid, 0] = maps["DoLP"].reshape(-1)[selected]
    polar[valid, 1] = np.where(angle_valid, np.cos(2 * angle), 0.)
    polar[valid, 2] = np.where(angle_valid, np.sin(2 * angle), 0.)
    valid &= np.isfinite(color).all(axis=1) & np.isfinite(polar).all(axis=1)
    return color, polar, pixel, valid


def filled_cloud(cloud: np.ndarray, pixels: np.ndarray, holes: np.ndarray,
                 rgb: np.ndarray, maps: dict[str, np.ndarray], camera: dict
                 ) -> tuple[np.ndarray, int]:
    height, width = maps["DoLP"].shape
    _, camera_depth = project_world(cloud[:, :3], camera)
    sparse = np.full(height * width, np.inf, dtype=np.float32)
    good = np.isfinite(camera_depth) & (camera_depth > .01)
    np.minimum.at(sparse, pixels[good], camera_depth[good])
    sparse[~np.isfinite(sparse)] = 0.
    estimated = fill_depth_multiscale(sparse.reshape(height, width)).reshape(-1)
    holes = np.unique(np.asarray(holes, dtype=np.int32))
    new_pixels = holes[(sparse[holes] <= .01) & (estimated[holes] > .01)]
    if not len(new_pixels):
        return cloud.copy(), 0
    z = estimated[new_pixels]
    u, v = new_pixels % width, new_pixels // width
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    camera_xyz = np.column_stack((
        (u - intrinsics[0, 2]) * z / intrinsics[0, 0],
        (v - intrinsics[1, 2]) * z / intrinsics[1, 1], z,
    ))
    to_world = np.asarray(camera["to_world"], dtype=np.float64)
    xyz = camera_xyz @ to_world[:3, :3].T + to_world[:3, 3]
    colors = np.asarray(rgb).reshape(-1, 3)[new_pixels].astype(np.float32) / 255.
    dolp = maps["DoLP"].reshape(-1)[new_pixels]
    angle = maps["AoLP"].reshape(-1)[new_pixels]
    angle_valid = maps["AoLP_valid_mask"].reshape(-1)[new_pixels]
    rows = np.column_stack((
        xyz, colors, dolp,
        np.where(angle_valid, np.cos(2 * angle), 0.),
        np.where(angle_valid, np.sin(2 * angle), 0.),
    )).astype(np.float32)
    return np.ascontiguousarray(np.concatenate((cloud, rows))), len(rows)


def frame_result(raw_episode: Path, frame_index: int, global_episode: int,
                 task_index: int, renderer: NativePolarizationRenderer,
                 voxel_size: float, corruption_seed: int) -> tuple[dict, dict]:
    archive = EpisodeSnapshotArchive(raw_episode / "snapshots")
    meshes, lights, cameras, _ = archive.load_frame(frame_index)
    workspace = [mesh for mesh in meshes if mesh["name"] == "workspace"]
    if len(workspace) != 1:
        raise ValueError(f"{raw_episode}: expected one workspace helper, got {len(workspace)}")
    meshes = [mesh for mesh in meshes if mesh["name"] != "workspace"]
    if not any(mesh["name"].startswith("diningTable_visible") for mesh in meshes):
        raise ValueError(f"{raw_episode}: true dining table is missing")
    summary = json.loads((raw_episode / "frames_spp512/render_summary.json").read_text())
    seed = int(summary["seed"] + frame_index) % 2**32
    render = renderer.render(meshes, lights, cameras["front"], seed=seed, geometry=False)
    maps = polar_maps(render)
    with np.load(raw_episode / "frames" / f"{frame_index:06d}.npz") as source:
        rgb = np.asarray(source["rgb"], dtype=np.uint8)
        depth = np.asarray(source["coppelia_depth_m"], dtype=np.float32)
    clean, clean_pixels = clean_cloud(depth, rgb, maps, cameras["front"], voxel_size)
    indexed = np.column_stack((clean[:, :6], np.arange(len(clean), dtype=np.float32)))
    corrupted = apply_corruption(indexed, task_index, global_episode, corruption_seed)
    source_rows = corrupted.cloud[:, 6].astype(np.int64)
    colors, polar, pixels, valid = sample_modalities(
        corrupted.cloud[:, :3], rgb, maps, cameras["front"])
    incomplete = np.ascontiguousarray(np.column_stack((
        corrupted.cloud[valid, :3], colors[valid], polar[valid])), dtype=np.float32)
    source_pixels = clean_pixels[source_rows[valid]].astype(np.int32)
    current_pixels = pixels[valid].astype(np.int32)
    holes = corruption_hole_pixels(clean_pixels, source_pixels)
    filled, added = filled_cloud(
        incomplete, current_pixels, holes, rgb, maps, cameras["front"])
    if not all(len(item) and np.isfinite(item).all() for item in (clean, incomplete, filled)):
        raise ValueError(f"invalid cloud at {raw_episode}, frame {frame_index}")
    # A clean RLBench depth sample must project back to its source integer pixel.
    check_uv, _ = project_world(clean[:, :3], cameras["front"])
    check_pixels = (np.floor(check_uv + 1e-4).astype(np.int64)[:, 1] * rgb.shape[1]
                    + np.floor(check_uv + 1e-4).astype(np.int64)[:, 0])
    mismatches = int(np.count_nonzero(check_pixels != clean_pixels))
    if mismatches:
        raise ValueError(f"RLBench depth round-trip failed for {mismatches} points")
    stats = dict(corrupted.stats)
    stats.update(
        polar_invalid_removed=int((~valid).sum()), final_output_points=len(incomplete),
        changed_projected_pixel=int(np.count_nonzero(current_pixels != source_pixels)),
        depth_fill_target_pixels=len(holes), depth_filled_points=added,
        filled_output_points=len(filled), clean_points=len(clean),
        polar_valid_pixels=int(maps["valid_mask"].sum()),
        aolp_valid_pixels=int(maps["AoLP_valid_mask"].sum()),
        excluded_workspace_meshes=1,
    )
    payload = {
        "clean": msgpack.packb(clean), "incomplete": msgpack.packb(incomplete),
        "filled": msgpack.packb(filled), "pixel": msgpack.packb(current_pixels),
        "source_pixel": msgpack.packb(source_pixels), "dense": encode_dense(maps),
    }
    return payload, stats


def episode_specs(raw: Path, episodes_per_task: int) -> list[tuple[int, int, str, Path]]:
    return [(task_index * episodes_per_task + episode_index, task_index, task,
             raw / task / f"episode_{episode_index:06d}")
            for task_index, task in enumerate(TASKS)
            for episode_index in range(episodes_per_task)]


def completed_episodes(stats_dir: Path) -> set[int]:
    complete = set()
    for path in stats_dir.glob("episode_*.json"):
        record = json.loads(path.read_text())
        if record.get("complete"):
            complete.add(int(record["episode_index"]))
    return complete


def prepare_staging(source: Path, staging: Path) -> None:
    if staging.exists():
        return
    staging.mkdir(parents=True)
    hardlink_tree(source / "data", staging / "data")
    hardlink_tree(source / "videos", staging / "videos")
    if (source / "images").exists():
        hardlink_tree(source / "images", staging / "images")
    shutil.copytree(source / "meta", staging / "meta")
    shutil.copy2(source / "material_profiles_10tasks.json",
                 staging / "material_profiles_10tasks.json")
    (staging / "repair_episode_stats").mkdir()


def open_envs(staging: Path) -> dict[str, lmdb.Environment]:
    sizes = {"dense": 8, "clean": 4, "incomplete": 4, "filled": 4,
             "pixel": 1, "source_pixel": 1}
    return {key: lmdb.open(str(staging / dirname), map_size=sizes[key] * 1024**3,
                           subdir=True, sync=True)
            for key, dirname in LMDB_NAMES.items()}


def validate_counts(staging: Path, expected: int) -> dict[str, int]:
    counts = {}
    for key, dirname in LMDB_NAMES.items():
        env = lmdb.open(str(staging / dirname), readonly=True, lock=False,
                        readahead=False, max_readers=2)
        try:
            counts[key] = int(env.stat()["entries"])
        finally:
            env.close()
        if counts[key] != expected:
            raise ValueError(f"{dirname}: expected {expected} entries, got {counts[key]}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--spp", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--voxel-size", type=float, default=.012)
    parser.add_argument("--corruption-seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace completed dataset: {args.output}")
    staging = args.output.with_name(args.output.name + ".building")
    prepare_staging(args.source, staging)
    specs = episode_specs(args.raw, args.episodes_per_task)
    if args.limit_episodes is not None:
        specs = specs[:args.limit_episodes]
    info = json.loads((args.source / "meta/info.json").read_text())
    if args.limit_episodes is None and len(specs) != int(info["total_episodes"]):
        raise ValueError("source episode count does not match requested raw episodes")
    done = completed_episodes(staging / "repair_episode_stats")
    envs = open_envs(staging)
    started = time.time()
    current_task = None
    renderer = None
    try:
        for position, (global_episode, task_index, task, raw_episode) in enumerate(specs, 1):
            if global_episode in done:
                print(f"skip {position}/{len(specs)} episode {global_episode}: already complete", flush=True)
                continue
            if not raw_episode.is_dir():
                raise FileNotFoundError(raw_episode)
            if task != current_task:
                if renderer is not None:
                    renderer.clear()
                materials = json.loads((raw_episode / "materials.json").read_text())
                renderer = NativePolarizationRenderer(NativePolarizationConfig(
                    spp=args.spp, max_depth=args.max_depth, device=args.device,
                    lighting="reference", geometry_source="rlbench",
                    material_overrides=materials,
                ))
                current_task = task
            frame_files = sorted((raw_episode / "frames").glob("[0-9]" * 6 + ".npz"))
            if not frame_files:
                raise ValueError(f"no frames in {raw_episode}")
            transactions = {key: env.begin(write=True) for key, env in envs.items()}
            frame_stats = []
            try:
                for frame_index in range(len(frame_files)):
                    payload, stats = frame_result(
                        raw_episode, frame_index, global_episode, task_index, renderer,
                        args.voxel_size, args.corruption_seed)
                    key = f"{global_episode}-{frame_index}".encode("ascii")
                    for name, value in payload.items():
                        transactions[name].put(key, value)
                    frame_stats.append({"episode_index": global_episode,
                                        "frame_index": frame_index, "task": task,
                                        "stats": stats})
                for transaction in transactions.values():
                    transaction.commit()
            except Exception:
                for transaction in transactions.values():
                    try:
                        transaction.abort()
                    except lmdb.Error:
                        pass
                raise
            record = {"complete": True, "episode_index": global_episode,
                      "task": task, "frame_count": len(frame_stats),
                      "frames": frame_stats}
            path = staging / "repair_episode_stats" / f"episode_{global_episode:06d}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, separators=(",", ":")) + "\n")
            temporary.replace(path)
            elapsed = time.time() - started
            print(f"rebuilt {position}/{len(specs)} episode {global_episode} {task}: "
                  f"{len(frame_stats)} frames; {elapsed:.1f}s", flush=True)
    finally:
        if renderer is not None:
            renderer.clear()
        for env in envs.values():
            env.close()
        gc.collect()

    if args.limit_episodes is not None:
        print(json.dumps({"staging": str(staging), "test_episodes": len(specs)}, indent=2))
        return
    records = []
    for path in sorted((staging / "repair_episode_stats").glob("episode_*.json")):
        records.extend(json.loads(path.read_text())["frames"])
    expected_frames = int(info["total_frames"])
    if len(records) != expected_frames:
        raise ValueError(f"expected {expected_frames} frame records, got {len(records)}")
    counts = validate_counts(staging, expected_frames)
    (staging / "frame_corruption_stats.jsonl").write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))
    metadata = {
        "complete": True, "source_dataset": args.source.name,
        "source_raw_dirname": args.raw.name, "total_frames": expected_frames,
        "total_episodes": len(specs), "tasks": list(TASKS),
        "channels": ["x", "y", "z", "r", "g", "b", "DoLP", "cos2AoLP", "sin2AoLP"],
        "point_geometry": "RLBench CoppeliaSim depth unprojected with archived per-frame calibration",
        "polar_geometry": "archived native scene with wireframe workspace helper excluded",
        "real_table_retained": "diningTable_visible",
        "voxel_size_m": args.voxel_size, "spp": args.spp,
        "max_depth": args.max_depth, "corruption_seed": args.corruption_seed,
        "lmdb_counts": counts,
        "material_profiles_sha256": hashlib.sha256(
            (staging / "material_profiles_10tasks.json").read_bytes()).hexdigest(),
    }
    (staging / "meta/polar_rlbench9_repair.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (staging / "README.md").write_text(
        "# Fixed RLBench ten-task polar dataset\n\n"
        "This is a rebuilt copy of `hybridvla_10tasks_train_keysteps_polar_incomplete9_v1`. "
        "RGB videos, states, actions, tasks, and episode ordering are unchanged.\n\n"
        "Point XYZ is reconstructed from the archived CoppeliaSim/RLBench depth map with the "
        "saved camera intrinsics and extrinsics. Polarization is re-rendered at 512 spp after "
        "excluding the non-optical wireframe `workspace` helper. The visible physical table "
        "meshes (`diningTable_visible`) remain in the render and therefore still affect both "
        "visibility and polarization. Each point row is "
        "`[x,y,z,r,g,b,DoLP,cos(2AoLP),sin(2AoLP)]`; an undefined AoLP is encoded by zero in "
        "both angle channels without dropping otherwise valid geometry.\n\n"
        "The clean, corrupted, and morphology-filled inputs are in "
        "`points_frontview_polar_clean9`, `points_frontview_polar_incomplete9`, and "
        "`points_frontview_polar_filled9`. See `meta/polar_rlbench9_repair.json`.\n")
    stats_dir = staging / "robot_state_action_stats"
    stats_dir.mkdir(exist_ok=True)
    for cloud_dir, filename in (
        ("points_frontview_polar_incomplete9", "euler_points_frontview_clf.json"),
        ("points_frontview_polar_filled9", "euler_points_frontview_filled_clf.json"),
    ):
        subprocess.run([
            sys.executable, str(REPO_ROOT / "data_prep/prepare_robot_state_action_stats.py"),
            "--dataset_dirs", str(staging), "--output_file", str(stats_dir / filename),
            "--point_cloud_dir", cloud_dir, "--state_xyz_slice", "0", "3",
            "--action_xyz_slice", "0", "3", "--state_rotation_slice", "3", "7",
            "--action_rotation_slice", "3", "7", "--rotation_type", "quat",
            "--target_rotation_type", "euler", "--replace_zero_std",
            "--classification_action_raw",
        ], cwd=REPO_ROOT, check=True)
    staging.rename(args.output)
    print(json.dumps({"complete": True, "output": str(args.output),
                      "episodes": len(specs), "frames": expected_frames,
                      "lmdb_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
