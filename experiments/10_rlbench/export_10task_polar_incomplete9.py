"""Export ten aligned RLBench tasks as LeRobot + clean/incomplete polar LMDBs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from collect_10task_polar_episodes import MATERIALS, RLBENCH, TASKS
from create_phone_polar_incomplete_episode import (
    sample_projected_modalities,
    source_points,
)
from create_rlbench_10task_realistic_failure_dataset import apply_corruption
from create_stack_wine_10episode_failure_dataset import project_world


msgpack_numpy.patch()
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / (
    "robot_data/rlbench/lerobot_point_lmdb/"
    "hybridvla_10tasks_train_keysteps_polar_incomplete9_v1"
)
FEATURES = {
    "observation.images.front_image": {
        "dtype": "video", "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32", "shape": (8,),
        "names": {"motors": ["x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "gripper"]},
    },
    "action": {
        "dtype": "float32", "shape": (8,),
        "names": {"motors": ["x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "gripper"]},
    },
}


def dense_polar_bytes(frame: np.lib.npyio.NpzFile) -> bytes:
    angle = np.asarray(frame["AoLP"], dtype=np.float32)
    valid = np.asarray(frame["valid_mask"], dtype=bool) & np.asarray(
        frame["AoLP_valid_mask"], dtype=bool)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        DoLP=np.asarray(frame["DoLP"], dtype=np.float32),
        cos2AoLP=np.where(valid, np.cos(2 * angle), 0.0).astype(np.float32),
        sin2AoLP=np.where(valid, np.sin(2 * angle), 0.0).astype(np.float32),
        valid_mask=valid,
    )
    return buffer.getvalue()


def make_clouds(frame_path: Path, task_index: int, episode_index: int,
                seed: int, voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    clean, source_pixels = source_points(frame_path, voxel_size)
    uv, depth = project_world(clean[:, :3])
    with np.load(frame_path) as frame:
        height, width = frame["DoLP"].shape
    projected = np.floor(uv).astype(np.int64)
    if not np.all(depth > 0) or not np.array_equal(
        source_pixels, projected[:, 1] * width + projected[:, 0]
    ):
        raise ValueError(f"Clean point/polar pixel mismatch: {frame_path}")

    indexed = np.column_stack((clean[:, :6], np.arange(len(clean), dtype=np.float32)))
    result = apply_corruption(indexed, task_index, episode_index, seed)
    source_rows = result.cloud[:, 6].astype(np.int64)
    if np.any(source_rows < 0) or np.any(source_rows >= len(clean)):
        raise ValueError(f"Lost point provenance in {frame_path}")
    rgb, polar, final_pixels, valid = sample_projected_modalities(
        frame_path, result.cloud[:, :3])
    incomplete = np.ascontiguousarray(
        np.column_stack((result.cloud[valid, :3], rgb[valid], polar[valid])),
        dtype=np.float32,
    )
    if not len(incomplete) or not np.isfinite(incomplete).all():
        raise ValueError(f"Invalid incomplete point cloud: {frame_path}")
    final_pixels = final_pixels[valid]
    original_pixels = source_pixels[source_rows[valid]].astype(np.int32)
    stats = dict(result.stats)
    stats.update(
        polar_invalid_removed=int((~valid).sum()),
        final_output_points=len(incomplete),
        changed_projected_pixel=int(np.count_nonzero(final_pixels != original_pixels)),
    )
    return clean, incomplete, final_pixels, original_pixels, stats


def depth_alignment_report(expected: list[tuple[str, int, Path]]) -> dict:
    """Summarize same-state Coppelia/native depth checks saved by the renderer."""
    groups: dict[str, list[dict]] = {"all": []}
    for task, _index, raw_episode in expected:
        render = json.loads((raw_episode / "frames_spp512/render_summary.json").read_text())
        records = [frame.get("alignment", {}) for frame in render["frames"]]
        records = [record for record in records
                   if record.get("depth_within_2cm_fraction") is not None]
        groups.setdefault(task, []).extend(records)
        groups["all"].extend(records)

    report = {}
    for name, records in groups.items():
        if not records:
            report[name] = {"checked_frames": 0}
            continue
        report[name] = {
            "checked_frames": len(records),
            "median_frame_depth_abs_median_m": float(np.median(
                [record["depth_abs_median_m"] for record in records])),
            "max_frame_depth_abs_p95_m": float(max(
                record["depth_abs_p95_m"] for record in records)),
            "min_frame_depth_within_2cm_fraction": float(min(
                record["depth_within_2cm_fraction"] for record in records)),
        }
    return report


def build_episode(dataset: LeRobotDataset, raw_episode: Path, task: str,
                  global_episode: int, task_index: int, seed: int,
                  voxel_size: float, transactions: dict[str, lmdb.Transaction],
                  records_stream) -> int:
    summary = json.loads((raw_episode / "summary.json").read_text())
    render = json.loads((raw_episode / "frames_spp512/render_summary.json").read_text())
    if summary.get("complete") is not True or summary.get("task") != task:
        raise ValueError(f"Bad RLBench demo: {raw_episode}")
    if render.get("complete") is not True or render.get("spp") != 512:
        raise ValueError(f"Missing high-quality polar render: {raw_episode}")
    files = sorted((raw_episode / "frames_spp512").glob("[0-9][0-9][0-9][0-9][0-9][0-9].npz"))
    with np.load(raw_episode / "training_episode.npz") as training:
        states = np.asarray(training["state"], dtype=np.float32)
        actions = np.asarray(training["action"], dtype=np.float32)
    if len(files) != len(states) or len(files) != len(actions) or len(files) != render["frame_count"]:
        raise ValueError(f"Frame/action count mismatch: {raw_episode}")
    task_text = "<br>".join(summary["descriptions"])
    staged = []
    for frame_index, frame_path in enumerate(files):
        clean, incomplete, pixels, source_pixels, stats = make_clouds(
            frame_path, task_index, global_episode, seed, voxel_size)
        with np.load(frame_path) as frame:
            rgb = np.asarray(frame["rgb"], dtype=np.uint8)
            dense = dense_polar_bytes(frame)
        if rgb.shape != (256, 256, 3):
            raise ValueError(f"Unexpected RGB size in {frame_path}: {rgb.shape}")
        staged.append((rgb, clean, incomplete, pixels, source_pixels, dense, stats))

    for frame_index, (rgb, *_rest) in enumerate(staged):
        dataset.add_frame({
            "observation.images.front_image": rgb,
            "observation.state": states[frame_index],
            "action": actions[frame_index],
        }, task=task_text)
    dataset.save_episode()

    for frame_index, (_rgb, clean, incomplete, pixels, source_pixels, dense, stats) in enumerate(staged):
        key = f"{global_episode}-{frame_index}".encode("ascii")
        transactions["clean"].put(key, msgpack.packb(clean))
        transactions["incomplete"].put(key, msgpack.packb(incomplete))
        transactions["pixel"].put(key, msgpack.packb(pixels))
        transactions["source_pixel"].put(key, msgpack.packb(source_pixels))
        transactions["dense"].put(key, dense)
        records_stream.write(json.dumps({
            "episode_index": global_episode, "frame_index": frame_index,
            "task": task, "seed": summary["seed"], "stats": stats,
        }) + "\n")
    records_stream.flush()
    return len(staged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RLBENCH / "output/ten_tasks_polar_train_20260921")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--voxel-size", type=float, default=0.012)
    args = parser.parse_args()
    if args.episodes_per_task < 1 or args.voxel_size <= 0:
        parser.error("episode count and voxel size must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    staging = args.output.with_name(args.output.name + ".building")
    if staging.exists():
        raise FileExistsError(staging)

    expected = [(task, index, args.raw / task / f"episode_{index:06d}")
                for task in args.tasks for index in range(args.episodes_per_task)]
    missing = [str(path) for _task, _index, path in expected
               if not (path / "summary.json").is_file()
               or not (path / "frames_spp512/render_summary.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} rendered episodes; first: {missing[0]}")

    staging.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=args.output.name, root=staging, fps=20, robot_type="franka",
        features=FEATURES, image_writer_processes=0, image_writer_threads=4,
    )
    envs = {
        name: lmdb.open(str(staging / path), map_size=8 * 1024**3)
        for name, path in {
            "clean": "points_frontview_polar_clean9",
            "incomplete": "points_frontview_polar_incomplete9",
            "pixel": "point_pixel_indices",
            "source_pixel": "point_source_pixel_indices",
            "dense": "polar_frontview_dense",
        }.items()
    }
    total_frames = 0
    try:
        with (staging / "frame_corruption_stats.jsonl").open("w") as stream:
            for global_episode, (task, _index, raw_episode) in enumerate(expected):
                transactions = {name: env.begin(write=True) for name, env in envs.items()}
                try:
                    frames = build_episode(
                        dataset, raw_episode, task, global_episode, TASKS.index(task),
                        args.seed, args.voxel_size, transactions, stream)
                    for txn in transactions.values():
                        txn.commit()
                except Exception:
                    for txn in transactions.values():
                        try:
                            txn.abort()
                        except lmdb.Error:
                            pass
                    raise
                total_frames += frames
                print(f"exported {global_episode + 1}/{len(expected)}: {task}, "
                      f"{frames} frames", flush=True)
    finally:
        for env in envs.values():
            env.close()
        del dataset
        gc.collect()

    metadata = {
        "complete": True,
        "tasks": list(args.tasks), "episodes_per_task": args.episodes_per_task,
        "total_episodes": len(expected), "total_frames": total_frames,
        "features": ["x", "y", "z", "r", "g", "b", "DoLP", "cos2AoLP", "sin2AoLP"],
        "point_cloud_dirname": "points_frontview_polar_incomplete9",
        "clean_point_cloud_dirname": "points_frontview_polar_clean9",
        "dense_polar_dirname": "polar_frontview_dense",
        "pixel_semantics": "RGB and polar sampled at each corrupted point's current projected pixel",
        "polar_dense_contents": ["DoLP", "cos2AoLP", "sin2AoLP", "valid_mask"],
        "source_raw_dirname": args.raw.name, "voxel_size_m": args.voxel_size,
        "corruption_seed": args.seed,
        "material_profiles_sha256": hashlib.sha256(MATERIALS.read_bytes()).hexdigest(),
    }
    (staging / "meta/polar_incomplete_features.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (staging / "meta/depth_alignment_qa.json").write_text(
        json.dumps(depth_alignment_report(expected), indent=2) + "\n")
    shutil.copy2(MATERIALS, staging / "material_profiles_10tasks.json")
    (staging / "README.md").write_text(
        "# RLBench ten-task polar + incomplete-point-cloud dataset\n\n"
        f"This dataset has {len(expected)} independently generated successful RLBench "
        f"demonstrations across {len(args.tasks)} tasks, with {total_frames} selected keyframes. "
        "It does not share trajectories or frame indices with the earlier "
        "`hybridvla_10tasks_train_keysteps` dataset. Each action targets the next "
        "keyframe absolute end-effector pose (xyz, xyzw quaternion) and gripper state; "
        "the final action repeats the final state.\n\n"
        "The training input is `points_frontview_polar_incomplete9`, an LMDB keyed "
        "by `episode_index-frame_index`. Each float32 row is "
        "`[x,y,z,r,g,b,DoLP,cos(2AoLP),sin(2AoLP)]`. The first three coordinates "
        "include the configured simulated point-cloud failures. RGB and polar are "
        "sampled from the current projected image pixel after corruption. Points "
        "with invalid projected polar values are dropped. The intact reference "
        "cloud is in `points_frontview_polar_clean9`. `polar_frontview_dense` holds "
        "compressed NPZ maps (`DoLP`, `cos2AoLP`, `sin2AoLP`, `valid_mask`) for "
        "every selected keyframe, independent of point-cloud corruption. Polar maps "
        "cover camera-visible valid pixels, not occluded surfaces.\n\n"
        "The RGB video, state, and action are stored in LeRobot v2.1 format. "
        "`point_pixel_indices` and `point_source_pixel_indices` record current "
        "and original pixel provenance for each incomplete point. "
        "`frame_corruption_stats.jsonl` records per-frame failure statistics. "
        "`material_profiles_10tasks.json` contains the assumed optical materials; "
        "these are realistic starting values, not measured properties of the "
        "RLBench assets. `meta/depth_alignment_qa.json` summarizes same-state "
        "Coppelia/native depth agreement for the rendered keyframes. See "
        "`meta/polar_incomplete_features.json` for schema.\n",
        encoding="utf-8",
    )
    norm_path = staging / "robot_state_action_stats/euler_points_frontview_clf.json"
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(REPO_ROOT / "data_prep/prepare_robot_state_action_stats.py"),
        "--dataset_dirs", str(staging), "--output_file", str(norm_path),
        "--point_cloud_dir", "points_frontview_polar_incomplete9",
        "--state_xyz_slice", "0", "3", "--action_xyz_slice", "0", "3",
        "--state_rotation_slice", "3", "7", "--action_rotation_slice", "3", "7",
        "--rotation_type", "quat", "--target_rotation_type", "euler",
        "--replace_zero_std",
    ], cwd=REPO_ROOT, check=True)
    staging.rename(args.output)
    print(json.dumps({"output": str(args.output), "episodes": len(expected),
                      "frames": total_frames}, indent=2))


if __name__ == "__main__":
    main()
