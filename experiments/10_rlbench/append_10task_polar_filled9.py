"""Append corrected depth-filled 9-channel points to the existing ten-task dataset.

Only a new LMDB and its normalization/manifest files are written. Clean and
incomplete LMDBs, LeRobot actions, videos, and dense polar maps stay untouched.
The offline hole locations come from clean voxel sample provenance; this is not
an inference-time point-completion algorithm.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np

from collect_10task_polar_episodes import RLBENCH
from create_stack_wine_10episode_failure_dataset import (
    CAMERA_CENTER, CAMERA_EXTRINSICS, CAMERA_FOCAL, project_world,
)
from polar_depth_fill import add_polar_depth_filled_points, corruption_hole_pixels


msgpack_numpy.patch()
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / (
    "robot_data/rlbench/lerobot_point_lmdb/"
    "hybridvla_10tasks_train_keysteps_polar_incomplete9_v1"
)
POINT_DIR = "points_frontview_polar_filled9"
NORM_FILE = "euler_points_frontview_filled_clf.json"


def read_array(txn: lmdb.Transaction, key: bytes, name: str) -> np.ndarray:
    packed = txn.get(key)
    if packed is None:
        raise KeyError(f"Missing {key!r} in {name}")
    return np.asarray(msgpack.unpackb(packed))


def clean_pixels_from_xyz(cloud: np.ndarray, width: int, height: int) -> np.ndarray:
    uv, depth = project_world(cloud[:, :3])
    xy = np.floor(uv).astype(np.int64)
    if not np.isfinite(uv).all() or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Clean cloud has invalid camera projections")
    if np.any((xy[:, 0] < 0) | (xy[:, 0] >= width) |
              (xy[:, 1] < 0) | (xy[:, 1] >= height)):
        raise ValueError("Clean cloud projects outside the polar image")
    return (xy[:, 1] * width + xy[:, 0]).astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw", type=Path,
                        default=RLBENCH / "output/ten_tasks_polar_train_20260921")
    args = parser.parse_args()
    root = args.dataset.resolve()
    source_meta = json.loads((root / "meta/polar_incomplete_features.json").read_text())
    if not source_meta.get("complete") or source_meta.get("total_frames") != 5051:
        raise ValueError("Expected the completed 5051-frame ten-task source dataset")
    tasks = source_meta["tasks"]
    episodes_per_task = source_meta["episodes_per_task"]
    frame_records = [json.loads(line) for line in
                     (root / "frame_corruption_stats.jsonl").read_text().splitlines()]
    if len(frame_records) != source_meta["total_frames"]:
        raise ValueError("Frame corruption records do not match source metadata")

    final = root / POINT_DIR
    stage = root / f"{POINT_DIR}.building"
    norm = root / "robot_state_action_stats" / NORM_FILE
    norm_stage = norm.with_name(norm.name + ".building")
    manifest = root / "meta/polar_filled_features.json"
    for path in (final, stage, norm, norm_stage, manifest):
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing output: {path}")

    source_names = {
        "clean": "points_frontview_polar_clean9",
        "incomplete": "points_frontview_polar_incomplete9",
        "pixel": "point_pixel_indices",
        "source_pixel": "point_source_pixel_indices",
    }
    sources = {
        name: lmdb.open(str(root / dirname), readonly=True, lock=False,
                        readahead=False, max_readers=2)
        for name, dirname in source_names.items()
    }
    stage.mkdir()
    output = lmdb.open(str(stage), map_size=4 * 1024**3)
    counts = {"frames": 0, "clean_points": 0, "incomplete_points": 0,
              "target_pixels": 0, "filled_points": 0}
    try:
        with (stage / "frame_fill_stats.jsonl").open("w") as stats_file:
            with (sources["clean"].begin() as clean_txn,
                  sources["incomplete"].begin() as incomplete_txn,
                  sources["pixel"].begin() as pixel_txn,
                  sources["source_pixel"].begin() as source_pixel_txn):
                current_episode = None
                write_txn = None
                for index, record in enumerate(frame_records):
                    episode = int(record["episode_index"])
                    frame_index = int(record["frame_index"])
                    task = tasks[episode // episodes_per_task]
                    if record["task"] != task:
                        raise ValueError(f"Frame {index} has an unexpected task")
                    if episode != current_episode:
                        if write_txn is not None:
                            write_txn.commit()
                        write_txn = output.begin(write=True)
                        current_episode = episode
                    key = f"{episode}-{frame_index}".encode("ascii")
                    clean = np.asarray(read_array(clean_txn, key, "clean"), dtype=np.float32)
                    incomplete = np.asarray(read_array(incomplete_txn, key, "incomplete"),
                                            dtype=np.float32)
                    pixels = np.asarray(read_array(pixel_txn, key, "pixel"), dtype=np.int32)
                    source_pixels = np.asarray(
                        read_array(source_pixel_txn, key, "source_pixel"), dtype=np.int32)
                    if clean.ndim != 2 or incomplete.ndim != 2 or clean.shape[1] != 9 or incomplete.shape[1] != 9:
                        raise ValueError(f"{key!r}: expected clean and incomplete Nx9 clouds")
                    if pixels.shape != (len(incomplete),) or source_pixels.shape != pixels.shape:
                        raise ValueError(f"{key!r}: pixel correspondence is misaligned")
                    raw_episode = args.raw / task / f"episode_{episode % episodes_per_task:06d}"
                    frame_path = raw_episode / "frames_spp512" / f"{frame_index:06d}.npz"
                    with np.load(frame_path) as frame:
                        height, width = frame["DoLP"].shape
                        clean_pixels = clean_pixels_from_xyz(clean, width, height)
                        source_valid = np.zeros(height * width, dtype=bool)
                        source_valid[np.asarray(frame["point_pixel_index"], dtype=np.int32)] = True
                        if not source_valid[clean_pixels].all():
                            raise ValueError(f"{key!r}: clean points do not match rendered pixels")
                        holes = corruption_hole_pixels(clean_pixels, source_pixels)
                        filled, filled_pixels, filled_mask = add_polar_depth_filled_points(
                            incomplete, pixels, frame,
                            CAMERA_EXTRINSICS, CAMERA_FOCAL, CAMERA_CENTER,
                            hole_pixel_indices=holes,
                        )
                    if (len(holes) > len(clean) - len(incomplete) or
                            int(filled_mask.sum()) > len(holes) or
                            not np.array_equal(filled[:len(incomplete)], incomplete) or
                            not np.isfinite(filled).all()):
                        raise ValueError(f"{key!r}: filled cloud failed alignment checks")
                    projected, _ = project_world(filled[filled_mask, :3])
                    projected_xy = np.floor(projected).astype(np.int32)
                    projected_pixels = projected_xy[:, 1] * width + projected_xy[:, 0]
                    if not np.array_equal(projected_pixels, filled_pixels[filled_mask]):
                        raise ValueError(f"{key!r}: filled point/pixel alignment failed")
                    assert write_txn is not None
                    write_txn.put(key, msgpack.packb(filled))
                    stats_file.write(json.dumps({
                        "episode_index": episode, "frame_index": frame_index,
                        "task": task, "clean_points": len(clean),
                        "incomplete_points": len(incomplete),
                        "target_pixels": len(holes),
                        "filled_points": int(filled_mask.sum()),
                        "output_points": len(filled),
                    }) + "\n")
                    counts["frames"] += 1
                    counts["clean_points"] += len(clean)
                    counts["incomplete_points"] += len(incomplete)
                    counts["target_pixels"] += len(holes)
                    counts["filled_points"] += int(filled_mask.sum())
                    if frame_index == 0 and episode % 25 == 0:
                        print(f"filled episode {episode + 1}/{source_meta['total_episodes']}",
                              flush=True)
                if write_txn is not None:
                    write_txn.commit()
        output.sync()
    finally:
        output.close()
        for source in sources.values():
            source.close()

    check = lmdb.open(str(stage), readonly=True, lock=False)
    try:
        if check.stat()["entries"] != source_meta["total_frames"]:
            raise ValueError("Filled LMDB entry count does not match source dataset")
    finally:
        check.close()

    subprocess.run([
        sys.executable, str(REPO_ROOT / "data_prep/prepare_robot_state_action_stats.py"),
        "--dataset_dirs", str(root), "--output_file", str(norm_stage),
        "--point_cloud_dir", stage.name,
        "--state_xyz_slice", "0", "3", "--action_xyz_slice", "0", "3",
        "--state_rotation_slice", "3", "7", "--action_rotation_slice", "3", "7",
        "--rotation_type", "quat", "--target_rotation_type", "euler",
        "--replace_zero_std",
    ], cwd=REPO_ROOT, check=True)
    metadata = {
        "complete": True,
        "source_dataset": root.name,
        "point_cloud_dirname": POINT_DIR,
        "point_feature_mode": "xyzrgb_polar",
        "channels": ["x", "y", "z", "r", "g", "b", "DoLP", "cos2AoLP", "sin2AoLP"],
        "hole_selection": "clean_voxel_source_pixels_missing_after_corruption",
        "depth_estimation": "multiscale_morphology_on_incomplete_projected_depth",
        "rgb_polar_selection": "current_filled_pixel_from_original_render",
        "inference_limit": "hole locations use clean source provenance and are unavailable at live inference",
        "normalization_file": str(norm.relative_to(root)),
        **counts,
    }
    (stage / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    stage.rename(final)
    norm_stage.rename(norm)
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"dataset": str(root), **counts}, indent=2))


if __name__ == "__main__":
    main()
