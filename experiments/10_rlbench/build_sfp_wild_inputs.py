#!/usr/bin/env python3
"""Append aligned single-view SfP-Wild inputs to the repaired RLBench dataset.

This exporter does not recollect trajectories or change the LeRobot episodes.
The immediately usable mode derives I_un from archived same-frame RGB luminance;
the manifest labels that intensity as a proxy rather than corrected physical S0.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import lmdb
import numpy as np

from pointact.data.polar_material import dense_polar_from_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / (
    "robot_data/rlbench/lerobot_point_lmdb/"
    "hybridvla_10tasks_train_keysteps_polar_rlbench9_v2"
)
DEFAULT_RAW = REPO_ROOT.parent / (
    "rlbench_custom_render/RLBench/output/ten_tasks_polar_train_20260921"
)
SIDECAR = "sfp_frontview_rgb_luminance_proxy"
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
CAMERA_AXIS_CONVERSION = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def encode_record(rgb: np.ndarray, camera: dict) -> tuple[bytes, float]:
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 256x256 RGB, got {rgb.shape} {rgb.dtype}")
    source_K = np.asarray(camera["intrinsics"], dtype=np.float64)
    to_world = np.asarray(camera["to_world"], dtype=np.float64)
    if source_K.shape != (3, 3) or to_world.shape != (4, 4):
        raise ValueError("invalid archived front-camera calibration")
    if source_K[0, 0] >= 0 or source_K[1, 1] >= 0:
        raise ValueError("expected the archived RLBench negative-focal convention")
    K = source_K.copy()
    K[0, 0] *= -1
    K[1, 1] *= -1
    camera_from_world = CAMERA_AXIS_CONVERSION @ np.linalg.inv(to_world)

    # PointACT's router uses the positive-focal OpenCV-like camera frame. Check
    # that the converted calibration still projects representative pixels onto
    # themselves before serializing it.
    rows, cols = np.meshgrid(
        np.linspace(0, rgb.shape[0] - 1, 17),
        np.linspace(0, rgb.shape[1] - 1, 17),
        indexing="ij",
    )
    depth = np.linspace(0.25, 4.0, rows.size)
    native = np.column_stack((
        (cols.ravel() - source_K[0, 2]) * depth / source_K[0, 0],
        (rows.ravel() - source_K[1, 2]) * depth / source_K[1, 1],
        depth,
        np.ones_like(depth),
    ))
    world = native @ to_world.T
    converted = world @ camera_from_world.T
    uv_h = converted[:, :3] @ K.T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    expected = np.column_stack((cols.ravel(), rows.ravel()))
    max_error = float(np.max(np.abs(uv - expected)))
    if max_error > 1e-4 or np.any(converted[:, 2] <= 0):
        raise ValueError(f"camera convention conversion failed: max error {max_error}")

    luminance = np.clip(np.rint(rgb.astype(np.float32) @ LUMA), 0, 255).astype(np.uint8)
    output = io.BytesIO()
    np.savez_compressed(
        output,
        I_un=luminance,
        K=K.astype(np.float32),
        T_camera_from_world=camera_from_world.astype(np.float32),
    )
    return output.getvalue(), max_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--map-size-gb", type=int, default=2)
    args = parser.parse_args()
    root = args.dataset.resolve()
    raw = args.raw.resolve()
    manifest_path = root / "meta/sfp_wild_inputs.json"
    final = root / SIDECAR
    stage = root / f"{SIDECAR}.building"
    for path in (manifest_path, final, stage):
        if path.exists():
            raise FileExistsError(f"refusing to replace existing output: {path}")

    source_meta = json.loads((root / "meta/polar_incomplete_features.json").read_text())
    repair_meta = json.loads((root / "meta/polar_rlbench9_repair.json").read_text())
    if not source_meta.get("complete") or not repair_meta.get("complete"):
        raise ValueError("the repaired source dataset is incomplete")
    records = [json.loads(line) for line in
               (root / "frame_corruption_stats.jsonl").read_text().splitlines()]
    expected = int(source_meta["total_frames"])
    if len(records) != expected:
        raise ValueError(f"expected {expected} frame records, got {len(records)}")
    tasks = source_meta["tasks"]
    episodes_per_task = int(source_meta["episodes_per_task"])

    dense_env = lmdb.open(str(root / source_meta["dense_polar_dirname"]), readonly=True,
                          lock=False, readahead=False, max_readers=2)
    stage.mkdir()
    output = lmdb.open(str(stage), map_size=args.map_size_gb * 1024**3)
    maximum_projection_error = 0.0
    current_episode = None
    write_txn = None
    try:
        with dense_env.begin(buffers=True) as dense_txn:
            for position, record in enumerate(records, 1):
                episode = int(record["episode_index"])
                frame = int(record["frame_index"])
                task = tasks[episode // episodes_per_task]
                if record["task"] != task:
                    raise ValueError(f"frame record {position} has an unexpected task")
                if current_episode != episode:
                    if write_txn is not None:
                        write_txn.commit()
                    write_txn = output.begin(write=True)
                    current_episode = episode
                key = f"{episode}-{frame}".encode("ascii")
                dense_bytes = dense_txn.get(key)
                if dense_bytes is None:
                    raise KeyError(f"missing dense polar frame {key!r}")
                dense = dense_polar_from_bytes(bytes(dense_bytes))
                raw_episode = raw / task / f"episode_{episode % episodes_per_task:06d}"
                frame_path = raw_episode / "frames" / f"{frame:06d}.npz"
                snapshot_path = raw_episode / "snapshots/frames" / f"{frame:06d}.json"
                with np.load(frame_path) as source:
                    rgb = np.asarray(source["rgb"])
                camera = json.loads(snapshot_path.read_text())["cameras"]["front"]
                if dense.shape != (4, *rgb.shape[:2]):
                    raise ValueError(f"dense polar/RGB mismatch at {key!r}")
                payload, error = encode_record(rgb, camera)
                maximum_projection_error = max(maximum_projection_error, error)
                write_txn.put(key, payload)
                if position % 250 == 0 or position == expected:
                    print(f"encoded {position}/{expected} frames", flush=True)
            if write_txn is not None:
                write_txn.commit()
                write_txn = None
    except Exception:
        if write_txn is not None:
            write_txn.abort()
        raise
    finally:
        output.close()
        dense_env.close()

    check = lmdb.open(str(stage), readonly=True, lock=False, readahead=False)
    try:
        entries = int(check.stat()["entries"])
    finally:
        check.close()
    if entries != expected:
        raise ValueError(f"sidecar has {entries} entries, expected {expected}")
    stage.rename(final)
    manifest = {
        "complete": True,
        "sidecar_dirname": SIDECAR,
        "dense_polar_dirname": source_meta["dense_polar_dirname"],
        "views": 1,
        "frames": entries,
        "image_shape": [256, 256],
        "serialized_fields": ["I_un", "K", "T_camera_from_world"],
        "assembled_channels": [
            "I_un", "DoLP", "cos2AoLP", "sin2AoLP", "view_x", "view_y", "view_z"
        ],
        "intensity_source": "archived_same_frame_coppeliasim_rgb_luminance_proxy",
        "intensity_units": "uint8_luminance_divided_by_255_at_load",
        "physical_s0": False,
        "warning": (
            "This proxy is suitable for pipeline training/smoke experiments but is not the "
            "corrected native-renderer S0 used to produce the v2 DoLP/AoLP. A physics claim "
            "requires re-rendering S0 from the corrected scene snapshots."
        ),
        "camera_convention": (
            "canonical positive-focal frame (+right,+down,+forward); legacy SfP-Wild "
            "conversion happens only at the released checkpoint boundary"
        ),
        "max_projection_roundtrip_error_pixels": maximum_projection_error,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": str(final), "frames": entries}, indent=2))


if __name__ == "__main__":
    main()
