#!/usr/bin/env python3
"""Build scene-level dense normal supervision for the repaired RLBench polar data.

Each LMDB entry is aligned with the existing ``episode-frame`` key and stores a
full-resolution normal map rather than object crops.  Normals are estimated
from archived Coppelia depth with same-object finite differences, intersected
with the corrected dense-polar validity mask, oriented toward the front camera,
and converted to the SfP router convention (+x left, +y down, +z forward).
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np

from build_polar_rotation_aux import dense_surface_geometry
from pointact.data.polar_material import dense_polar_from_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / (
    "robot_data/rlbench/lerobot_point_lmdb/"
    "hybridvla_10tasks_train_keysteps_polar_rlbench9_v2"
)
DEFAULT_RAW = REPO_ROOT.parent / (
    "rlbench_custom_render/RLBench/output/ten_tasks_polar_train_20260921"
)
SIDECAR = "normal_frontview_dense"


def audit_sidecar(path: Path, expected_keys: list[bytes]) -> dict[str, float | int | bool]:
    """Read every staged record and verify schema, geometry, and key coverage."""
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=1)
    totals: defaultdict[str, int | float] = defaultdict(int)
    totals["max_facing_dot"] = -float("inf")
    totals["max_normal_length_error"] = 0.0
    try:
        if int(env.stat()["entries"]) != len(expected_keys):
            raise ValueError(
                f"staging has {env.stat()['entries']} entries, expected {len(expected_keys)}"
            )
        with env.begin(buffers=True) as txn:
            actual_keys = [bytes(key) for key, _ in txn.cursor()]
            if actual_keys != sorted(expected_keys):
                raise ValueError("staging LMDB keys do not match source frame keys")
            for position, key in enumerate(expected_keys, 1):
                payload = txn.get(key)
                if payload is None:
                    raise KeyError(key)
                with np.load(io.BytesIO(bytes(payload))) as record:
                    normal = np.asarray(record["normal_gt"], dtype=np.float32)
                    mask = np.asarray(record["normal_valid_mask"], dtype=bool)
                    K = np.asarray(record["K"], dtype=np.float32)
                    transform = np.asarray(record["T_camera_from_world"], dtype=np.float32)
                if normal.shape != (256, 256, 3) or mask.shape != (256, 256):
                    raise ValueError(f"{key!r}: invalid normal/mask shapes")
                if K.shape != (3, 3) or transform.shape != (4, 4):
                    raise ValueError(f"{key!r}: invalid calibration shapes")
                if not (np.isfinite(normal).all() and np.isfinite(K).all()
                        and np.isfinite(transform).all()):
                    raise ValueError(f"{key!r}: non-finite data")
                if np.any(normal[~mask] != 0):
                    raise ValueError(f"{key!r}: invalid pixels must contain zero normals")
                lengths = np.linalg.norm(normal[mask], axis=-1)
                if lengths.size:
                    length_error = float(np.max(np.abs(lengths - 1.0)))
                    if length_error > 1e-3:
                        raise ValueError(f"{key!r}: normal length error {length_error}")
                    totals["max_normal_length_error"] = max(
                        float(totals["max_normal_length_error"]), length_error
                    )
                rows, cols = np.meshgrid(
                    np.arange(256, dtype=np.float32),
                    np.arange(256, dtype=np.float32),
                    indexing="ij",
                )
                rays = np.stack(
                    (
                        (K[0, 2] - cols) / K[0, 0],
                        (rows - K[1, 2]) / K[1, 1],
                        np.ones_like(rows),
                    ),
                    axis=-1,
                )
                rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-8)
                facing_dot = np.sum(normal * rays, axis=-1)
                frame_max = float(np.max(facing_dot[mask])) if mask.any() else 0.0
                if frame_max > 1e-3:
                    raise ValueError(f"{key!r}: camera-facing check failed: {frame_max}")
                totals["max_facing_dot"] = max(float(totals["max_facing_dot"]), frame_max)
                totals["pixels"] += int(mask.size)
                totals["normal_valid"] += int(mask.sum())
                totals["frames"] += 1
                if position % 500 == 0 or position == len(expected_keys):
                    print(f"audited {position}/{len(expected_keys)} frames", flush=True)
    finally:
        env.close()
    return {
        **dict(totals),
        "all_keys_match": True,
        "all_shapes_match": True,
        "all_finite": True,
        "invalid_pixels_zero": True,
    }


def encode_record(
    depth: np.ndarray,
    object_mask: np.ndarray,
    camera: dict,
    dense_polar: np.ndarray,
    K: np.ndarray,
    camera_from_world: np.ndarray,
) -> tuple[bytes, dict[str, float | int]]:
    """Encode one full scene in the canonical SfP camera frame."""
    depth = np.asarray(depth, dtype=np.float32)
    object_mask = np.asarray(object_mask, dtype=np.int32)
    if depth.shape != object_mask.shape or dense_polar.shape != (4, *depth.shape):
        raise ValueError(
            f"unaligned depth/object/polar shapes: {depth.shape}, "
            f"{object_mask.shape}, {dense_polar.shape}"
        )
    if K.shape != (3, 3) or camera_from_world.shape != (4, 4):
        raise ValueError("invalid SfP calibration shapes")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("SfP intrinsics must have positive focal lengths")

    points_world, normal_world, geometry_valid = dense_surface_geometry(
        depth, object_mask, camera
    )
    polar_valid = dense_polar[3] > 0.5
    valid = geometry_valid & polar_valid

    # T_camera_from_world uses an OpenCV-like (+right,+down,+forward) frame.
    # The SfP router negates camera x, yielding (+left,+down,+forward).
    normal_opencv = normal_world @ camera_from_world[:3, :3].T
    normal_sfp = normal_opencv * np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
    lengths = np.linalg.norm(normal_sfp, axis=-1)
    valid &= np.isfinite(normal_sfp).all(axis=-1) & np.isfinite(lengths) & (lengths > 1e-8)
    normal_sfp[valid] /= lengths[valid, None]
    normal_sfp[~valid] = 0.0

    # Independently verify that valid normals face the camera in the same SfP
    # coordinate convention used by PointACT's viewing-direction channels.
    height, width = depth.shape
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    rays = np.stack(
        (
            (K[0, 2] - cols) / K[0, 0],
            (rows - K[1, 2]) / K[1, 1],
            np.ones_like(rows),
        ),
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-8)
    facing_dot = np.sum(normal_sfp * rays, axis=-1)
    if valid.any() and float(np.max(facing_dot[valid])) > 1e-4:
        raise ValueError("normal orientation check failed")

    # Validate that the archived transform still maps the depth-derived world
    # surface in front of the camera after the native-to-OpenCV conversion.
    homogeneous = np.concatenate(
        (points_world, np.ones((*depth.shape, 1), dtype=np.float32)), axis=-1
    )
    points_opencv = homogeneous @ camera_from_world.T
    if valid.any() and np.any(points_opencv[..., 2][valid] <= 0):
        raise ValueError("valid normal supervision lies behind the camera")

    output = io.BytesIO()
    np.savez_compressed(
        output,
        normal_gt=normal_sfp.astype(np.float16),
        normal_valid_mask=valid.astype(np.uint8),
        K=np.asarray(K, dtype=np.float32),
        T_camera_from_world=np.asarray(camera_from_world, dtype=np.float32),
    )
    stats = {
        "pixels": int(depth.size),
        "geometry_valid": int(geometry_valid.sum()),
        "polar_valid": int(polar_valid.sum()),
        "normal_valid": int(valid.sum()),
        "max_facing_dot": float(np.max(facing_dot[valid])) if valid.any() else 0.0,
    }
    return output.getvalue(), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--map-size-gb", type=int, default=4)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Audit and publish a complete .building LMDB after an interrupted final sync",
    )
    args = parser.parse_args()

    root = args.dataset.resolve()
    raw = args.raw.resolve()
    final = (args.output.resolve() if args.output else root / SIDECAR)
    stage = final.with_name(final.name + ".building")
    manifest_path = root / "meta/normal_frontview_dense.json"
    manifest_stage = manifest_path.with_name(manifest_path.name + ".building")
    protected = [final]
    if args.output is None:
        protected.extend((manifest_path, manifest_stage))
    if not args.finalize_existing:
        protected.append(stage)
    for path in protected:
        if path.exists():
            raise FileExistsError(f"refusing to replace existing output: {path}")
    if args.limit_frames is not None and args.limit_frames <= 0:
        raise ValueError("--limit-frames must be positive")

    repair_meta = json.loads((root / "meta/polar_rlbench9_repair.json").read_text())
    if args.finalize_existing:
        if args.output is not None or args.limit_frames is not None:
            raise ValueError("--finalize-existing is only valid for the default full sidecar")
        if not stage.is_dir():
            raise FileNotFoundError(stage)
        source_records = [
            json.loads(line)
            for line in (root / "frame_corruption_stats.jsonl").read_text().splitlines()
        ]
        expected_keys = [
            f"{int(record['episode_index'])}-{int(record['frame_index'])}".encode("ascii")
            for record in source_records
        ]
        audit = audit_sidecar(stage, expected_keys)
        sync_env = lmdb.open(str(stage), map_size=args.map_size_gb * 1024**3, sync=True)
        try:
            sync_env.sync()
        finally:
            sync_env.close()
        summary = {
            "complete": True,
            "sidecar_dirname": final.name,
            "source_dataset": root.name,
            "source_dense_polar_dirname": "polar_frontview_dense",
            "source_geometry": "archived Coppelia depth and object mask",
            "normal_estimator": "same-object finite differences",
            "normal_field": "normal_gt",
            "normal_storage_dtype": "float16",
            "serialization": "compressed NPZ inside LMDB",
            "normal_shape": [256, 256, 3],
            "mask_field": "normal_valid_mask",
            "mask_definition": "geometry normal valid AND corrected dense polar valid",
            "coordinate_convention": "+x left, +y down, +z forward; camera-facing",
            "render_scope": "full scene; no object crop or object-only supervision",
            "calibration_fields": ["K", "T_camera_from_world"],
            "key_format": "{global_episode_index}-{episode_frame_index}",
            "entries": len(expected_keys),
            "totals": {
                "frames": int(audit["frames"]),
                "pixels": int(audit["pixels"]),
                "normal_valid": int(audit["normal_valid"]),
            },
            "valid_fraction": float(
                int(audit["normal_valid"]) / max(int(audit["pixels"]), 1)
            ),
            "audit": audit,
        }
        manifest_stage.write_text(json.dumps(summary, indent=2) + "\n")
        stage.rename(final)
        manifest_stage.rename(manifest_path)
        print(json.dumps(summary, indent=2), flush=True)
        return

    sfp_meta = json.loads((root / "meta/sfp_wild_inputs.json").read_text())
    if not repair_meta.get("complete") or not sfp_meta.get("complete"):
        raise ValueError("source dataset sidecars are incomplete")
    records = [
        json.loads(line)
        for line in (root / "frame_corruption_stats.jsonl").read_text().splitlines()
    ]
    expected_full = int(repair_meta["total_frames"])
    if len(records) != expected_full:
        raise ValueError(f"expected {expected_full} source records, got {len(records)}")
    if args.limit_frames is not None:
        records = records[: args.limit_frames]

    tasks = list(repair_meta["tasks"])
    episodes_per_task = int(repair_meta["total_episodes"]) // len(tasks)
    polar_env = lmdb.open(
        str(root / "polar_frontview_dense"),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=2,
    )
    sfp_env = lmdb.open(
        str(root / sfp_meta["sidecar_dirname"]),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=2,
    )
    stage.mkdir(parents=True)
    # This is a new staging database, so batch transactions and perform one
    # forced sync before publication instead of fsyncing every short episode.
    output_env = lmdb.open(str(stage), map_size=args.map_size_gb * 1024**3, sync=False)
    totals: defaultdict[str, int | float] = defaultdict(int)
    totals["max_facing_dot"] = -float("inf")
    write_txn = None
    try:
        with polar_env.begin(buffers=True) as polar_txn, sfp_env.begin(buffers=True) as sfp_txn:
            for position, record in enumerate(records, 1):
                episode = int(record["episode_index"])
                frame = int(record["frame_index"])
                task = tasks[episode // episodes_per_task]
                if record["task"] != task:
                    raise ValueError(f"source task mismatch at record {position}")
                if write_txn is None:
                    write_txn = output_env.begin(write=True)

                key = f"{episode}-{frame}".encode("ascii")
                polar_payload = polar_txn.get(key)
                sfp_payload = sfp_txn.get(key)
                if polar_payload is None or sfp_payload is None:
                    raise KeyError(f"missing aligned sidecar key {key!r}")
                dense_polar = dense_polar_from_bytes(bytes(polar_payload))
                with np.load(io.BytesIO(bytes(sfp_payload))) as calibration:
                    K = np.asarray(calibration["K"], dtype=np.float32)
                    camera_from_world = np.asarray(
                        calibration["T_camera_from_world"], dtype=np.float32
                    )

                raw_episode = raw / task / f"episode_{episode % episodes_per_task:06d}"
                frame_path = raw_episode / "frames" / f"{frame:06d}.npz"
                snapshot_path = raw_episode / "snapshots/frames" / f"{frame:06d}.json"
                with np.load(frame_path) as source:
                    depth = np.asarray(source["coppelia_depth_m"], dtype=np.float32)
                    object_mask = np.asarray(source["coppelia_object_mask"], dtype=np.int32)
                camera = json.loads(snapshot_path.read_text())["cameras"]["front"]

                payload, stats = encode_record(
                    depth, object_mask, camera, dense_polar, K, camera_from_world
                )
                if not write_txn.put(key, payload, overwrite=False):
                    raise ValueError(f"duplicate key {key!r}")
                for name in ("pixels", "geometry_valid", "polar_valid", "normal_valid"):
                    totals[name] += int(stats[name])
                totals["max_facing_dot"] = max(
                    float(totals["max_facing_dot"]), float(stats["max_facing_dot"])
                )
                totals["frames"] += 1
                if position % 100 == 0 or position == len(records):
                    print(f"encoded {position}/{len(records)} frames", flush=True)
                if position % 250 == 0 or position == len(records):
                    write_txn.commit()
                    write_txn = None
            if write_txn is not None:
                write_txn.commit()
                write_txn = None
            output_env.sync()
    except Exception:
        if write_txn is not None:
            write_txn.abort()
        raise
    finally:
        output_env.close()
        sfp_env.close()
        polar_env.close()

    check = lmdb.open(str(stage), readonly=True, lock=False, readahead=False)
    try:
        entries = int(check.stat()["entries"])
    finally:
        check.close()
    if entries != len(records):
        raise ValueError(f"sidecar has {entries} entries, expected {len(records)}")

    summary = {
        "complete": args.limit_frames is None,
        "sidecar_dirname": final.name,
        "source_dataset": root.name,
        "source_dense_polar_dirname": "polar_frontview_dense",
        "source_geometry": "archived Coppelia depth and object mask",
        "normal_estimator": "same-object finite differences",
        "normal_field": "normal_gt",
        "normal_storage_dtype": "float16",
        "serialization": "compressed NPZ inside LMDB",
        "normal_shape": [256, 256, 3],
        "mask_field": "normal_valid_mask",
        "mask_definition": "geometry normal valid AND corrected dense polar valid",
        "coordinate_convention": "+x left, +y down, +z forward; camera-facing",
        "render_scope": "full scene; no object crop or object-only supervision",
        "calibration_fields": ["K", "T_camera_from_world"],
        "key_format": "{global_episode_index}-{episode_frame_index}",
        "entries": entries,
        "totals": dict(totals),
        "valid_fraction": float(int(totals["normal_valid"]) / max(int(totals["pixels"]), 1)),
    }
    if args.output is None:
        manifest_stage.write_text(json.dumps(summary, indent=2) + "\n")
    stage.rename(final)
    if args.output is None:
        manifest_stage.rename(manifest_path)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
