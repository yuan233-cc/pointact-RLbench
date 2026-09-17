"""Create a non-destructive 25% structured-missing derivative of an RLBench LMDB.

The stored PointACT training clouds are unordered voxelized xyzrgb arrays and do
not contain the original instance mask or pixel grid.  Consequently this tool
uses deterministic local 3-D ellipsoidal holes over the complete stored scene;
it does not claim to reproduce target-semantic image-space corruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np


msgpack_numpy.patch()


def key_seed(base_seed: int, key: bytes) -> int:
    digest = hashlib.blake2b(key, digest_size=8, person=b"ptmiss25").digest()
    return (base_seed + int.from_bytes(digest, "little")) % (2**63 - 1)


def structured_missing_mask(
    xyz: np.ndarray,
    missing_rate: float,
    seed: int,
    num_holes: int = 6,
) -> np.ndarray:
    """Choose exactly ``missing_rate`` points in spatially local 3-D patches."""
    count = int(round(len(xyz) * missing_rate))
    missing = np.zeros(len(xyz), dtype=bool)
    if count <= 0:
        return missing
    if count >= len(xyz):
        missing[:] = True
        return missing

    rng = np.random.default_rng(seed)
    center_indices = rng.choice(len(xyz), size=min(num_holes, len(xyz)), replace=False)
    centers = xyz[center_indices]
    # Different anisotropic radii produce holes resembling local surface loss
    # rather than independent Bernoulli point dropout.
    radii = rng.uniform(
        low=np.asarray([0.035, 0.035, 0.025]),
        high=np.asarray([0.14, 0.14, 0.18]),
        size=(len(centers), 3),
    )
    normalized = (xyz[:, None, :] - centers[None, :, :]) / radii[None, :, :]
    score = np.square(normalized).sum(axis=-1).min(axis=1)
    score += rng.normal(0.0, 1e-4, size=len(score))
    chosen = np.argpartition(score, count - 1)[:count]
    missing[chosen] = True
    return missing


def link_unchanged_content(source: Path, output: Path) -> None:
    for name in ("data", "meta", "videos", "robot_state_action_stats"):
        source_item = source / name
        if not source_item.exists():
            raise FileNotFoundError(source_item)
        os.symlink(source_item, output / name, target_is_directory=source_item.is_dir())


def create_dataset(
    source: Path,
    output: Path,
    missing_rate: float,
    seed: int,
    num_holes: int,
    commit_every: int,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not 0.0 < missing_rate < 1.0:
        raise ValueError("missing_rate must be in (0, 1)")
    source_points = source / "points_frontview"
    if not source_points.is_dir():
        raise FileNotFoundError(source_points)

    output.mkdir(parents=True)
    link_unchanged_content(source, output)
    output_points = output / "points_frontview"
    source_size = (source_points / "data.mdb").stat().st_size
    map_size = max(2 * source_size + 512 * 1024**2, 2 * 1024**3)

    total_input = 0
    total_output = 0
    total_frames = 0
    minimum_input = None
    maximum_input = 0
    input_env = lmdb.open(
        str(source_points), readonly=True, lock=False, readahead=False, max_readers=2
    )
    output_env = lmdb.open(str(output_points), map_size=map_size, subdir=True)
    try:
        with input_env.begin(buffers=True) as input_txn:
            cursor = input_txn.cursor()
            output_txn = output_env.begin(write=True)
            try:
                for key_view, value_view in cursor:
                    key = bytes(key_view)
                    cloud = np.asarray(msgpack.unpackb(value_view), dtype=np.float32)
                    if cloud.ndim != 2 or cloud.shape[1] != 6:
                        raise ValueError(f"{key!r}: expected Nx6 xyzrgb, got {cloud.shape}")
                    missing = structured_missing_mask(
                        cloud[:, :3], missing_rate, key_seed(seed, key), num_holes
                    )
                    incomplete = np.ascontiguousarray(cloud[~missing], dtype=np.float32)
                    output_txn.put(key, msgpack.packb(incomplete))

                    total_frames += 1
                    total_input += len(cloud)
                    total_output += len(incomplete)
                    minimum_input = len(cloud) if minimum_input is None else min(minimum_input, len(cloud))
                    maximum_input = max(maximum_input, len(cloud))
                    if total_frames % commit_every == 0:
                        output_txn.commit()
                        output_txn = output_env.begin(write=True)
                        print(
                            f"processed {total_frames:,} frames; "
                            f"kept {total_output:,}/{total_input:,} points",
                            flush=True,
                        )
                output_txn.commit()
                output_txn = None
            finally:
                if output_txn is not None:
                    output_txn.abort()
        output_env.sync()
    finally:
        input_env.close()
        output_env.close()

    report = {
        "source_dataset": str(source.resolve()),
        "output_dataset": str(output.resolve()),
        "source_point_cloud_dir": "points_frontview",
        "output_point_cloud_dir": "points_frontview",
        "total_frames": total_frames,
        "total_input_points": total_input,
        "total_output_points": total_output,
        "requested_missing_rate": missing_rate,
        "actual_global_missing_rate": 1.0 - total_output / total_input,
        "per_frame_count_rule": "round(N * missing_rate) points are removed from every frame",
        "corruption_geometry": {
            "type": "union of deterministic anisotropic local 3-D holes",
            "num_hole_seeds_per_frame": num_holes,
            "seed": seed,
            "independent_bernoulli_dropout": False,
        },
        "unchanged": [
            "RGB videos", "robot state", "actions", "language", "episode/task split",
            "point coordinates and RGB of retained points",
        ],
        "important_limitation": (
            "The source LMDB has no instance masks, organized pixel grid, depth image, "
            "or camera metadata. This derivative applies structured holes to the whole "
            "stored scene and is not the target-semantic corruption used in online evaluation."
        ),
        "source_point_count_range": [minimum_input, maximum_input],
        "storage": (
            "data/meta/videos/robot_state_action_stats are read-only symlinks to the source; "
            "points_frontview is a new independent LMDB"
        ),
    }
    (output / "corruption_metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# RLBench structured incomplete point-cloud dataset (25%)\n\n"
        "This derivative preserves the original dataset and replaces only the "
        "`points_frontview` LMDB. Each frame has exactly 25% of its stored points "
        "removed in deterministic, spatially local 3-D holes.\n\n"
        "It is a whole-scene geometry-only approximation. The original LeRobot "
        "dataset does not retain instance masks or the organized depth grid, so this "
        "is not target-semantic transparent-object corruption. See "
        "`corruption_metadata.json` for the exact protocol.\n",
        encoding="utf-8",
    )
    return report


def validate(source: Path, output: Path, missing_rate: float) -> dict:
    source_env = lmdb.open(str(source / "points_frontview"), readonly=True, lock=False)
    output_env = lmdb.open(str(output / "points_frontview"), readonly=True, lock=False)
    failures = []
    checked = 0
    try:
        with source_env.begin(buffers=True) as source_txn, output_env.begin(buffers=True) as output_txn:
            if source_txn.stat()["entries"] != output_txn.stat()["entries"]:
                failures.append("LMDB entry counts differ")
            for key, source_value in source_txn.cursor():
                output_value = output_txn.get(key)
                if output_value is None:
                    failures.append(f"missing output key {bytes(key)!r}")
                    continue
                source_cloud = np.asarray(msgpack.unpackb(source_value))
                output_cloud = np.asarray(msgpack.unpackb(output_value))
                expected = len(source_cloud) - int(round(len(source_cloud)*missing_rate))
                if len(output_cloud) != expected or output_cloud.shape[1:] != (6,):
                    failures.append(
                        f"{bytes(key)!r}: output {output_cloud.shape}, expected ({expected}, 6)"
                    )
                checked += 1
    finally:
        source_env.close()
        output_env.close()
    result = {"checked_frames": checked, "passed": not failures, "failures": failures[:20]}
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Validation failed: {failures[:3]}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-rate", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--num-holes", type=int, default=6)
    parser.add_argument("--commit-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = create_dataset(
        args.source, args.output, args.missing_rate, args.seed,
        args.num_holes, args.commit_every,
    )
    report["validation"] = validate(args.source, args.output, args.missing_rate)
    (args.output / "corruption_metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
