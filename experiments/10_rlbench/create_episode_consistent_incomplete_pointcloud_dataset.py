"""Create a temporally consistent structured-missing RLBench point-cloud dataset.

Every frame in one episode is evaluated against the same deterministic 3-D
ellipsoid field.  The lowest-scoring ``missing_rate`` fraction is removed from
each frame, so the spatial corruption pattern is stable while the per-frame
missing count remains exact.

The source PointACT LMDB contains unordered xyzrgb points without semantic
labels.  This script therefore corrupts the complete scene and does not claim
to identify or protect robot points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np


msgpack_numpy.patch()


@dataclass(frozen=True)
class HoleField:
    centers: np.ndarray
    radii: np.ndarray


def parse_point_key(key: bytes) -> tuple[int, int]:
    try:
        episode, frame = key.decode("ascii").rsplit("-", 1)
        return int(episode), int(frame)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Unexpected point-cloud key: {key!r}") from exc


def episode_seed(base_seed: int, episode: int) -> int:
    token = str(episode).encode("ascii")
    digest = hashlib.blake2b(token, digest_size=8, person=b"ptepisode").digest()
    return (base_seed + int.from_bytes(digest, "little")) % (2**63 - 1)


def make_hole_field(
    reference_xyz: np.ndarray,
    seed: int,
    num_holes: int,
) -> HoleField:
    """Sample one fixed world-coordinate hole field for an entire episode."""
    rng = np.random.default_rng(seed)
    count = min(num_holes, len(reference_xyz))
    indices = rng.choice(len(reference_xyz), size=count, replace=False)
    centers = np.asarray(reference_xyz[indices], dtype=np.float32)
    radii = rng.uniform(
        low=np.asarray([0.035, 0.035, 0.025]),
        high=np.asarray([0.14, 0.14, 0.18]),
        size=(count, 3),
    ).astype(np.float32)
    return HoleField(centers=centers, radii=radii)


def consistent_missing_mask(
    xyz: np.ndarray,
    missing_rate: float,
    field: HoleField,
) -> np.ndarray:
    """Remove the exact requested fraction using a shared spatial score field."""
    count = int(round(len(xyz) * missing_rate))
    missing = np.zeros(len(xyz), dtype=bool)
    if count <= 0:
        return missing
    if count >= len(xyz):
        missing[:] = True
        return missing

    normalized = (
        (xyz[:, None, :] - field.centers[None, :, :])
        / field.radii[None, :, :]
    )
    score = np.square(normalized).sum(axis=-1).min(axis=1)
    chosen = np.argpartition(score, count - 1)[:count]
    missing[chosen] = True
    return missing


def link_unchanged_content(source: Path, output: Path) -> None:
    for name in ("data", "meta", "videos", "robot_state_action_stats"):
        source_item = source / name
        if not source_item.exists():
            raise FileNotFoundError(source_item)
        os.symlink(source_item, output / name, target_is_directory=source_item.is_dir())


def load_cloud(txn: lmdb.Transaction, key: bytes) -> np.ndarray:
    value = txn.get(key)
    if value is None:
        raise KeyError(key)
    cloud = np.asarray(msgpack.unpackb(value), dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 6:
        raise ValueError(f"{key!r}: expected Nx6 xyzrgb, got {cloud.shape}")
    return cloud


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
    if num_holes <= 0:
        raise ValueError("num_holes must be positive")

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
    episodes: set[int] = set()
    fields: dict[int, HoleField] = {}

    input_env = lmdb.open(
        str(source_points), readonly=True, lock=False, readahead=False, max_readers=2
    )
    output_env = lmdb.open(str(output_points), map_size=map_size, subdir=True)
    try:
        with input_env.begin(buffers=False) as input_txn:
            output_txn = output_env.begin(write=True)
            try:
                for key_view, value_view in input_txn.cursor():
                    key = bytes(key_view)
                    episode, _ = parse_point_key(key)
                    cloud = np.asarray(msgpack.unpackb(value_view), dtype=np.float32)
                    if cloud.ndim != 2 or cloud.shape[1] != 6:
                        raise ValueError(
                            f"{key!r}: expected Nx6 xyzrgb, got {cloud.shape}"
                        )

                    if episode not in fields:
                        reference_key = f"{episode}-0".encode("ascii")
                        reference_cloud = load_cloud(input_txn, reference_key)
                        fields[episode] = make_hole_field(
                            reference_cloud[:, :3],
                            episode_seed(seed, episode),
                            num_holes,
                        )
                    field = fields[episode]
                    missing = consistent_missing_mask(
                        cloud[:, :3], missing_rate, field
                    )
                    incomplete = np.ascontiguousarray(cloud[~missing], dtype=np.float32)
                    output_txn.put(key, msgpack.packb(incomplete))

                    episodes.add(episode)
                    total_frames += 1
                    total_input += len(cloud)
                    total_output += len(incomplete)
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
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_input_points": total_input,
        "total_output_points": total_output,
        "requested_missing_rate": missing_rate,
        "actual_global_missing_rate": 1.0 - total_output / total_input,
        "per_frame_count_rule": "round(N * missing_rate) points removed per frame",
        "temporal_consistency": {
            "scope": "episode",
            "coordinate_frame": "RLBench world coordinates",
            "rule": (
                "All frames in an episode use identical ellipsoid centers and radii; "
                "each frame applies its own 25% score quantile for an exact count."
            ),
            "new_field_at_next_episode": True,
        },
        "corruption_geometry": {
            "type": "shared anisotropic 3-D ellipsoid score field",
            "num_hole_seeds_per_episode": num_holes,
            "seed": seed,
            "independent_per_frame_randomness": False,
        },
        "unchanged": [
            "RGB videos",
            "robot state",
            "actions",
            "language",
            "episode/task split",
            "point coordinates and RGB of retained points",
        ],
        "important_limitation": (
            "The source LMDB has no semantic/instance masks, organized depth image, "
            "or camera metadata. The shared holes affect the complete stored scene "
            "and do not protect robot points."
        ),
        "storage": (
            "data/meta/videos/robot_state_action_stats are read-only symlinks to the "
            "source; points_frontview is a new independent LMDB"
        ),
    }
    (output / "corruption_metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# RLBench episode-consistent incomplete point clouds (25%)\n\n"
        "This derivative preserves the original dataset and replaces only the "
        "`points_frontview` LMDB. Every frame removes exactly 25% of its stored "
        "points. Frames in one episode share identical world-coordinate ellipsoid "
        "centers and radii; a new field is sampled for the next episode.\n\n"
        "This is a whole-scene geometry-only approximation. It does not identify "
        "or protect robot points because semantic masks are absent from the source "
        "LMDB. See `corruption_metadata.json` for the exact protocol.\n",
        encoding="utf-8",
    )
    return report


def validate(source: Path, output: Path, missing_rate: float) -> dict:
    source_env = lmdb.open(str(source / "points_frontview"), readonly=True, lock=False)
    output_env = lmdb.open(str(output / "points_frontview"), readonly=True, lock=False)
    failures: list[str] = []
    checked = 0
    episodes: set[int] = set()
    try:
        with source_env.begin(buffers=True) as source_txn, output_env.begin(
            buffers=True
        ) as output_txn:
            if source_txn.stat()["entries"] != output_txn.stat()["entries"]:
                failures.append("LMDB entry counts differ")
            for key, source_value in source_txn.cursor():
                output_value = output_txn.get(key)
                if output_value is None:
                    failures.append(f"missing output key {bytes(key)!r}")
                    continue
                source_cloud = np.asarray(msgpack.unpackb(source_value))
                output_cloud = np.asarray(msgpack.unpackb(output_value))
                expected = len(source_cloud) - int(round(len(source_cloud) * missing_rate))
                if len(output_cloud) != expected or output_cloud.shape[1:] != (6,):
                    failures.append(
                        f"{bytes(key)!r}: output {output_cloud.shape}, "
                        f"expected ({expected}, 6)"
                    )
                episode, _ = parse_point_key(bytes(key))
                episodes.add(episode)
                checked += 1
    finally:
        source_env.close()
        output_env.close()

    result = {
        "checked_frames": checked,
        "checked_episodes": len(episodes),
        "passed": not failures,
        "failures": failures[:20],
    }
    (output / "validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Validation failed: {failures[:3]}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--num-holes", type=int, default=6)
    parser.add_argument("--commit-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = create_dataset(
        args.source,
        args.output,
        args.missing_rate,
        args.seed,
        args.num_holes,
        args.commit_every,
    )
    report["validation"] = validate(args.source, args.output, args.missing_rate)
    (args.output / "corruption_metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
