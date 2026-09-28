"""Pack water-plants pre-pour/pour observations for the state influence test."""

from __future__ import annotations

import argparse
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


msgpack_numpy.patch()


def pose_to_euler(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.concatenate(
        [
            value[:3],
            Rotation.from_quat(value[3:7]).as_euler("xyz").astype(np.float32),
            value[7:8],
        ]
    ).astype(np.float32)


def object_array(values: list[np.ndarray]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    output[:] = values
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--incomplete-dataset-root",
        type=Path,
        help="Optional derivative dataset whose points are packed as incomplete_points.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-start", type=int, default=900)
    parser.add_argument("--episode-end", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    episodes: list[int] = []
    frames: list[int] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    points: list[np.ndarray] = []
    environment = lmdb.open(
        str(args.dataset_root / "points_frontview"),
        readonly=True,
        lock=False,
        readahead=False,
    )
    incomplete_environment = None
    if args.incomplete_dataset_root is not None:
        incomplete_environment = lmdb.open(
            str(args.incomplete_dataset_root / "points_frontview"),
            readonly=True,
            lock=False,
            readahead=False,
        )
    incomplete_points: list[np.ndarray] = []
    try:
        with environment.begin() as transaction:
            incomplete_transaction = (
                incomplete_environment.begin()
                if incomplete_environment is not None
                else None
            )
            for episode in range(args.episode_start, args.episode_end):
                parquet = (
                    args.dataset_root
                    / "data/chunk-000"
                    / f"episode_{episode:06d}.parquet"
                )
                frame_table = pd.read_parquet(parquet)
                for frame in (2, 3):
                    row = frame_table.iloc[frame]
                    value = transaction.get(f"{episode}-{frame}".encode("ascii"))
                    if value is None:
                        raise KeyError(f"{episode}-{frame}")
                    incomplete_value = (
                        incomplete_transaction.get(f"{episode}-{frame}".encode("ascii"))
                        if incomplete_transaction is not None
                        else value
                    )
                    if incomplete_value is None:
                        raise KeyError(f"incomplete {episode}-{frame}")
                    episodes.append(episode)
                    frames.append(frame)
                    states.append(pose_to_euler(row["observation.state"]))
                    actions.append(pose_to_euler(row["action"]))
                    points.append(np.asarray(msgpack.unpackb(value), dtype=np.float32))
                    incomplete_points.append(
                        np.asarray(msgpack.unpackb(incomplete_value), dtype=np.float32)
                    )
    finally:
        environment.close()
        if incomplete_environment is not None:
            incomplete_environment.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    packed_points = object_array(points)
    np.savez_compressed(
        args.output,
        episode=np.asarray(episodes),
        frame=np.asarray(frames),
        state=np.stack(states),
        action=np.stack(actions),
        complete_points=packed_points,
        incomplete_points=object_array(incomplete_points),
    )
    print(
        f"Wrote {len(episodes)} samples, {sum(map(len, points))} points, "
        f"{args.output.stat().st_size} bytes to {args.output}"
    )


if __name__ == "__main__":
    main()
