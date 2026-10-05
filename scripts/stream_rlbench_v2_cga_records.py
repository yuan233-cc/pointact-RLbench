#!/usr/bin/env python3
"""Stream RLBench v2 CGA records as a tar archive without local output storage.

The archive contains NPZ records plus episode-disjoint JSON manifests. Extract
it into a new dataset directory; do not overlay an existing published dataset.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import av
import lmdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointact.data.rlbench_polar_normal_lmdb import generate_rlbench_cga_input  # noqa: E402


def read_npz(env: lmdb.Environment, key: bytes, label: str) -> dict[str, np.ndarray]:
    with env.begin() as transaction:
        payload = transaction.get(key)
    if payload is None:
        raise KeyError(f"Missing {label} for {key.decode()}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def episode_frames(root: Path, episode: int) -> list[np.ndarray]:
    path = root / f"videos/chunk-{episode // 1000:03d}/observation.images.front_image/episode_{episode:06d}.mp4"
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--stop-episode", type=int, default=1000000)
    parser.add_argument("--manifests-only", action="store_true")
    args = parser.parse_args()
    if args.val_every < 2:
        parser.error("--val-every must be >= 2")
    root = args.dataset_root.resolve()
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    task_counts: dict[str, int] = {}
    selected: list[tuple[str, dict, list[dict]]] = []
    manifests: dict[str, list[dict]] = {"train": [], "val": []}
    for episode in episodes:
        index = int(episode["episode_index"])
        task = str(episode["tasks"][0]).split("<br>")[0]
        ordinal = task_counts.get(task, 0)
        task_counts[task] = ordinal + 1
        split = "val" if ordinal % args.val_every == 0 else "train"
        entries = []
        for frame in range(int(episode["length"])):
            sample_id = f"episode_{index:06d}_frame_{frame:06d}"
            entries.append({
                "path": f"records/{sample_id}.npz",
                "group": f"rlbench/{task}/episode_{index:06d}",
                "id": sample_id,
                "dataset": "rlbench_polar_v2_10tasks",
            })
        manifests[split].extend(entries)
        if args.start_episode <= index < args.stop_episode:
            selected.append((split, episode, entries))
    if not args.manifests_only and not selected:
        raise ValueError("No episodes in the selected range")

    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        if not args.manifests_only:
            names = ("polar_frontview_dense", "sfp_frontview_rgb_luminance_proxy", "normal_frontview_dense")
            envs = {
                name: lmdb.open(str(root / name), readonly=True, lock=False, readahead=False, max_readers=16)
                for name in names
            }
            try:
                for number, (_, episode, entries) in enumerate(selected, start=1):
                    index = int(episode["episode_index"])
                    frames = episode_frames(root, index)
                    if len(frames) != len(entries):
                        raise ValueError(f"Episode {index}: {len(frames)} video frames != {len(entries)} manifest frames")
                    for frame, (rgb, entry) in enumerate(zip(frames, entries, strict=True)):
                        key = f"{index}-{frame}".encode()
                        polar = read_npz(envs[names[0]], key, names[0])
                        sfp = read_npz(envs[names[1]], key, names[1])
                        normal = read_npz(envs[names[2]], key, names[2])
                        if not np.allclose(sfp["K"], normal["K"], atol=1e-4):
                            raise ValueError(f"Camera calibration mismatch: {key.decode()}")
                        observation, prior = generate_rlbench_cga_input(polar, sfp, input_mode="native_cga")
                        normal_gt = np.asarray(normal["normal_gt"], dtype=np.float32).copy()
                        normal_gt[..., 0] *= -1.0  # PointACT +x left -> CGA +x right.
                        mask = np.asarray(normal["normal_valid_mask"], dtype=bool)
                        if observation.shape[1:] != rgb.shape[:2] or mask.shape != rgb.shape[:2]:
                            raise ValueError(f"Spatial shape mismatch: {key.decode()}")
                        record = {
                            "polar_observation": observation,
                            "physical_prior": prior,
                            "normal_gt": normal_gt,
                            "normal_valid_mask": mask,
                            "K": np.asarray(normal["K"], dtype=np.float32),
                            "T_camera_from_world": np.asarray(normal["T_camera_from_world"], dtype=np.float32),
                            "rgb": rgb,
                            "dataset_id": np.asarray("rlbench_polar_v2_10tasks"),
                            "source_id": np.asarray(key.decode()),
                        }
                        buffer = io.BytesIO()
                        np.savez_compressed(buffer, **record)
                        add_bytes(archive, entry["path"], buffer.getvalue())
                    if number % 10 == 0 or number == len(selected):
                        print(f"processed {number}/{len(selected)} episodes", file=sys.stderr, flush=True)
            finally:
                for env in envs.values():
                    env.close()
        else:
            for split, entries in manifests.items():
                add_bytes(archive, f"{split}_manifest.json", (json.dumps({"samples": entries}, indent=2) + "\n").encode())
            metadata = {
                "adapter": "rlbench_polar_v2_10tasks",
                "source": str(root),
                "records": sum(map(len, manifests.values())),
                "train_records": len(manifests["train"]),
                "val_records": len(manifests["val"]),
                "coordinate_frame": "+x right, +y down, +z forward; normals face camera",
                "polarization": "corrected DoLP/AoLP; four analyzer channels reconstructed from RGB-luminance proxy, not measured",
                "prior": "approximate ambiguous-normal candidates, not GT",
                "normal_supervision": "archived Coppelia depth + same-object mask finite-difference normals",
            }
            add_bytes(archive, "conversion.json", (json.dumps(metadata, indent=2) + "\n").encode())


if __name__ == "__main__":
    main()
