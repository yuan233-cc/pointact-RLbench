#!/usr/bin/env python3
"""Render a TaskNet input sidecar from one physically consistent Stokes render.

The existing V2 RGB is a CoppeliaSim image while its DoLP/AoLP comes from the
native Mueller path tracer.  This builder replays the corrected native scene,
removes the non-optical ``workspace`` helper exactly as the V2 repair did, and
derives S0/DoLP/AoLP from the same four native analyzer images using PolarAPP's
official grayscale convention.  The existing dataset is never overwritten.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import types
from pathlib import Path

import lmdb
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT.parent
RLBENCH_ROOT = PROJECT_ROOT / "rlbench_custom_render/RLBench"
for path in (SCRIPT_DIR, RLBENCH_ROOT, RLBENCH_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if "rlbench" not in sys.modules:
    package = types.ModuleType("rlbench")
    package.__path__ = [str(RLBENCH_ROOT / "rlbench")]
    sys.modules["rlbench"] = package

from episode_snapshot_archive import EpisodeSnapshotArchive  # noqa: E402
from rlbench.native_config import NativePolarizationConfig  # noqa: E402
from rlbench.native_renderer import NativePolarizationRenderer  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / (
    "robot_data/rlbench/lerobot_point_lmdb/"
    "hybridvla_10tasks_train_keysteps_polar_rlbench9_v2"
)
DEFAULT_RAW = RLBENCH_ROOT / "output/ten_tasks_polar_train_20260921"
SIDECAR = "tasknet_frontview_native_stokes"
TASKNET_LUMA = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)


def tasknet_channels(render: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    analyzers = np.stack(
        [np.asarray(render[name], dtype=np.float32) for name in ("I0", "I45", "I90", "I135")]
    )
    rendered_s0 = np.asarray(render["S0"], dtype=np.float32)
    tolerance = 3e-6 * max(1.0, float(np.max(np.abs(rendered_s0))))
    if not (
        np.allclose(analyzers[0] + analyzers[2], rendered_s0, atol=tolerance, rtol=0.0)
        and np.allclose(analyzers[1] + analyzers[3], rendered_s0, atol=tolerance, rtol=0.0)
    ):
        raise AssertionError("native analyzer images and rendered S0 are inconsistent")
    valid = np.asarray(render["valid_mask"], dtype=bool)
    finite = np.isfinite(analyzers).all(axis=(0, 3)) & valid
    selected = analyzers[:, finite]
    if selected.size == 0:
        raise ValueError("native render has no finite valid analyzer samples")
    scale = max(float(np.percentile(selected, 99.5)), 1e-6)
    analyzers = np.clip(np.nan_to_num(analyzers / scale), 0.0, 1.0)
    gray = np.einsum("nhwc,c->nhw", analyzers, TASKNET_LUMA, optimize=True)
    i0, i45, i90, i135 = gray
    s0 = (i0 + i45 + i90 + i135) * 0.5
    s1 = i0 - i90
    s2 = i45 - i135
    linear = np.sqrt(s1 * s1 + s2 * s2 + 1e-5)
    dolp = np.clip(linear / (s0 + 1e-5), 0.0, 1.0)
    aolp = 0.5 * np.arctan2(s2 + 1e-5, s1 + 1e-5)
    angle_valid = finite & np.isfinite(aolp)
    cos2 = np.where(angle_valid, np.cos(2.0 * aolp), 0.0)
    sin2 = np.where(angle_valid, np.sin(2.0 * aolp), 0.0)
    s0 = np.where(finite, s0, 0.0)
    dolp = np.where(finite, dolp, 0.0)

    return {
        "S0": s0.astype(np.float16),
        "DoLP": dolp.astype(np.float16),
        "cos2AoLP": cos2.astype(np.float16),
        "sin2AoLP": sin2.astype(np.float16),
        "valid_mask": finite.astype(np.uint8),
        "AoLP_valid_mask": angle_valid.astype(np.uint8),
    }, scale


def encode(channels: dict[str, np.ndarray], scale: float, calibration: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        **channels,
        analyzer_normalization_scale=np.asarray(scale, dtype=np.float32),
        K=np.asarray(calibration["K"], dtype=np.float32),
        T_camera_from_world=np.asarray(calibration["T_camera_from_world"], dtype=np.float32),
        intensity_source=np.asarray("same_native_stokes_render"),
        coordinate_frame=np.asarray("canonical"),
    )
    return output.getvalue()


def read_calibration(txn: lmdb.Transaction, key: bytes) -> dict[str, np.ndarray]:
    payload = txn.get(key)
    if payload is None:
        raise KeyError(f"missing calibration key {key.decode()}")
    with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as record:
        return {
            "K": np.asarray(record["K"], dtype=np.float32),
            "T_camera_from_world": np.asarray(record["T_camera_from_world"], dtype=np.float32),
        }


def audit(path: Path, expected: int) -> dict[str, int | float | bool]:
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=1)
    maximum_phase_error = 0.0
    try:
        entries = int(env.stat()["entries"])
        if entries != expected:
            raise ValueError(f"sidecar has {entries} entries, expected {expected}")
        with env.begin(buffers=True) as txn:
            for position, (key, payload) in enumerate(txn.cursor(), 1):
                with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as record:
                    required = ("S0", "DoLP", "cos2AoLP", "sin2AoLP", "valid_mask", "K", "T_camera_from_world")
                    missing = [name for name in required if name not in record]
                    if missing:
                        raise KeyError(f"{bytes(key)!r}: missing {missing}")
                    s0 = np.asarray(record["S0"], dtype=np.float32)
                    dolp = np.asarray(record["DoLP"], dtype=np.float32)
                    cos2 = np.asarray(record["cos2AoLP"], dtype=np.float32)
                    sin2 = np.asarray(record["sin2AoLP"], dtype=np.float32)
                    valid = np.asarray(record["valid_mask"], dtype=bool)
                if not all(value.shape == (256, 256) for value in (s0, dolp, cos2, sin2, valid)):
                    raise ValueError(f"{bytes(key)!r}: invalid image shape")
                if not all(np.isfinite(value).all() for value in (s0, dolp, cos2, sin2)):
                    raise ValueError(f"{bytes(key)!r}: non-finite TaskNet channels")
                if np.any((dolp < 0) | (dolp > 1.001)):
                    raise ValueError(f"{bytes(key)!r}: invalid DoLP")
                polarized = valid & (dolp > 2e-3)
                if polarized.any():
                    error = float(np.max(np.abs(cos2[polarized] ** 2 + sin2[polarized] ** 2 - 1.0)))
                    maximum_phase_error = max(maximum_phase_error, error)
                if position % 500 == 0 or position == expected:
                    print(f"audited {position}/{expected} frames", flush=True)
    finally:
        env.close()
    return {"frames": expected, "all_finite": True, "maximum_phase_norm_error": maximum_phase_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--spp", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--map-size-gb", type=int, default=12)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root, raw = args.dataset.resolve(), args.raw.resolve()
    if args.limit_frames is not None and args.output is None:
        raise ValueError("--limit-frames requires an explicit --output test destination")
    final = args.output.resolve() if args.output else root / SIDECAR
    stage = root / f"{SIDECAR}.building"
    if args.output is not None:
        stage = final.with_name(final.name + ".building")
    manifest = root / f"meta/{SIDECAR}.json" if args.output is None else None
    if final.exists() or (manifest is not None and manifest.exists()):
        raise FileExistsError(f"refusing to replace completed sidecar: {final}")
    stage.mkdir(exist_ok=True)
    records = [json.loads(line) for line in (root / "frame_corruption_stats.jsonl").read_text().splitlines()]
    if args.limit_frames is not None:
        records = records[: args.limit_frames]
    expected = len(records)
    proxy = lmdb.open(str(root / "sfp_frontview_rgb_luminance_proxy"), readonly=True, lock=False, readahead=False)
    output = lmdb.open(str(stage), map_size=args.map_size_gb * 1024**3)
    current_task = None
    renderer = None
    rendered = skipped = 0
    try:
        with proxy.begin(buffers=True) as calibration_txn:
            for position, record in enumerate(records, 1):
                episode, frame = int(record["episode_index"]), int(record["frame_index"])
                task = str(record["task"])
                key = f"{episode}-{frame}".encode("ascii")
                with output.begin() as check_txn:
                    if check_txn.get(key) is not None:
                        skipped += 1
                        continue
                raw_episode = raw / task / f"episode_{episode % 100:06d}"
                if task != current_task:
                    if renderer is not None:
                        renderer.clear()
                    materials = json.loads((raw_episode / "materials.json").read_text())
                    renderer = NativePolarizationRenderer(NativePolarizationConfig(
                        spp=args.spp, max_depth=args.max_depth, device=args.device,
                        lighting="reference", geometry_source="rlbench", material_overrides=materials,
                    ))
                    current_task = task
                archive = EpisodeSnapshotArchive(raw_episode / "snapshots")
                meshes, lights, cameras, _ = archive.load_frame(frame)
                helpers = [mesh for mesh in meshes if mesh["name"] == "workspace"]
                if len(helpers) != 1:
                    raise ValueError(f"{raw_episode}: expected exactly one workspace helper")
                meshes = [mesh for mesh in meshes if mesh["name"] != "workspace"]
                summary = json.loads((raw_episode / "frames_spp512/render_summary.json").read_text())
                seed = int(summary["seed"] + frame) % 2**32
                render = renderer.render(meshes, lights, cameras["front"], seed=seed, geometry=False)
                channels, scale = tasknet_channels(render)
                payload = encode(channels, scale, read_calibration(calibration_txn, key))
                with output.begin(write=True) as txn:
                    txn.put(key, payload, overwrite=False)
                rendered += 1
                if position % 50 == 0 or position == expected:
                    print(f"rendered {position}/{expected} frames (new={rendered}, resumed={skipped})", flush=True)
    finally:
        if renderer is not None:
            renderer.clear()
        output.sync()
        output.close()
        proxy.close()
        gc.collect()

    report = audit(stage, expected)
    stage.rename(final)
    metadata = {
        "complete": True,
        "sidecar_dirname": SIDECAR,
        "frames": expected,
        "image_shape": [256, 256],
        "renderer": "native-cuda-spectral-mueller-path",
        "spp": args.spp,
        "max_depth": args.max_depth,
        "workspace_helper_excluded": True,
        "tasknet_preprocessing": "shared analyzer p99.5 normalization, clip [0,1], PolarAPP RGB-to-gray and Stokes equations",
        "intensity_source": "S0 and DoLP/AoLP from the same native analyzer render",
        "serialized_fields": ["S0", "DoLP", "cos2AoLP", "sin2AoLP", "valid_mask", "AoLP_valid_mask", "K", "T_camera_from_world"],
        "audit": report,
    }
    if manifest is not None:
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": str(final), **report}, indent=2))


if __name__ == "__main__":
    main()
