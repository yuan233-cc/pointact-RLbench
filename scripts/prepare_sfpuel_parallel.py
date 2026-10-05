#!/usr/bin/env python3
"""Resume SfPUEL conversion without overwriting records, using CPU workers.

The old directory is read-only. Valid records are hard-linked into a new
directory, and missing or incomplete records are computed independently.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointact.data.polar_normal_sources import build_cga_record, stack_analyzers  # noqa: E402
from scripts.prepare_polar_normal_sources import (  # noqa: E402
    read_rgb,
    sfpuel_group,
    write_manifest,
)


def record_name(index: int, source: Path) -> str:
    return f"{index:06d}_{source.stem}.npz"


def valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def convert_one(task: tuple[int, str, str, str, int, float]) -> int:
    index, source_name, input_name, output_name, stride, refractive_index = task
    source = Path(source_name)
    input_root = Path(input_name)
    destination = Path(output_name) / record_name(index, source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    name = source.name
    paths = [input_root / f"pol{angle}" / name for angle in ("000", "045", "090", "135")]
    normal_path = input_root / "normal" / name
    mask_path = input_root / "mask" / name
    missing = [path for path in (*paths, normal_path, mask_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete SfPUEL sample {name}: {missing}")
    analyzer_images = [read_rgb(path) for path in paths]
    normal_image = read_rgb(normal_path)
    if not np.issubdtype(normal_image.dtype, np.integer):
        raise TypeError(f"Expected integer-encoded SfPUEL normal image: {normal_path}")
    normal = normal_image.astype(np.float32) / float(np.iinfo(normal_image.dtype).max)
    normal = normal * 2.0 - 1.0
    normal[..., 1:] *= -1.0
    mask_image = read_rgb(mask_path)
    mask = mask_image[..., 0] > 0 if mask_image.ndim == 3 else mask_image > 0
    if stride > 1:
        analyzer_images = [image[::stride, ::stride] for image in analyzer_images]
        normal = normal[::stride, ::stride]
        mask = mask[::stride, ::stride]
    analyzers = stack_analyzers(analyzer_images, normalization="dtype")
    record = build_cga_record(
        analyzers,
        normal,
        mask,
        refractive_index=refractive_index,
        normal_orientation="optical_axis",
    )
    payload = {key: np.asarray(value) for key, value in record.items()}
    payload["dataset_id"] = np.asarray("sfpuel")
    payload["source_id"] = np.asarray(str(source))
    temporary = destination.with_name(destination.name + f".pending.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    try:
        os.link(temporary, destination)  # atomic creation; never replace an existing sample
    finally:
        temporary.unlink()
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--spatial-stride", type=int, default=4)
    parser.add_argument("--refractive-index", type=float, default=1.5)
    parser.add_argument("--group-prefix-components", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.spatial_stride < 1 or args.group_prefix_components < 1:
        parser.error("workers, spatial stride, and group prefix components must be positive")
    if args.refractive_index <= 1.0:
        parser.error("refractive index must be greater than one")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard index must be in [0, num-shards)")
    sources = sorted((args.input_root / "pol000").glob("*.png"))
    if not sources:
        parser.error("no SfPUEL pol000 images found")
    expected = {record_name(index, source) for index, source in enumerate(sources)}
    selected = [(index, source) for index, source in enumerate(sources)
                if index % args.num_shards == args.shard_index]
    selected_names = {record_name(index, source) for index, source in selected}
    old_files = sorted(args.reuse_root.glob("*.npz"))
    unknown = [path.name for path in old_files if path.name not in expected]
    if unknown:
        raise RuntimeError(f"Unrecognized old records: {unknown[:3]}")
    args.output.mkdir(parents=True, exist_ok=False)
    reused = set()
    corrupt = []
    for old_path in old_files:
        if old_path.name not in selected_names:
            continue
        if valid_zip(old_path):
            os.link(old_path, args.output / old_path.name)
            reused.add(old_path.name)
        else:
            corrupt.append(old_path.name)
    print(
        f"SfPUEL shard={args.shard_index}/{args.num_shards} sources={len(selected)} "
        f"reused={len(reused)} corrupt_recomputed={len(corrupt)} workers={args.workers}",
        flush=True,
    )
    if corrupt:
        print(f"Corrupt old records retained in read-only source: {corrupt[:8]}", flush=True)
    tasks = [
        (index, str(source), str(args.input_root), str(args.output), args.spatial_stride,
         args.refractive_index)
        for index, source in selected
        if record_name(index, source) not in reused
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for completed, _ in enumerate(pool.map(convert_one, tasks, chunksize=8), start=1):
            if completed % 1000 == 0 or completed == len(tasks):
                print(f"SfPUEL newly converted={completed}/{len(tasks)}", flush=True)
    entries = [
        {
            "path": record_name(index, source),
            "group": sfpuel_group(source.stem, args.group_prefix_components),
            "id": source.stem,
            "dataset": "sfpuel",
        }
        for index, source in selected
    ]
    write_manifest(
        args.output,
        entries,
        {
            "adapter": "sfpuel",
            "source": str(args.input_root.resolve()),
            "coordinate_frame": "+x right, +y down, +z forward; normals face camera",
            "analyzer_order_degrees": [0, 45, 90, 135],
            "normal_decode": "official RGB image / dtype_max * 2 - 1; (x,y,z) to (x,-y,-z)",
            "normal_orientation": "optical axis because camera intrinsics are unavailable",
            "refractive_index": args.refractive_index,
            "group_prefix_components": args.group_prefix_components,
            "spatial_stride": args.spatial_stride,
            "records": len(entries),
            "total_source_records": len(sources),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "reused_valid_records": len(reused),
            "recomputed_corrupt_records": len(corrupt),
            "workers": args.workers,
        },
    )


if __name__ == "__main__":
    main()
