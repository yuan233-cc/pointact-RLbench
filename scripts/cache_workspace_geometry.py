"""Cache calibrated observations and frozen polar outputs once, without GT."""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from pointact.data.workspace_geometry_dataset import WorkspaceGeometryDataset, collate_geometry
from pointact.data.workspace_geometry_cache import GeometryColumnWriter, file_sha256
from pointact.model.vla_pointact.workspace_geometry import WorkspaceGeometryModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=("tasknet", "cga_dinov3"), required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--cga-records")
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--concerto-checkpoint", required=True)
    p.add_argument("--dino-weights")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--scratch-dir", type=Path, help="Build on Job-local SSD, then checksum-publish cache-dir")
    p.add_argument("--reuse-prefix-cache", type=Path, help="Owned interrupted column cache; verified before reuse")
    p.add_argument("--reuse-prefix-rows", type=int, default=0, help="Explicit completed-row count from the interrupted log")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="Testing only; do not use for production")
    a = p.parse_args()
    if a.cache_dir.exists():
        raise FileExistsError(a.cache_dir)
    if bool(a.reuse_prefix_cache) != bool(a.reuse_prefix_rows):
        p.error("Prefix path and confirmed row count must be supplied together")
    if a.limit and a.reuse_prefix_cache:
        p.error("Testing --limit cannot be combined with prefix recovery")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    datasets = [WorkspaceGeometryDataset(a.dataset_root, a.backbone, split=s,
                max_points=1_000_000, cga_records=a.cga_records) for s in ("train", "val")]
    keys = [k for d in datasets for k in d.keys]
    # One dataset/environment per worker. ConcatDataset(train, val) would open
    # the same LMDB twice when a worker crosses the split boundary.
    source = datasets[0]
    source.keys = keys
    if a.limit:
        source = torch.utils.data.Subset(source, range(a.limit))
        keys = keys[:a.limit]
    model = WorkspaceGeometryModel(a.backbone, a.teacher_checkpoint,
        a.concerto_checkpoint, a.dino_weights).cuda().eval()
    build_root = a.scratch_dir or a.cache_dir
    writer = GeometryColumnWriter(build_root, keys, a.backbone, a.teacher_checkpoint)
    started = time.monotonic()
    if a.reuse_prefix_cache:
        rows = a.reuse_prefix_rows
        if not 0 < rows < len(keys) or rows % a.batch_size:
            raise ValueError("Confirmed prefix must contain complete teacher batches")
        columns = {f.stem: np.load(f, mmap_mode="r", allow_pickle=False)
                   for f in a.reuse_prefix_cache.glob("*.npy")}
        if not columns or any(len(v) != len(keys) for v in columns.values()):
            raise ValueError("Prefix columns/frame count mismatch")
        if np.any(columns["point_count"][:rows] <= 0):
            raise ValueError("Prefix contains unwritten point counts")
        for start in (0, rows - a.batch_size):
            samples = [source[i] for i in range(start, start + a.batch_size)]
            cpu = collate_geometry(samples)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = model.encode_teacher({k: v.cuda(non_blocking=True) for k, v in cpu.items()})
            for name, value in encoded.items():
                torch.testing.assert_close(value.cpu(), torch.from_numpy(np.array(columns[name][start:start+a.batch_size])),
                                           rtol=1e-4, atol=1e-4)
            for offset, sample in enumerate(samples):
                row = start + offset
                n = int(columns["point_count"][row])
                if n != len(sample["points"]):
                    raise ValueError("Prefix observation count mismatch")
                for name, value in sample.items():
                    if name in ("points", "point_pixel_indices", "cga_observation", "cga_prior", "rgb"):
                        continue
                    np.testing.assert_equal(columns[name][row], value.numpy())
                old = np.column_stack((columns["point_pixel_indices"][row, :n], columns["points"][row, :n]))
                new = np.column_stack((sample["point_pixel_indices"].numpy(), sample["points"].numpy()))
                np.testing.assert_equal(old[np.lexsort(old.T[::-1])], new[np.lexsort(new.T[::-1])])
            print(f"PREFIX_VERIFIED frames={start}:{start+a.batch_size}", flush=True)
        n = int(columns["point_count"][0])
        sample = {k: torch.from_numpy(np.array(v[0, :n] if k in ("points", "point_pixel_indices") else v[0]))
                  for k, v in columns.items() if k != "point_count"}
        writer.append(sample)
        if set(writer.columns) != set(columns):
            raise ValueError("Prefix field mismatch")
        for name, target in writer.columns.items():
            if target.shape != columns[name].shape or target.dtype != columns[name].dtype:
                raise ValueError(f"Prefix shape/dtype mismatch: {name}")
            target[:rows] = columns[name][:rows]
        writer.rows = rows
        source = torch.utils.data.Subset(source, range(rows, len(keys)))
        print(f"PREFIX_REUSED rows={rows}; old cache preserved", flush=True)
        del columns, encoded, samples, cpu
    # Prefix verification may have opened a parent LMDB. Do not inherit it
    # into forked workers; each worker opens the single dataset lazily.
    base = source.dataset if isinstance(source, torch.utils.data.Subset) else source
    for env in base.envs.values():
        env.close()
    base.envs.clear()
    loader = DataLoader(source, batch_size=a.batch_size, num_workers=a.workers,
        collate_fn=collate_geometry, pin_memory=True,
        **({"prefetch_factor": 2} if a.workers else {}))
    for cpu in loader:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.encode_teacher({k: v.cuda(non_blocking=True) for k, v in cpu.items()})
        encoded = {k: v.cpu() for k, v in encoded.items()}
        cursor = 0
        for row, count in enumerate(cpu["npoints_in_batch"].tolist()):
            sample = {k: (v[cursor:cursor+count] if k in ("points", "point_pixel_indices") else v[row])
                for k, v in cpu.items() if k not in ("npoints_in_batch", "cga_observation", "cga_prior", "rgb")}
            sample.update({k: v[row] for k, v in encoded.items()})
            writer.append(sample)
            cursor += count
        if writer.rows % (a.batch_size * 4) == 0:
            print(json.dumps(dict(cache_frames=writer.rows, total=len(keys),
                  elapsed=time.monotonic()-started)), flush=True)
    writer.finish()
    if a.scratch_dir:
        partial = a.cache_dir.with_name(a.cache_dir.name + ".partial")
        partial.mkdir(exist_ok=False, parents=True)
        for source_file in sorted(build_root.iterdir()):
            destination = partial / source_file.name
            shutil.copyfile(source_file, destination)
            if source_file.stat().st_size != destination.stat().st_size or file_sha256(source_file) != file_sha256(destination):
                raise ValueError(f"Cache publication checksum mismatch: {source_file.name}")
            print(f"CACHE_PUBLISHED_COLUMN={source_file.name}", flush=True)
        if a.cache_dir.exists():
            raise FileExistsError(a.cache_dir)
        partial.rename(a.cache_dir)
    print(f"CACHE_COMPLETE={a.cache_dir} frames={len(keys)} seconds={time.monotonic()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
