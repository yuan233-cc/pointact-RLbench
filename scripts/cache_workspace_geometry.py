"""Cache calibrated observations and frozen polar outputs once, without GT."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
from pointact.data.workspace_geometry_dataset import WorkspaceGeometryDataset, collate_geometry
from pointact.data.workspace_geometry_cache import GeometryColumnWriter
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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="Testing only; do not use for production")
    a = p.parse_args()
    if a.cache_dir.exists():
        raise FileExistsError(a.cache_dir)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    datasets = [WorkspaceGeometryDataset(a.dataset_root, a.backbone, split=s,
                max_points=1_000_000, cga_records=a.cga_records) for s in ("train", "val")]
    keys = [k for d in datasets for k in d.keys]
    source = ConcatDataset(datasets)
    if a.limit:
        source = torch.utils.data.Subset(source, range(a.limit))
        keys = keys[:a.limit]
    loader = DataLoader(source, batch_size=a.batch_size, num_workers=a.workers,
        collate_fn=collate_geometry, pin_memory=True,
        **({"prefetch_factor": 2} if a.workers else {}))
    model = WorkspaceGeometryModel(a.backbone, a.teacher_checkpoint,
        a.concerto_checkpoint, a.dino_weights).cuda().eval()
    writer = GeometryColumnWriter(a.cache_dir, keys, a.backbone, a.teacher_checkpoint)
    started = time.monotonic()
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
    print(f"CACHE_COMPLETE={a.cache_dir} frames={len(keys)} seconds={time.monotonic()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
