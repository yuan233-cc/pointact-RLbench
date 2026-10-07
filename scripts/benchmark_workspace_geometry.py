"""Bounded real optimizer probes: choose throughput, not VRAM occupancy alone."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from pointact.data.workspace_geometry_cache import CachedWorkspaceGeometryDataset
from pointact.data.workspace_geometry_dataset import collate_geometry
from pointact.model.vla_pointact.workspace_geometry import WorkspaceGeometryModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("backbone", "dataset-root", "cache-dir", "teacher-checkpoint", "concerto-checkpoint", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--dino-weights")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[384, 768, 1024, 1280, 1536])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    a = p.parse_args()
    root = Path(a.output_dir)
    root.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dataset = CachedWorkspaceGeometryDataset(a.cache_dir, a.backbone, a.teacher_checkpoint)
    model = WorkspaceGeometryModel(a.backbone, a.teacher_checkpoint, a.concerto_checkpoint, a.dino_weights).cuda()
    model.teacher.cpu()
    torch.cuda.empty_cache()
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, fused=True)
    run = None
    if a.report_to == "wandb":
        import wandb
        viewer = wandb.Api().viewer
        if "weihang-li" in (viewer.username, viewer.entity):
            raise RuntimeError("Prohibited W&B identity")
        run = wandb.init(entity=viewer.entity, project="pointact-workspace-geometry", name=root.name,
                         config=vars(a))
        print(f"WANDB_RUN_URL={run.url}", flush=True)
    results = []
    for batch_size in a.batch_sizes:
        # Recreate only the loader, not another multi-GB model or teacher.
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=a.workers, shuffle=True,
            collate_fn=collate_geometry, pin_memory=True, persistent_workers=a.workers > 0,
            **({"prefetch_factor": 2} if a.workers else {}))
        iterator = iter(loader)
        torch.cuda.reset_peak_memory_stats()
        elapsed, waits, computes, samples = [], [], [], []
        output = batch = loss = cpu = None
        record = dict(batch_size=batch_size, workers=a.workers)
        try:
            for step in range(a.steps + 2):
                tick = time.monotonic()
                try:
                    cpu = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    cpu = next(iterator)
                ready = time.monotonic()
                batch = {k: v.cuda(non_blocking=True) for k, v in cpu.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(batch)
                loss = output["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite benchmark loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                torch.cuda.synchronize()
                end = time.monotonic()
                if step >= 2:
                    elapsed.append(end-tick)
                    waits.append(ready-tick)
                    computes.append(end-ready)
                    samples.append(len(cpu["npoints_in_batch"]))
                del output, batch, loss, cpu
                output = batch = None
            used = torch.cuda.max_memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            record.update(status="ok", peak_memory_gb=used/2**30,
                reserved_memory_gb=torch.cuda.max_memory_reserved()/2**30,
                headroom_percent=100*(1-torch.cuda.max_memory_reserved()/total), samples_per_second=sum(samples)/sum(elapsed),
                median_step_seconds=float(np.median(elapsed)), data_wait_fraction=sum(waits)/sum(elapsed),
                median_compute_seconds=float(np.median(computes)))
        except torch.cuda.OutOfMemoryError:
            record.update(status="oom")
            output = batch = loss = cpu = None
            optimizer.zero_grad(set_to_none=True)
        finally:
            # Bound each loader lifetime, including its host-memory prefetch queue.
            if hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
            del iterator, loader
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        results.append(record)
        print("BENCHMARK="+json.dumps(record), flush=True)
        (root / "benchmark.json").write_text(json.dumps(results, indent=2))
        if run:
            run.log(record)
        if record["status"] == "oom":
            break
    good = [r for r in results if r["status"] == "ok" and r["headroom_percent"] >= 8]
    if not good:
        raise RuntimeError("No safe benchmark candidate")
    best = max(good, key=lambda r: r["samples_per_second"])
    (root / "selected.json").write_text(json.dumps(best, indent=2))
    print("SELECTED="+json.dumps(best), flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
