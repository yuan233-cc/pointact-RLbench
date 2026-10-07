"""Normal-filtered PTv3 geometry training with two weighted workspace losses."""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from pointact.data.workspace_geometry_dataset import WorkspaceGeometryDataset, collate_geometry
from pointact.model.vla_pointact.workspace_geometry import WorkspaceGeometryModel
from pointact.data.observed_workspace_mask import operation_workspace
from pointact.data.workspace_geometry_cache import CachedWorkspaceGeometryDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("tasknet", "cga_dinov3"), required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cga-records", help="Verified cached native-CGA inputs; GT fields are never read")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--concerto-checkpoint", required=True)
    parser.add_argument("--dino-weights")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--supervision-mode", choices=("weighted_workspace",), default="weighted_workspace")
    parser.add_argument("--point-fit-scale-m", type=float, default=0.05)
    parser.add_argument("--inconsistent-point-weight", type=float, default=0.1)
    parser.add_argument("--probe-seconds", type=float, default=0)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--validate-every-epochs", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    parser.add_argument("--cache-dir", type=Path, help="Verified FP32 frozen-teacher column cache")
    parser.add_argument("--cache-mmap", action="store_true", help="Default loads split columns into RAM")
    parser.add_argument("--validation-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.validate_every_epochs < 1:
        parser.error("--validate-every-epochs must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(workspace_bounds=operation_workspace().tolist(), workspace_mask_version="observed_neighbors_v1",
                  hole_neighbor_radius_pixels=12, hole_min_support=4)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    if args.cache_dir:
        train = CachedWorkspaceGeometryDataset(args.cache_dir, args.backbone, args.teacher_checkpoint,
                                               preload=not args.cache_mmap)
        val = CachedWorkspaceGeometryDataset(args.cache_dir, args.backbone, args.teacher_checkpoint,
                                             split="val", preload=not args.cache_mmap)
    else:
        train = WorkspaceGeometryDataset(args.dataset_root, args.backbone, cga_records=args.cga_records)
        val = WorkspaceGeometryDataset(args.dataset_root, args.backbone, split="val", cga_records=args.cga_records)
    def loader(dataset, shuffle):
        kwargs = dict(batch_size=args.batch_size if shuffle else args.validation_batch_size,
            shuffle=shuffle, num_workers=args.workers,
            pin_memory=True, collate_fn=collate_geometry, persistent_workers=args.workers > 0)
        if args.workers:
            kwargs["prefetch_factor"] = 2
        return DataLoader(dataset, **kwargs)
    train_loader, val_loader = loader(train, True), loader(val, False)
    model = WorkspaceGeometryModel(args.backbone, args.teacher_checkpoint,
        args.concerto_checkpoint, args.dino_weights, supervision_mode=args.supervision_mode,
        point_fit_scale_m=args.point_fit_scale_m,
        inconsistent_point_weight=args.inconsistent_point_weight).cuda()
    # Cached tensors are fingerprinted above; the frozen teacher remains on CPU.
    # No trainable adapter/Concerto/decoder tensor is cached or detached.
    if args.cache_dir:
        model.teacher.cpu()
        torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=.01, fused=True)
    max_steps = args.max_steps or args.epochs * len(train_loader)
    warmup = max(1, int(.03 * max_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min(1., (step+1)/warmup) * .5 * (1 + math.cos(math.pi * max(0, step-warmup) / max(1, max_steps-warmup))))
    step, best = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        # A resumed run has a fresh output directory and validation protocol;
        # select its best checkpoint locally rather than inheriting a stale score.
        best = float("inf")
    run = None
    if args.report_to == "wandb":
        import wandb
        viewer = wandb.Api().viewer
        username, entity = viewer.username, viewer.entity
        if "weihang-li" in (username, entity):
            raise RuntimeError("Prohibited W&B identity")
        requested = os.environ.get("WANDB_ENTITY", entity)
        if requested != entity:
            raise RuntimeError("W&B entity does not match authenticated viewer")
        run = wandb.init(entity=entity, project=os.environ.get("WANDB_PROJECT", "pointact-workspace-geometry"),
            name=args.output_dir.name, config=config)
        print(f"WANDB_RUN_URL={run.url}", flush=True)
    def save(name):
        state = {k: v for k, v in model.state_dict().items() if not k.startswith("teacher.")}
        destination = args.output_dir / name
        temporary = destination.with_suffix(".partial")
        torch.save(dict(model=state, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        step=step, best=best, config=config, max_steps=max_steps), temporary)
        temporary.replace(destination)
    def to_device(batch):
        return {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    started = time.monotonic()
    step_times = []
    metrics = args.output_dir / "metrics.jsonl"
    phase = args.output_dir / "phase.json"
    def set_phase(name):
        temporary = phase.with_suffix(".partial")
        temporary.write_text(json.dumps(dict(phase=name, updated=time.time(), step=step)))
        temporary.replace(phase)
    print(json.dumps(dict(train_frames=len(train), val_frames=len(val), max_steps=max_steps,
                         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))), flush=True)
    model.train()
    epoch = step // len(train_loader)
    while step < max_steps:
        set_phase("train")
        iterator = iter(train_loader)
        previous = time.monotonic()
        for batch in iterator:
            tick = time.monotonic()
            data_wait = tick - previous
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(to_device(batch))
            loss = output["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite geometry loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            step += 1
            step_times.append(time.monotonic() - tick)
            if step % 5 == 0:
                record = {k: float(output[k].detach()) for k in
                    ("loss", "normal_consistency_loss", "normal_error_raw", "point_fit_loss", "point_fit_mae_m")}
                record.update(step=step, lr=scheduler.get_last_lr()[0], elapsed=time.monotonic()-started,
                    max_steps=max_steps, peak_memory_gb=torch.cuda.max_memory_allocated()/2**30,
                    anchors=float((output["anchor_weights"] > 0).sum())/len(output["input_point_counts"]),
                    input_points_mean=float(output["input_point_counts"].float().mean()),
                    input_points_min=int(output["input_point_counts"].min()),
                    skipped_samples=int(output["skipped_samples"]))
                record.update(data_wait_seconds=data_wait, compute_step_seconds=step_times[-1],
                              samples_per_second=len(batch["npoints_in_batch"])/max(1e-6, data_wait+step_times[-1]))
                with metrics.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if run:
                    run.log(record, step=step)
            if step % args.save_steps == 0:
                set_phase("checkpoint")
                save("last.pt")
                set_phase("train")
            if args.probe_seconds and time.monotonic()-started >= args.probe_seconds:
                set_phase("checkpoint")
                save("last.pt")
                # Wall time includes loader stalls, unlike GPU-only timing.
                steady = step_times[5:] or step_times
                seconds_per_step = (time.monotonic()-started) / max(1, step - (state["step"] if args.resume else 0))
                estimate = dict(step=step, max_steps=max_steps, seconds_per_step=seconds_per_step,
                    gpu_step_median=float(np.median(steady)), remaining_seconds=(max_steps-step)*seconds_per_step*1.05,
                    suggested_wall_seconds=math.ceil((max_steps-step)*seconds_per_step*1.05 + 2400))
                (args.output_dir / "timing_estimate.json").write_text(json.dumps(estimate, indent=2))
                print("PROBE_COMPLETE=" + json.dumps(estimate), flush=True)
                if run:
                    run.finish()
                set_phase("complete")
                return
            if step >= max_steps:
                break
            previous = time.monotonic()
        epoch += 1
        if epoch % args.validate_every_epochs and step < max_steps:
            continue
        model.eval()
        set_phase("validation")
        validation = []
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if index >= 8:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    validation.append(float(model(to_device(batch))["loss"]))
        value = float(np.mean(validation))
        if run:
            run.log({"val/loss": value}, step=step)
        if value < best:
            best = value
            set_phase("checkpoint")
            save("best.pt")
        set_phase("checkpoint")
        save("last.pt")
        model.train()
    set_phase("complete")
    print(f"TRAINING_COMPLETE step={step}", flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # DataLoader/W&B teardown can otherwise hang after a fatal exception,
        # leaving the SSH allocation idle. The host supervisor cancels the
        # dedicated allocation when this foreground process exits, including
        # any remaining worker processes. Never write a checkpoint on failure.
        import sys
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
