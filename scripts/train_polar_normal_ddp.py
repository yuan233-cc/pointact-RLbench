#!/usr/bin/env python3
"""Two-GPU DDP pretraining for the mixed polarization-normal datasets."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Sampler
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_polar_normal import (  # noqa: E402
    aggregate_metrics,
    assert_disjoint_groups,
    build_datasets,
    build_model,
    collate_training_fields,
    masked_cosine_normal_loss,
    normal_metrics,
    save_checkpoint,
)


class BalancedDistributedSampler(Sampler[int]):
    """Equal-length, equal-dataset-weight random shards for all DDP ranks."""

    def __init__(self, datasets: list, rank: int, world_size: int, seed: int):
        lengths = [len(dataset) for dataset in datasets]
        if any(length <= 0 for length in lengths):
            raise ValueError(f"empty training dataset: {lengths}")
        self.weights = torch.cat(
            [torch.full((length,), 1.0 / length, dtype=torch.double) for length in lengths]
        )
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(sum(lengths) / world_size)
        self.total_size = self.num_samples * world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        )
        return iter(indices[self.rank : self.total_size : self.world_size].tolist())

    def __len__(self) -> int:
        return self.num_samples


class ExactShardSampler(Sampler[int]):
    """Partition validation once, without DistributedSampler padding duplicates."""

    def __init__(self, length: int, rank: int, world_size: int):
        self.length = length
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return len(range(self.rank, self.length, self.world_size))


@torch.no_grad()
def validate_distributed(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    # angle_sum, within_11_25, within_22_5, valid_count, loss_sum, batches
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    for batch in loader:
        observation = batch["polar_observation"].to(device, non_blocking=True)
        prior = batch["physical_prior"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        target = batch["normal_gt"].to(device, non_blocking=True)
        mask = batch["normal_valid_mask"].to(device, non_blocking=True)
        prediction = model(observation, prior, rgb)["normal"]
        metrics = normal_metrics(prediction, target, mask)
        totals[:4] += torch.stack(
            [metrics[key] for key in ("angle_sum", "within_11_25", "within_22_5", "valid_count")]
        ).to(torch.float64)
        totals[4] += masked_cosine_normal_loss(prediction, target, mask).to(torch.float64)
        totals[5] += 1
    dist.all_reduce(totals)
    values = totals.tolist()
    result = aggregate_metrics(dict(zip(("angle_sum", "within_11_25", "within_22_5", "valid_count"), values[:4])))
    result["loss"] = values[4] / max(values[5], 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="resume from a completed epoch checkpoint")
    parser.add_argument("--smoke-optimizer-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, help="override physical batch per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    if args.smoke_optimizer_steps < 0:
        parser.error("--smoke-optimizer-steps must be nonnegative")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"this launch requires exactly two ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    try:
        config = yaml.safe_load(args.config.read_text())
        train_cfg = config["training"]
        seed = int(train_cfg.get("seed", 0))
        torch.manual_seed(seed + rank)
        named_train = build_datasets(config, "train")
        named_val = build_datasets(config, "val")
        assert_disjoint_groups(
            *(dataset for _, dataset in named_train), *(dataset for _, dataset in named_val)
        )
        train_datasets = [dataset for _, dataset in named_train]
        train_dataset = ConcatDataset(train_datasets)
        sampler = BalancedDistributedSampler(train_datasets, rank, world_size, seed)
        batch_size = int(train_cfg.get("batch_size", 2) if args.batch_size is None else args.batch_size)
        accumulation = int(train_cfg.get("gradient_accumulation_steps", 2) if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps)
        if batch_size <= 0 or accumulation <= 0:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        workers = int(train_cfg.get("num_workers", 4) if args.num_workers is None else args.num_workers)
        if workers < 0:
            raise ValueError("num_workers must be nonnegative")
        from functools import partial

        collate = partial(collate_training_fields, use_rgb=config["model"].get("use_dino", True))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=workers, pin_memory=True, collate_fn=collate,
        )
        val_loaders = {
            name: DataLoader(
                dataset, batch_size=batch_size,
                sampler=ExactShardSampler(len(dataset), rank, world_size),
                num_workers=workers, pin_memory=True, collate_fn=collate,
            )
            for name, dataset in named_val
        }

        base_model = build_model(config).to(device)
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(train_cfg.get("learning_rate", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(train_cfg.get("step_size", 10)),
            gamma=float(train_cfg.get("gamma", 0.5)),
        )
        start_epoch, best_mae = 0, float("inf")
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            missing, unexpected = model.module.load_state_dict(checkpoint["model"], strict=False)
            missing = [name for name in missing if not name.startswith("dino.")]
            if missing or unexpected:
                raise RuntimeError(f"Resume mismatch: missing={missing}, unexpected={unexpected}")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_mae = float(checkpoint["best_mae"])

        output_dir = Path(train_cfg["output_dir"])
        if not args.smoke_optimizer_steps:
            if rank == 0:
                if not output_dir.is_dir() or not os.access(output_dir, os.W_OK):
                    raise RuntimeError(f"verified run directory is not writable: {output_dir}")
                if (output_dir / "metrics.jsonl").exists() or (output_dir / "last.pt").exists():
                    raise RuntimeError(f"run directory already contains training state: {output_dir}")
            dist.barrier()
        if rank == 0:
            print(json.dumps({"event": "initialized", "world_size": world_size,
                "train_samples": len(train_dataset),
                "train_steps_per_rank": len(train_loader),
                "effective_batch_size": batch_size * accumulation * world_size,
                "start_epoch": start_epoch,
                "resume_checkpoint": str(args.resume) if args.resume else None,
                "output_dir": str(output_dir) if not args.smoke_optimizer_steps else None}), flush=True)

        wandb_run = None
        tracking = config.get("tracking", {})
        if rank == 0 and tracking.get("backend") == "wandb" and not args.smoke_optimizer_steps:
            import wandb

            entity = str(tracking["entity"])
            if entity == "weihang-li":
                raise RuntimeError("prohibited W&B entity")
            if os.environ.get("WANDB_ENTITY") != entity:
                raise RuntimeError("WANDB_ENTITY does not match the verified config entity")
            viewer = wandb.Api().viewer
            if viewer.username == "weihang-li" or viewer.entity == "weihang-li":
                raise RuntimeError("prohibited W&B viewer identity")
            if viewer.entity != entity:
                raise RuntimeError(f"W&B viewer entity mismatch: {viewer.entity!r}")
            wandb_run = wandb.init(
                project=str(tracking["project"]), entity=entity,
                name=str(tracking["run_name"]), dir=os.environ["WANDB_DIR"],
                config={"training": train_cfg, "model": config["model"],
                        "datasets": [name for name, _ in named_train],
                        "resume_checkpoint": str(args.resume) if args.resume else None},
                resume="never",
            )
            if wandb_run is None:
                raise RuntimeError("W&B initialization did not return a run")
            print(json.dumps({"event": "wandb_initialized", "entity": entity,
                "project": tracking["project"], "run_url": wandb_run.url}), flush=True)

        optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, int(train_cfg.get("epochs", 30))):
            sampler.set_epoch(epoch)
            model.train()
            running = torch.zeros(2, dtype=torch.float64, device=device)
            window = torch.zeros(2, dtype=torch.float64, device=device)
            window_start = time.monotonic()
            optimizer_steps = 0
            for step, batch in enumerate(train_loader):
                observation = batch["polar_observation"].to(device, non_blocking=True)
                prior = batch["physical_prior"].to(device, non_blocking=True)
                rgb = batch["rgb"].to(device, non_blocking=True)
                target = batch["normal_gt"].to(device, non_blocking=True)
                mask = batch["normal_valid_mask"].to(device, non_blocking=True)
                end_group = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
                group_start = (step // accumulation) * accumulation
                group_steps = min(accumulation, len(train_loader) - group_start)
                with nullcontext() if end_group else model.no_sync():
                    prediction = model(observation, prior, rgb)["normal"]
                    loss = masked_cosine_normal_loss(prediction, target, mask)
                    (loss / group_steps).backward()
                detached = loss.detach().to(torch.float64)
                running[0] += detached
                running[1] += 1
                window[0] += detached
                window[1] += 1
                if end_group:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if optimizer_steps % 10 == 0 or (args.smoke_optimizer_steps and optimizer_steps == args.smoke_optimizer_steps):
                        dist.all_reduce(window)
                        if rank == 0:
                            progress = {"event": "train_progress", "epoch": epoch,
                                "optimizer_step": optimizer_steps,
                                "loss": float(window[0] / window[1].clamp_min(1)),
                                "elapsed_s": round(time.monotonic() - window_start, 2)}
                            print(json.dumps(progress), flush=True)
                            if wandb_run is not None:
                                wandb_run.log({"train/loss": progress["loss"],
                                    "train/epoch": epoch, "train/optimizer_step": optimizer_steps,
                                    "train/seconds_per_10_steps": progress["elapsed_s"]})
                        window.zero_()
                        window_start = time.monotonic()
                    if args.smoke_optimizer_steps and optimizer_steps >= args.smoke_optimizer_steps:
                        dist.barrier()
                        if rank == 0:
                            print(json.dumps({"event": "smoke_ok", "optimizer_steps": optimizer_steps,
                                "batch_size_per_gpu": batch_size,
                                "peak_memory_gib_gpu0": round(torch.cuda.max_memory_allocated() / 2**30, 2)}),
                                flush=True)
                        return

            dist.all_reduce(running)
            per_dataset = {
                name: validate_distributed(model.module, loader, device)
                for name, loader in val_loaders.items()
            }
            metrics = {
                f"val/{name}/{key}": value
                for name, values in per_dataset.items()
                for key, value in values.items()
            }
            for key in ("normal_mae", "within_11_25", "within_22_5", "loss"):
                metrics[f"{key}_macro"] = sum(values[key] for values in per_dataset.values()) / len(per_dataset)
            metrics["valid_pixels"] = sum(values["valid_pixels"] for values in per_dataset.values())
            metrics.update(epoch=epoch, train_loss=float(running[0] / running[1].clamp_min(1)),
                learning_rate=optimizer.param_groups[0]["lr"],
                effective_batch_size=batch_size * accumulation * world_size)
            scheduler.step()
            if rank == 0:
                with (output_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                print(json.dumps({"event": "epoch", **metrics}), flush=True)
                if wandb_run is not None:
                    wandb_run.log(metrics)
                is_best = metrics["normal_mae_macro"] < best_mae
                best_mae = min(best_mae, metrics["normal_mae_macro"])
                save_checkpoint(output_dir / "last.pt", model.module, optimizer, scheduler,
                    epoch, best_mae, config)
                if is_best:
                    save_checkpoint(output_dir / "best.pt", model.module, optimizer, scheduler,
                        epoch, best_mae, config)
            dist.barrier()
        if wandb_run is not None:
            wandb_run.finish()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
