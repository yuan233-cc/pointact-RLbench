#!/usr/bin/env python3
"""Train and validate CGA(+DINOv3) without loading the VLA."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointact.data.polar_normal_dataset import PolarNormalDataset, assert_disjoint_groups  # noqa: E402
from pointact.data.rlbench_polar_normal_lmdb import RLBenchPolarNormalLmdbDataset  # noqa: E402
from pointact.model.vla_pointact.action_head_3d.cga_dino_normal import (
    CgaDinoNormalNet,
    masked_cosine_normal_loss,
    normal_metrics,
)  # noqa: E402


def build_model(config: dict) -> CgaDinoNormalNet:
    model_cfg = config["model"]
    use_dino = model_cfg.get("use_dino", True)
    dataset_configs = config["data"].get("datasets") or [config["data"]]
    input_modes = {source.get("input_mode", config["data"].get("input_mode")) for source in dataset_configs}
    if None in input_modes or len(input_modes) != 1:
        raise ValueError(f"All mixed datasets must use one explicit input_mode, got {input_modes}")
    input_mode = input_modes.pop()
    kwargs = {
        "observation_channels": 11 if input_mode == "native_cga" else 7,
        "physical_prior_channels": 11,
        "transformer_blocks": model_cfg.get("transformer_blocks", 8),
        "transformer_dropout": model_cfg.get("transformer_dropout", 0.0),
        "use_dino": use_dino,
    }
    if use_dino:
        return CgaDinoNormalNet.from_dinov3(model_cfg["dinov3_weights"], **kwargs)
    return CgaDinoNormalNet(None, **kwargs)


def build_dataset(
    config: dict, split: str, dataset_config: dict | None = None
) -> PolarNormalDataset | RLBenchPolarNormalLmdbDataset:
    data = config["data"]
    source = data if dataset_config is None else {**data, **dataset_config}
    if source.get("format") == "rlbench_polar_lmdb":
        return RLBenchPolarNormalLmdbDataset(
            source[f"{split}_manifest"],
            dataset_root=source["dataset_root"],
            image_size=source.get("image_size", 256),
            require_rgb=config["model"].get("use_dino", True),
            refractive_index=source.get("refractive_index", 1.5),
            input_mode=source.get("input_mode", "robot"),
            ray_dropout_prob=source.get(f"ray_dropout_{split}", 0.0),
            limit=source.get("overfit_samples"),
        )
    return PolarNormalDataset(
        source[f"{split}_manifest"],
        input_mode=source["input_mode"],
        image_size=source.get("image_size", 256),
        require_rgb=config["model"].get("use_dino", True),
        require_calibration=source.get("require_calibration", True),
        normal_gt_source=source["normal_gt_source"],
        normal_transform=source.get("normal_transform"),
        normal_sign=source.get("normal_sign", 1.0),
        ray_dropout_prob=source.get(f"ray_dropout_{split}", 0.0),
        limit=source.get("overfit_samples"),
    )


def build_datasets(config: dict, split: str) -> list[tuple[str, PolarNormalDataset]]:
    sources = config["data"].get("datasets")
    if not sources:
        return [("all", build_dataset(config, split))]
    datasets = []
    names: set[str] = set()
    for source in sources:
        if not source.get("name"):
            raise ValueError("Every data.datasets entry must have a non-empty name")
        if not source.get(f"{split}_manifest"):
            continue
        name = str(source["name"])
        if name in names:
            raise ValueError(f"Duplicate data.datasets name: {name!r}")
        names.add(name)
        datasets.append((name, build_dataset(config, split, source)))
    if not datasets:
        raise ValueError(f"No data.datasets entry provides a {split}_manifest")
    return datasets


def balanced_sampler(datasets: list[PolarNormalDataset], seed: int) -> WeightedRandomSampler:
    weights = []
    for dataset in datasets:
        if len(dataset) == 0:
            raise ValueError("Cannot balance an empty dataset")
        weights.extend([1.0 / len(dataset)] * len(dataset))
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def aggregate_metrics(total: dict[str, float]) -> dict[str, float]:
    count = max(total["valid_count"], 1.0)
    return {
        "normal_mae": total["angle_sum"] / count,
        "within_11_25": total["within_11_25"] / count,
        "within_22_5": total["within_22_5"] / count,
        "valid_pixels": total["valid_count"],
    }


def collate_training_fields(samples: list[dict], *, use_rgb: bool) -> dict:
    """Batch only fields shared by all normal datasets and consumed by the model."""
    required = ("polar_observation", "physical_prior", "normal_gt", "normal_valid_mask")
    batch = default_collate([{key: sample[key] for key in required} for sample in samples])
    if use_rgb:
        batch["rgb"] = default_collate([sample["rgb"] for sample in samples])
    else:
        _, height, width = batch["polar_observation"].shape[1:]
        batch["rgb"] = torch.empty(len(samples), 0, height, width)
    return batch


@torch.no_grad()
def validate(model, loader, device) -> dict[str, float]:
    model.eval()
    total = dict.fromkeys(("angle_sum", "within_11_25", "within_22_5", "valid_count"), 0.0)
    loss_sum = 0.0
    batches = 0
    for batch in loader:
        inputs = {
            "polar_observation": batch["polar_observation"].to(device),
            "physical_prior": batch["physical_prior"].to(device),
            "rgb": batch["rgb"].to(device) if batch["rgb"].numel() else None,
        }
        target = batch["normal_gt"].to(device)
        mask = batch["normal_valid_mask"].to(device)
        prediction = model(**inputs)["normal"]
        loss_sum += float(masked_cosine_normal_loss(prediction, target, mask))
        metrics = normal_metrics(prediction, target, mask)
        for key in total:
            total[key] += float(metrics[key])
        batches += 1
    result = aggregate_metrics(total)
    result["loss"] = loss_sum / max(batches, 1)
    return result


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_mae, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_mae": best_mae,
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    train_cfg = config["training"]
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(train_cfg.get("seed", 0))

    named_train = build_datasets(config, "train")
    named_val = build_datasets(config, "val")
    assert_disjoint_groups(
        *(dataset for _, dataset in named_train), *(dataset for _, dataset in named_val)
    )
    train_datasets = [dataset for _, dataset in named_train]
    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    batch_size = train_cfg.get("batch_size", 2)
    accumulation = train_cfg.get("gradient_accumulation_steps", 4)
    sampler = (
        balanced_sampler(train_datasets, train_cfg.get("seed", 0))
        if len(train_datasets) > 1
        else None
    )
    collate = partial(collate_training_fields, use_rgb=config["model"].get("use_dino", True))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    val_loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )
        for name, dataset in named_val
    }
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg.get("step_size", 10),
        gamma=train_cfg.get("gamma", 0.5),
    )
    start_epoch, best_mae = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        missing = [name for name in missing if not name.startswith("dino.")]
        if missing or unexpected:
            raise RuntimeError(f"Resume mismatch: missing={missing}, unexpected={unexpected}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mae = float(checkpoint["best_mae"])

    output_dir = Path(train_cfg.get("output_dir", "outputs/polar_normal"))
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "metrics.jsonl"
    epochs = train_cfg.get("epochs", 30)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device) if batch["rgb"].numel() else None
            prediction = model(
                batch["polar_observation"].to(device),
                batch["physical_prior"].to(device),
                rgb,
            )["normal"]
            loss = masked_cosine_normal_loss(
                prediction,
                batch["normal_gt"].to(device),
                batch["normal_valid_mask"].to(device),
            )
            group_start = (step // accumulation) * accumulation
            group_steps = min(accumulation, len(train_loader) - group_start)
            (loss / group_steps).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach())
        per_dataset = {
            name: validate(model, loader, device) for name, loader in val_loaders.items()
        }
        metrics = {
            f"val/{name}/{key}": value
            for name, values in per_dataset.items()
            for key, value in values.items()
        }
        for key in ("normal_mae", "within_11_25", "within_22_5", "loss"):
            metrics[f"{key}_macro"] = sum(values[key] for values in per_dataset.values()) / len(
                per_dataset
            )
        metrics["valid_pixels"] = sum(values["valid_pixels"] for values in per_dataset.values())
        metrics.update(
            epoch=epoch,
            train_loss=running_loss / max(len(train_loader), 1),
            learning_rate=optimizer.param_groups[0]["lr"],
            effective_batch_size=batch_size * accumulation,
        )
        with history_path.open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics, sort_keys=True))
        is_best = metrics["normal_mae_macro"] < best_mae
        best_mae = min(best_mae, metrics["normal_mae_macro"])
        scheduler.step()
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, best_mae, config)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, best_mae, config)


if __name__ == "__main__":
    main()
