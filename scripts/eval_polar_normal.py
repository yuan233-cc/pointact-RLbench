#!/usr/bin/env python3
"""Evaluate angular normal metrics, visualizations, and inference latency."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointact.data.polar_normal_dataset import PolarNormalDataset  # noqa: E402
from pointact.model.vla_pointact.action_head_3d.cga_dino_normal import (
    CgaDinoNormalNet,
    load_cga_dino_normal_checkpoint,
    normal_metrics,
)  # noqa: E402


def make_model(config: dict) -> CgaDinoNormalNet:
    model_cfg, data_cfg = config["model"], config["data"]
    kwargs = {
        "observation_channels": 11 if data_cfg["input_mode"] == "native_cga" else 7,
        "physical_prior_channels": 11,
        "transformer_blocks": model_cfg.get("transformer_blocks", 8),
        "transformer_dropout": model_cfg.get("transformer_dropout", 0.0),
        "use_dino": model_cfg.get("use_dino", True),
    }
    if kwargs["use_dino"]:
        return CgaDinoNormalNet.from_dinov3(model_cfg["dinov3_weights"], **kwargs)
    return CgaDinoNormalNet(None, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Override the config embedded in checkpoint")
    parser.add_argument("--manifest", type=Path, help="Override the validation manifest")
    parser.add_argument("--output", type=Path, default=Path("outputs/polar_normal_eval"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--visualizations", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = (
        yaml.safe_load(args.config.read_text()) if args.config else raw_checkpoint.get("config")
    )
    if not config:
        raise ValueError("Checkpoint has no config; pass --config")
    if args.manifest:
        config["data"]["val_manifest"] = str(args.manifest)
    device_name = config.get("training", {}).get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    dataset = PolarNormalDataset(
        config["data"]["val_manifest"],
        input_mode=config["data"]["input_mode"],
        image_size=config["data"].get("image_size", 256),
        require_rgb=config["model"].get("use_dino", True),
        require_calibration=config["data"].get("require_calibration", True),
        normal_gt_source=config["data"]["normal_gt_source"],
        normal_transform=config["data"].get("normal_transform"),
        normal_sign=config["data"].get("normal_sign", 1.0),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = make_model(config).to(device).eval()
    load_cga_dino_normal_checkpoint(model, args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    totals = dict.fromkeys(("angle_sum", "within_11_25", "within_22_5", "valid_count"), 0.0)
    latencies = []
    visualized = 0
    warmed_up = False
    with torch.inference_mode():
        for batch in loader:
            observation = batch["polar_observation"].to(device)
            prior = batch["physical_prior"].to(device)
            rgb = batch["rgb"].to(device) if batch["rgb"].numel() else None
            if not warmed_up:
                for _ in range(args.warmup):
                    model(observation, prior, rgb)
                warmed_up = True
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = model(observation, prior, rgb)["normal"]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0 / len(observation))
            target = batch["normal_gt"].to(device)
            mask = batch["normal_valid_mask"].to(device)
            metrics = normal_metrics(prediction, target, mask)
            for key in totals:
                totals[key] += float(metrics[key])
            for item in range(len(prediction)):
                if visualized >= args.visualizations:
                    break
                valid = mask[item].float()
                panel = torch.cat(
                    (
                        (prediction[item].cpu() + 1.0) * 0.5 * valid.cpu(),
                        (target[item].cpu() + 1.0) * 0.5 * valid.cpu(),
                    ),
                    dim=2,
                )
                save_image(panel, args.output / f"{visualized:04d}_{batch['sample_id'][item]}.png")
                visualized += 1
    count = max(totals["valid_count"], 1.0)
    summary = {
        "normal_mae": totals["angle_sum"] / count,
        "within_11_25": totals["within_11_25"] / count,
        "within_22_5": totals["within_22_5"] / count,
        "valid_pixels": totals["valid_count"],
        "latency_ms_per_image_mean": sum(latencies) / max(len(latencies), 1),
        "latency_ms_per_image_median": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
