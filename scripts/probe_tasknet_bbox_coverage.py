"""Geometry-only real-V2 coverage probe; no TaskNet or attention forward.

Use the production dataset row operations, PTv3 serialization/padding and
GridPooling. Compare groups by canonical POINT anchor, never by group number.
This is a deferred diagnostic tool, not a training launcher or speed benchmark.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pointact.data.robot.multi_data import load_single_lerobot_dataset
from pointact.data.schema import DataConfig
from pointact.model.vla_pointact.action_head_3d.polar_bbox_fusion import (
    expand_image_boxes,
    group_image_boxes,
    image_support_from_pixels,
    mask_points_at_pixel_targets,
)
from pointact.model.vla_pointact.action_head_3d.polarapp_tasknet_encoder import PolarAppTaskAwareEncoder


def area(boxes):
    # Inclusive pixel-center bounds have a positive area even for one pixel.
    return (boxes[:, 2:] - boxes[:, :2] + 1).clamp_min(0).prod(-1)


def intersection(left, right):
    low = torch.maximum(left[:, :2], right[:, :2])
    high = torch.minimum(left[:, 2:], right[:, 2:])
    return (high - low + 1).clamp_min(0).prod(-1)


def summarize(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not len(values):
        return {"count": 0}
    quantiles = torch.quantile(values, values.new_tensor([0.05, 0.10, 0.50, 0.90, 0.95]))
    return {
        "count": len(values), "mean": values.mean().item(),
        "median": quantiles[2].item(), "p5": quantiles[0].item(),
        "p10": quantiles[1].item(), "p90": quantiles[3].item(),
        "p95": quantiles[4].item(), "minimum": values.min().item(),
        "fraction_ge_90pct": (values >= 0.90).double().mean().item(),
        "fraction_ge_95pct": (values >= 0.95).double().mean().item(),
        "fraction_ge_99pct": (values >= 0.99).double().mean().item(),
    }


def feature_area_fraction(boxes, image_hw, level):
    stride = PolarAppTaskAwareEncoder.task_feature_strides[level]
    offset = PolarAppTaskAwareEncoder.task_feature_offsets[level]
    feature_hw = torch.div(image_hw + stride - 1, stride, rounding_mode="floor")
    limit = feature_hw.flip(-1).float() - 1
    low = torch.minimum(((boxes[:, :2] - offset) / stride).clamp_min(0), limit)
    high = torch.minimum(((boxes[:, 2:] - offset) / stride).clamp_min(0), limit)
    return area(torch.cat((low, high), -1)) / feature_hw.prod(-1)


@torch.no_grad()
def probe(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing probe report: {args.output}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    configs = DataConfig.from_yaml(str(args.data_path)).lerobot_datasets
    if not 0 <= args.dataset_index < len(configs):
        raise ValueError("dataset-index is outside the configured dataset list")
    config = configs[args.dataset_index]
    if not config.use_point_image_support:
        raise ValueError("Probe requires use_point_image_support=True and current point_pixel_indices")
    if args.dataset_root is not None:
        root = args.dataset_root.resolve(strict=True)
        if root.name != config.repo_id or not (root / "meta/info.json").is_file():
            raise ValueError("dataset-root must be the configured V2 dataset directory")
        config.root = str(root.parent)
        if config.state_action_norm_file:
            config.state_action_norm_file = str(root / "robot_state_action_stats" / Path(config.state_action_norm_file).name)
    dataset = load_single_lerobot_dataset(
        args.dataset_index, configs, image_transforms=None, download_videos=False,
        video_backend=args.video_backend, chunk_size=1,
    )
    model_module = importlib.import_module(f"pointact.model.ptv3.{args.backend}.model")
    structure = importlib.import_module(f"pointact.model.ptv3.{args.backend}.structure")
    # Feature values/norms do not affect the pooling membership. Six dummy
    # channels allow both backends (including Utonia's 3D RoPE constructor).
    pools = [model_module.GridPooling(
        6, 6, stride=2, norm_layer=torch.nn.LayerNorm,
        act_layer=torch.nn.GELU, shuffle_orders=False,
    ).to(args.device)
             for _ in range(4)]
    attention = model_module.SerializedAttention(
        channels=6, num_heads=1, patch_size=args.patch_size,
        enable_flash=False, upcast_attention=False, upcast_softmax=False,
    )
    # Only get_padding_and_inverse is used. Flash-style grouping therefore
    # does not require calling or installing FlashAttention for this probe.
    selected = np.random.default_rng(args.seed).choice(
        len(dataset), size=min(args.samples, len(dataset)), replace=False,
    ).tolist()
    per_sample, per_batch = [], []
    aggregate = defaultdict(list)
    for batch_id, start in enumerate(range(0, len(selected), args.batch_size)):
        sample_indices = selected[start:start + args.batch_size]
        items = [dataset[index] for index in sample_indices]
        hw = torch.stack([item["point_pixel_image_hw"] for item in items]).to(args.device)
        points = torch.cat([item["observation.points"] for item in items]).to(args.device)
        pixels = torch.cat([item["point_pixel_indices"] for item in items]).to(args.device)
        counts = torch.tensor([len(item["observation.points"]) for item in items], device=args.device)
        if args.depth_keep_probability < 1:
            valid = torch.stack([item["observed_depth_valid"] for item in items]).to(args.device).bool()
            targets = valid & ~(torch.rand_like(valid.float()) < args.depth_keep_probability)
            points, counts, keep = mask_points_at_pixel_targets(points, counts, pixels, targets, source_hw=hw)
            pixels = pixels[keep]
        batch_ids = torch.arange(len(items), device=args.device).repeat_interleave(counts)
        support = image_support_from_pixels(pixels, batch_ids, hw)
        point = structure.Point({
            "coord": points[:, :3], "feat": points.new_zeros((len(points), 6)),
            "batch": batch_ids, "offset": counts.cumsum(0), "grid_size": args.voxel_size,
            "image_support": support, "input_image_support": support,
            "input_point_batch": batch_ids, "input_to_stage": torch.arange(len(points), device=args.device),
        })
        point.serialization(order=args.orders, shuffle_orders=False)
        point.sparsify()
        for stage in range(5):
            if stage:
                point = pools[stage - 1](point)
            stage_counts = torch.diff(point.offset, prepend=point.offset.new_zeros(1))
            attention.patch_size = (args.patch_size if args.grouping == "flash" else
                                    min(args.patch_size, int(stage_counts.min().item())))
            pad, unpad, cumulative = attention.get_padding_and_inverse(point)
            lengths = torch.diff(cumulative).long()
            group_ids = torch.arange(len(lengths), device=args.device).repeat_interleave(lengths)
            by_order, area_rows = {}, []
            for order_id, name in enumerate(args.orders):
                order = point.serialized_order[order_id][pad]
                inverse = unpad[point.serialized_inverse[order_id]]
                boxes, group_samples = group_image_boxes(point, order, cumulative)
                # Each canonical point is anchored to its primary (unpadded)
                # occurrence; padding replicas contribute to bbox membership.
                anchors = group_ids[inverse]
                by_order[name] = boxes[anchors]
                for alpha in args.alphas:
                    expanded = expand_image_boxes(boxes, hw[group_samples, 0], alpha)
                    sizes = {
                        "image_area_fraction": area(expanded) / hw[group_samples, 0].prod(-1),
                        "feature_area_fraction": feature_area_fraction(expanded, hw[group_samples, 0], args.feature_levels[stage]),
                        "aspect_ratio": torch.maximum(
                            (expanded[:, 2] - expanded[:, 0] + 1) / (expanded[:, 3] - expanded[:, 1] + 1),
                            (expanded[:, 3] - expanded[:, 1] + 1) / (expanded[:, 2] - expanded[:, 0] + 1),
                        ),
                    }
                    for metric, values in sizes.items():
                        aggregate[(stage, name, float(alpha), metric)].extend(values.cpu().tolist())
                    batch_row = {"kind": "region", "batch": batch_id, "stage": stage,
                                 "order": name, "alpha": alpha, "weighting": "attention_group"}
                    batch_row.update({key: summarize(value.cpu()) for key, value in sizes.items()})
                    per_batch.append(batch_row)
                    for sample_id, dataset_index in enumerate(sample_indices):
                        row = {"kind": "region", "batch": batch_id, "sample": dataset_index,
                               "stage": stage, "order": name, "alpha": alpha}
                        row.update({key: summarize(value[group_samples == sample_id].cpu()) for key, value in sizes.items()})
                        area_rows.append(row)
            per_sample.extend(area_rows)
            for left_name, right_name in itertools.combinations(args.orders, 2):
                left, right = by_order[left_name], by_order[right_name]
                overlap = intersection(left, right)
                iou = overlap / (area(left) + area(right) - overlap).clamp_min(1e-8)
                for alpha in args.alphas:
                    image_hw = hw[point.batch, 0]
                    values = {
                        "iou": iou,
                        "coverage_A_to_B": intersection(expand_image_boxes(left, image_hw, alpha), right) / area(right),
                        "coverage_B_to_A": intersection(expand_image_boxes(right, image_hw, alpha), left) / area(left),
                    }
                    pair = f"{left_name}__{right_name}"
                    row = {"kind": "coverage", "batch": batch_id, "stage": stage,
                           "pair": pair, "alpha": alpha, "weighting": "canonical_stage_point"}
                    row.update({key: summarize(value.cpu()) for key, value in values.items()})
                    per_batch.append(row)
                    for key, value in values.items():
                        aggregate[(stage, pair, float(alpha), key)].extend(value.cpu().tolist())
                    for sample_id, dataset_index in enumerate(sample_indices):
                        sample = dict(row, sample=dataset_index)
                        sample.update({key: summarize(value[point.batch == sample_id].cpu()) for key, value in values.items()})
                        per_sample.append(sample)
            print(f"batch={batch_id} stage={stage} points={len(point.coord)} groups={len(lengths)}", flush=True)
    report = {
        "metadata": {
            "data_path": str(args.data_path), "dataset_root": config.root,
            "repo_id": config.repo_id, "indices": selected, "seed": args.seed,
            "backend": args.backend, "patch_size": args.patch_size, "voxel_size": args.voxel_size,
            "orders": args.orders, "grouping": args.grouping, "feature_levels": args.feature_levels,
            "depth_keep_probability": args.depth_keep_probability,
            "coordinate_convention": "inclusive image pixel centers; feature bounds divided by native stride then clipped",
            "attention_forward": False, "tasknet_forward": False,
            "order_shuffle": False,
            "note": "Order families are fixed for labeling. Training may shuffle their slots; compare by point anchor. Region stats are group-weighted; coverage is point-weighted. Alpha is not auto-selected.",
        },
        "aggregate": [{"stage": key[0], "order_or_pair": key[1], "alpha": key[2],
                       "metric": key[3], **summarize(value)} for key, value in sorted(aggregate.items())],
        "per_batch": per_batch, "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved coverage report: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("experiments/10_rlbench/data_configs/data-10task-polar-rlbench9-v2-tasknet-bbox.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--backend", choices=("concerto", "utonia"), default="concerto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--grouping", choices=("flash", "reference"), default="flash")
    parser.add_argument("--depth-keep-probability", type=float, default=0.7)
    parser.add_argument("--orders", nargs="+", default=["z", "z-trans", "hilbert", "hilbert-trans"])
    parser.add_argument("--feature-levels", nargs=5, type=int, default=[0, 0, 1, 2, 2])
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.patch_size) <= 0 or not math.isfinite(args.voxel_size) or args.voxel_size <= 0:
        parser.error("Sample/batch/patch counts and voxel size must be positive")
    if len(set(args.orders)) != len(args.orders) or len(args.orders) < 2:
        parser.error("Provide at least two distinct serialization orders")
    if any(not math.isfinite(alpha) or alpha < 1 for alpha in args.alphas):
        parser.error("All expansions must be finite and >=1")
    if any(level not in (0, 1, 2) for level in args.feature_levels):
        parser.error("TaskNet feature levels must be 0, 1 or 2")
    if not 0 < args.depth_keep_probability <= 1:
        parser.error("depth-keep-probability must be in (0,1]")
    probe(args)


if __name__ == "__main__":
    main()
