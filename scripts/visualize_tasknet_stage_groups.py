"""Draw the actual PTv3 block groups for one RLBench V2 batch and anchor.

This evaluates dataset row selection, optional depth holdout, serialization,
padding and pooling. It never runs TaskNet, attention, or a training step.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from pointact.data.robot.multi_data import load_single_lerobot_dataset
from pointact.data.schema import DataConfig
from pointact.model.vla_pointact.action_head_3d.polar_bbox_fusion import (
    group_image_boxes, image_support_from_pixels, mask_points_at_pixel_targets,
)
from pointact.model.ptv3.concerto.serialization import encode


ORDER_NAMES = ("z", "z-trans", "hilbert", "hilbert-trans")
DEPTHS = (3, 3, 3, 12, 3)
OTHER_GROUP_COLORS = np.array(("#3674a5", "#31968f", "#7563a6", "#66815f",
                               "#4f7e9d", "#8a6e9e", "#397d79", "#6c79a4"))


def image_from_item(item):
    image = item.get("observation.images.front_image")
    if image is None:
        gray = item["polar_images"][0, 0].detach().cpu().numpy()
        return np.repeat(gray[..., None], 3, axis=-1), "SfP intensity"
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    while image.ndim > 3:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (3, 4):
        image = np.moveaxis(image[:3], 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unexpected RLBench RGB shape {image.shape}")
    image = image.astype(np.float32)
    if image.max() > 1:
        image /= 255.0
    return np.clip(image, 0, 1), "RLBench RGB"


def anchored_group_jaccard(first, second):
    """Point-wise Jaccard of PRIMARY (unpadded) group memberships."""
    n_second = int(second.max()) + 1
    first_size = np.bincount(first)
    second_size = np.bincount(second)
    intersection = np.bincount(first * n_second + second, minlength=len(first_size) * n_second)
    shared = intersection[first * n_second + second]
    return shared / (first_size[first] + second_size[second] - shared)


def boxes_iou(first, second):
    lo = np.maximum(first[:, :2], second[:, :2])
    hi = np.minimum(first[:, 2:], second[:, 2:])
    overlap = np.maximum(hi - lo + 1, 0).prod(-1)
    left = np.maximum(first[:, 2:] - first[:, :2] + 1, 0).prod(-1)
    right = np.maximum(second[:, 2:] - second[:, :2] + 1, 0).prod(-1)
    return overlap / np.maximum(left + right - overlap, 1e-8)


def summaries(values):
    return {"min": float(np.min(values)), "p10": float(np.percentile(values, 10)),
            "median": float(np.median(values)), "mean": float(np.mean(values))}


def family_names(point):
    expected = {
        name: encode(point.grid_coord, point.batch, point.serialized_depth, order=name)
        for name in ORDER_NAMES
    }
    return [next((name for name, code in expected.items()
                  if torch.equal(code, row)), f"slot {slot}")
            for slot, row in enumerate(point.serialized_code)]


def draw_stage(stage, slots, original_uv, reference_image, source_label, anchor, output):
    height, width = reference_image.shape[:2]
    count = DEPTHS[stage]
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 4.6 * rows + 0.7), squeeze=False)
    for block_index, axis in enumerate(axes.flat):
        if block_index >= count:
            axis.axis("off")
            continue
        slot_index = block_index % len(slots)
        block = slots[slot_index]
        point_group = block["original_group"]
        anchor_group = point_group[anchor]
        chosen = point_group == anchor_group
        assigned_boxes = block["boxes"][point_group]
        inside = ((original_uv[:, 0] >= assigned_boxes[:, 0]) &
                  (original_uv[:, 0] <= assigned_boxes[:, 2]) &
                  (original_uv[:, 1] >= assigned_boxes[:, 1]) &
                  (original_uv[:, 1] <= assigned_boxes[:, 3]))
        if not np.all(inside):
            raise ValueError(f"Stage {stage}, block {block['numbers']}: "
                             f"{np.count_nonzero(~inside)} primary-group points outside bbox")
        # Reserve orange exclusively for the selected group. tab20 itself
        # contains orange, which made unrelated points appear outside its box.
        other = ~chosen
        colors = OTHER_GROUP_COLORS[point_group[other] % len(OTHER_GROUP_COLORS)]
        axis.imshow(reference_image, alpha=0.60, interpolation="nearest")
        axis.scatter(original_uv[other, 0], original_uv[other, 1], c=colors, s=3.0,
                     alpha=0.60, linewidths=0, rasterized=True)
        axis.scatter(original_uv[chosen, 0], original_uv[chosen, 1], c="#ff9c24",
                     s=5.5, alpha=0.95, linewidths=0, rasterized=True)
        box = block["boxes"][anchor_group]
        axis.add_patch(Rectangle((box[0] - 0.5, box[1] - 0.5),
                                 box[2] - box[0] + 1, box[3] - box[1] + 1,
                                 fill=False, edgecolor="#ff9c24", linewidth=2))
        axis.scatter([original_uv[anchor, 0]], [original_uv[anchor, 1]],
                     marker="x", color="white", s=65, linewidths=2.1)
        image_fraction = (box[2] - box[0] + 1) * (box[3] - box[1] + 1) / (width * height)
        repeated = f" · same support as B{slot_index}" if block_index >= len(slots) else ""
        axis.set_title(f"Block {block_index} · {block['family']}{repeated}\n"
                       f"{block['n_groups']} groups · anchor {np.sum(chosen)} pts · box {image_fraction:.1%}",
                       fontsize=10)
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(height - 0.5, -0.5)
        axis.set_xlabel("u / pixel")
        axis.set_ylabel("v / pixel")
    fig.suptitle(f"Stage {stage} · all {count} blocks · {source_label} image coordinates\n"
                 "Orange = anchor's group; white × = fixed anchor; repeated support does not mean repeated features",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def draw_summary(rows, output, sample_index):
    metrics = (
        ("membership_jaccard", "Primary group membership Jaccard · median"),
        ("bbox_iou", "Inherited group bbox IoU · median"),
        ("bbox_iou_p10", "Inherited group bbox IoU · P10"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.2), squeeze=False)
    for axis, (key, title) in zip(axes[0], metrics):
        grid = np.full((5, 4), np.nan)
        for entry in rows:
            grid[entry["stage"], entry["pair_slot"]] = entry[key]
        display = axis.imshow(np.ma.masked_invalid(grid), vmin=0, vmax=1,
                              cmap="viridis", aspect="auto")
        for stage in range(5):
            for pair in range(4):
                if np.isfinite(grid[stage, pair]):
                    value = grid[stage, pair]
                    axis.text(pair, stage, f"{value:.2f}", ha="center", va="center",
                              color="black" if value > 0.55 else "white", fontsize=10)
        axis.set_title(title, fontsize=11)
        axis.set_yticks(range(5), [f"Stage {stage}" for stage in range(5)])
        axis.set_xticks(range(4), ["0→1", "1→2", "2→3", "3→0"])
        axis.set_xlabel("Adjacent block's order slot")
        fig.colorbar(display, ax=axis, fraction=0.045, pad=0.03)
    fig.suptitle(f"V2 frame {sample_index} · same-stage adjacent-block changes · alpha = 1.0\n"
                 "Blank = that block transition does not occur in the stage", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-json", type=Path, required=True,
                        help="Select the exact first batch recorded by the real-data coverage probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    coverage = json.loads(args.coverage_json.read_text())
    meta = coverage["metadata"]
    if meta["grouping"] != "flash":
        raise ValueError("This figure currently uses FlashAttention-style patch grouping")
    data_config = DataConfig.from_yaml(meta["data_path"]).lerobot_datasets[0]
    if meta.get("dataset_root"):
        data_config.root = meta["dataset_root"]
    torch.manual_seed(meta["seed"])
    np.random.seed(meta["seed"])
    dataset = load_single_lerobot_dataset(
        0, [data_config], image_transforms=None, download_videos=False,
        video_backend="pyav", chunk_size=1,
    )
    selected = meta["indices"][:4]
    items = [dataset[index] for index in selected]
    # The training config deliberately omits RGB from VLM inputs, so fetch
    # this one frame explicitly for the display background only.
    first_item = dict(items[0])
    if "observation.images.front_image" not in first_item:
        record = dataset.hf_dataset[selected[0]]
        rgb = dataset._query_videos(
            {"observation.images.front_image": [float(record["timestamp"].item())]},
            int(record["episode_index"].item()),
        )
        first_item.update(rgb)
    reference_image, source_label = image_from_item(first_item)
    points = torch.cat([item["observation.points"] for item in items]).to(args.device)
    pixels = torch.cat([item["point_pixel_indices"] for item in items]).to(args.device)
    hw = torch.stack([item["point_pixel_image_hw"] for item in items]).to(args.device)
    counts = torch.tensor([len(item["observation.points"]) for item in items], device=args.device)
    valid = torch.stack([item["observed_depth_valid"] for item in items]).to(args.device).bool()
    holdout = valid & ~(torch.rand_like(valid.float()) < meta["depth_keep_probability"])
    points, counts, keep = mask_points_at_pixel_targets(points, counts, pixels, holdout, source_hw=hw)
    pixels = pixels[keep]
    batch_ids = torch.arange(len(items), device=args.device).repeat_interleave(counts)
    support = image_support_from_pixels(pixels, batch_ids, hw)
    module = importlib.import_module(f"pointact.model.ptv3.{meta['backend']}.model")
    structure = importlib.import_module(f"pointact.model.ptv3.{meta['backend']}.structure")
    pool = [module.GridPooling(6, 6, stride=2, norm_layer=torch.nn.LayerNorm,
                               act_layer=torch.nn.GELU, shuffle_orders=True).to(args.device)
            for _ in range(4)]
    attention = module.SerializedAttention(channels=6, num_heads=1,
                                            patch_size=meta["patch_size"], enable_flash=False,
                                            upcast_attention=False, upcast_softmax=False)
    point = structure.Point({"coord": points[:, :3], "feat": points.new_zeros((len(points), 6)),
                             "batch": batch_ids, "offset": counts.cumsum(0),
                             "grid_size": meta["voxel_size"], "image_support": support,
                             "input_image_support": support, "input_point_batch": batch_ids,
                             "input_to_stage": torch.arange(len(points), device=args.device)})
    point.serialization(order=ORDER_NAMES, shuffle_orders=True)
    point.sparsify()
    focus_count = int(counts[0].item())
    original_pixels = pixels[:focus_count].cpu().numpy()
    width = int(hw[0, 0, 1].item())
    original_uv = np.column_stack((original_pixels % width, original_pixels // width))
    if reference_image.shape[:2] != tuple(hw[0, 0].cpu().tolist()):
        raise ValueError("Displayed image and point pixel grid have different dimensions")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stage_data = []
    for stage in range(5):
        if stage:
            point = pool[stage - 1](point)
        attention.patch_size = meta["patch_size"]
        pad, unpad, cumulative = attention.get_padding_and_inverse(point)
        lengths = torch.diff(cumulative).long()
        group_ids = torch.arange(len(lengths), device=args.device).repeat_interleave(lengths)
        lineage = point.input_to_stage[:focus_count]
        families = family_names(point)
        nslots = min(4, DEPTHS[stage])
        slots = []
        for slot in range(nslots):
            order = point.serialized_order[slot][pad]
            boxes, _ = group_image_boxes(point, order, cumulative)
            inverse = unpad[point.serialized_inverse[slot]]
            canonical_group = group_ids[inverse]
            repeats = [str(block) for block in range(DEPTHS[stage]) if block % 4 == slot]
            slots.append({"numbers": ",".join(repeats), "family": families[slot],
                          "groups": canonical_group[point.batch == 0].cpu().numpy(),
                          "original_group": canonical_group[lineage].cpu().numpy(),
                          "boxes": boxes.cpu().numpy(),
                          "n_groups": int((point.batch == 0).sum().item() // attention.patch_size + 1)})
        # Recompute sample group count from unique canonical labels; the last
        # padded patch is still a single group even if it repeats old points.
        for slot in slots:
            slot["n_groups"] = int(np.unique(slot["groups"]).size)
        stage_data.append({"stage": stage, "slots": slots,
                           "stage_points": int((point.batch == 0).sum().item())})

    first, second = stage_data[0]["slots"][:2]
    group_jaccard = anchored_group_jaccard(first["groups"], second["groups"])
    stage0_lineage = original_uv.shape[0]
    score = anchored_group_jaccard(first["original_group"], second["original_group"])
    tenth = np.percentile(score, 10)
    anchor = int(np.argmin(np.abs(score - tenth)))
    rows = []
    for stage in stage_data:
        slots = stage["slots"]
        draw_stage(stage["stage"], slots, original_uv, reference_image, source_label,
                   anchor, args.output_dir / f"stage_{stage['stage']}_groups.png")
        for block in range(DEPTHS[stage["stage"]] - 1):
            left_slot, right_slot = block % 4, (block + 1) % 4
            left, right = slots[left_slot], slots[right_slot]
            member = anchored_group_jaccard(left["groups"], right["groups"])
            box = boxes_iou(left["boxes"][left["groups"]],
                            right["boxes"][right["groups"]])
            rows.append({"stage": stage["stage"], "block_pair": [block, block + 1],
                         "pair_slot": left_slot, "orders": [left["family"], right["family"]],
                         "sample": selected[0], "stage_points": stage["stage_points"],
                         "group_counts": [left["n_groups"], right["n_groups"]],
                         "membership_jaccard": summaries(member),
                         "bbox_iou": summaries(box)})
    summary_rows = []
    for row in rows:
        summary_rows.append({**row, "membership_jaccard": row["membership_jaccard"]["median"],
                             "bbox_iou": row["bbox_iou"]["median"],
                             "bbox_iou_p10": row["bbox_iou"]["p10"]})
    draw_summary(summary_rows, args.output_dir / "stage_comparison.png", selected[0])
    report = {"source": str(args.coverage_json), "selected_batch": selected,
              "sample": selected[0], "anchor_pixel_uv": original_uv[anchor].tolist(),
              "anchor_selection": "nearest to P10 primary-group Jaccard of stage 0 block 0 vs 1",
              "original_points_after_holdout": stage0_lineage,
              "membership_definition": "point-wise Jaccard of canonical primary groups; padding replicas still contribute to bboxes",
              "order_shuffle": "one seeded training-style realization; repeated block slots reuse geometry",
              "per_adjacent_block": rows}
    with (args.output_dir / "summary.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output_dir), "sample": selected[0],
                      "stage0_group_jaccard_median": float(np.median(group_jaccard))}))


if __name__ == "__main__":
    main()
