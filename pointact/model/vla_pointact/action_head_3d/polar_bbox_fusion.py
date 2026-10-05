"""Dense polar sampling from inherited image support, never from pooled XYZ.

The first version supports one calibrated image per sample (RLBench frontview).
Boxes store inclusive pixel-center bounds [u_min, v_min, u_max, v_max].
Serialization only changes index maps: support stays in canonical point order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def image_support_from_pixels(
    pixels: Tensor,
    batch_ids: Tensor,
    image_hw: Tensor,
    pixel_transform: Tensor | None = None,
    source_hw: Tensor | None = None,
) -> Tensor:
    """Decode the CURRENT observation's flat pixel IDs, not clean/source IDs."""
    if image_hw.ndim != 3 or image_hw.shape[1:] != (1, 2):
        raise ValueError("BBox fusion requires one image per sample: image_hw [B,1,2]")
    if pixels.ndim != 1 or len(pixels) != len(batch_ids):
        raise ValueError("point_pixel_indices must be [sum(N)] in final point-row order")
    if pixels.dtype.is_floating_point or pixels.dtype == torch.bool:
        raise ValueError("point_pixel_indices must contain integer pixel IDs")
    original_hw = image_hw if source_hw is None else source_hw
    if original_hw.shape != image_hw.shape:
        raise ValueError("point_pixel_image_hw must be [B,1,2]")
    if source_hw is not None and pixel_transform is None:
        torch._assert((source_hw == image_hw).all(),
                      "Changing image size requires an explicit polar_pixel_transform")
    height, width = original_hw[batch_ids, 0].long().unbind(-1)
    torch._assert(((pixels >= 0) & (pixels < height * width)).all(),
                  "Input point pixel IDs are outside their observation image")
    uv = torch.stack((pixels.remainder(width), pixels.div(width, rounding_mode="floor")), -1).float()
    if pixel_transform is not None:
        if pixel_transform.shape != (len(image_hw), 1, 3, 3):
            raise ValueError("polar_pixel_transform must be [B,1,3,3]")
        homogeneous = torch.cat((uv, torch.ones_like(uv[:, :1])), -1)
        warped = torch.bmm(pixel_transform[batch_ids, 0].float(), homogeneous.unsqueeze(-1)).squeeze(-1)
        torch._assert((warped[:, 2].abs() > 1e-6).all(), "Invalid pixel transform denominator")
        uv = warped[:, :2] / warped[:, 2:]
    image_limit = image_hw[batch_ids, 0].flip(-1).float() - 1
    torch._assert((torch.isfinite(uv).all(-1) & (uv >= 0).all(-1)
                   & (uv <= image_limit).all(-1)).all(),
                  "Transformed point pixels must remain inside the current image")
    return torch.cat((uv, uv), -1)


def inherit_image_support(point, pooled: dict, cluster: Tensor, indices: Tensor, idx_ptr: Tensor) -> None:
    """Reuse the EXACT PTv3 pooling membership for support and input lineage."""
    if "image_support" not in point:
        return
    support = point.image_support
    if support.shape != (len(point.coord), 4):
        raise ValueError("Every canonical point row must have one image support box")
    lengths = torch.diff(idx_ptr).long()
    sorted_support = support[indices]
    minimum = torch.segment_reduce(sorted_support[:, :2], "min", lengths=lengths)
    maximum = torch.segment_reduce(sorted_support[:, 2:], "max", lengths=lengths)
    pooled["image_support"] = torch.cat((minimum, maximum), -1)
    pooled["input_to_stage"] = cluster[point.input_to_stage]
    pooled["input_image_support"] = point.input_image_support
    pooled["input_point_batch"] = point.input_point_batch


def group_image_boxes(point, order: Tensor, cumulative: Tensor) -> tuple[Tensor, Tensor]:
    """Union current patch members, including the actual attention padding."""
    if "image_support" not in point or point.image_support.shape != (len(point.feat), 4):
        raise ValueError("BBox attention requires image_support aligned to canonical point rows")
    lengths = torch.diff(cumulative).long()
    supports = point.image_support[order]
    minimum = torch.segment_reduce(supports[:, :2], "min", lengths=lengths)
    maximum = torch.segment_reduce(supports[:, 2:], "max", lengths=lengths)
    samples = point.batch[order[cumulative[:-1].long()]].long()
    torch._assert(torch.equal(point.batch[order], samples.repeat_interleave(lengths)),
                  "A serialized patch must not cross sample boundaries")
    return torch.cat((minimum, maximum), -1), samples


def expand_image_boxes(boxes: Tensor, image_hw: Tensor, alpha: float) -> Tensor:
    """Expand inclusive pixel cells about their center and clip to the image."""
    if alpha < 1:
        raise ValueError("BBox expansion must be >= 1")
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    span = ((boxes[:, 2:] - boxes[:, :2] + 1) * alpha - 1) * 0.5
    limit = image_hw.flip(-1).to(boxes.dtype) - 1
    low = torch.minimum((center - span).clamp_min(0), limit)
    high = torch.minimum((center + span).clamp_min(0), limit)
    return torch.cat((low, high), -1)


def regular_box_grid(boxes: Tensor, side: int) -> Tensor:
    """Fixed cell-center grid: [groups, side*side, 2] in image pixels."""
    if side <= 0:
        raise ValueError("polar_bbox_grid_size must be positive")
    line = (torch.arange(side, device=boxes.device, dtype=boxes.dtype) + 0.5) / side
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    fractions = torch.stack((xx, yy), -1).reshape(1, side * side, 2)
    # A singleton support is exactly one pixel; no hidden minimum-size heuristic.
    return boxes[:, None, :2] + fractions * (boxes[:, None, 2:] - boxes[:, None, :2])


@dataclass
class BboxPolarTokens:
    features: Tensor
    xy: Tensor
    boxes: Tensor
    expanded_boxes: Tensor
    sample_ids: Tensor


class BboxPolarSampler:
    def __init__(self, stage: int, side: int = 4, expansion: float = 1.0):
        self.stage = stage
        self.side = side
        self.expansion = expansion

    def __call__(self, point, order: Tensor, cumulative: Tensor) -> BboxPolarTokens:
        with torch.profiler.record_function("polar_bbox/union"):
            boxes, samples = group_image_boxes(point, order, cumulative)
            features = point.polar_features
            if features.ndim != 5 or features.shape[1] != 1:
                raise ValueError("BBox fusion currently requires features [B,1,Hf,Wf,C]")
            torch._assert(point.view_valid[:, 0].all(), "BBox fusion requires a valid frontview for every sample")
            image_hw = point.polar_image_hw[samples, 0]
            expanded = expand_image_boxes(boxes, image_hw, self.expansion)
        with torch.profiler.record_function("polar_bbox/grid"):
            uv = regular_box_grid(expanded, self.side)
            stride, offset = point.polar_bbox_geometry
            feature_uv = (uv - offset) / stride
            batch, _, height, width, channels = features.shape
            feature_size = uv.new_tensor((width, height))
            grid = 2 * (feature_uv + 0.5) / feature_size - 1
            # Pack all group grids into the sample dimension. Do not replicate
            # a full-resolution image feature bank once for each point group.
            counts = torch.bincount(samples, minlength=batch)
            max_groups = int(counts.max().item())
            starts = counts.cumsum(0) - counts
            local_group = torch.arange(len(samples), device=samples.device) - starts[samples]
            padded_grid = grid.new_zeros((batch, max_groups, self.side * self.side, 2))
            padded_grid[samples, local_group] = grid
        with torch.profiler.record_function("polar_bbox/grid_sample"):
            # Geometry and bilinear interpolation use float32 (including
            # bf16 training). The context converts the native bank once, not
            # once per block; adapter/qkv still follow autocast afterwards.
            bank = features[:, 0].permute(0, 3, 1, 2).float()
            sampled = F.grid_sample(
                bank, padded_grid.reshape(batch, max_groups * self.side * self.side, 1, 2),
                mode="bilinear", padding_mode="border", align_corners=False,
            )
            sampled = sampled.reshape(batch, channels, max_groups, self.side * self.side).permute(0, 2, 3, 1)
            tokens = sampled[samples, local_group]
            xy = 2 * (uv + 0.5) / image_hw.flip(-1)[:, None].to(uv.dtype) - 1
        with torch.profiler.record_function("polar_bbox/token_adapter"):
            tokens = point.polar_bbox_norm(point.polar_bbox_adapter(tokens.to(point.polar_bbox_adapter.weight.dtype)))
        # No point-valid / depth-valid mask is applied to the dense polar bank.
        return BboxPolarTokens(tokens, xy, boxes, expanded, samples)


def forward_bbox_joint_attention(attention, point, flash_function=None, point_rope=None):
    """Vectorized [A | P | Z] packing, attention and canonical output restore."""
    counts = torch.diff(point.offset, prepend=point.offset.new_zeros(1))
    if not attention.enable_flash:
        attention.patch_size = min(int(counts.min().item()), attention.patch_size_max)
    pad, unpad, point_cu = attention.get_padding_and_inverse(point)
    lengths = torch.diff(point_cu).long()
    order = point.serialized_order[attention.order_index][pad]
    inverse = unpad[point.serialized_inverse[attention.order_index]]
    routed = attention.polar_bbox_sampler(point, order, point_cu)
    groups, tokens, channels = routed.features.shape
    heads, actions = attention.num_heads, point.action_feat.shape[1]
    head_dim = channels // heads
    with torch.profiler.record_function("polar_bbox/projection_pack"):
        polar = routed.features + attention.polar_position(routed.xy.to(routed.features.dtype))
        polar = polar + attention.polar_view_embedding.weight[0] + attention.polar_modality_embedding
        polar_qkv = attention.polar_qkv(attention.polar_norm(polar)).reshape(-1, 3, heads, head_dim)
        point_qkv = attention.qkv(point.feat)[order].reshape(-1, 3, heads, head_dim)
        if point_rope is not None:
            query, key, value = point_qkv.unbind(1)
            query, key = point_rope(query, key, point.coord[order].clone())
            point_qkv = torch.stack((query, key, value), 1)
        action_qkv = attention.qkv(point.action_feat)[routed.sample_ids].reshape(-1, 3, heads, head_dim)
        joint_lengths = lengths + actions + tokens
        joint_cu = torch.cat((lengths.new_zeros(1), joint_lengths.cumsum(0)))
        group_ids = torch.arange(groups, device=order.device)
        point_group = group_ids.repeat_interleave(lengths)
        point_slot = torch.arange(len(order), device=order.device) - point_cu[point_group].long()
        point_indices = joint_cu[point_group] + actions + point_slot
        action_indices = joint_cu[:-1, None] + torch.arange(actions, device=order.device)[None]
        polar_indices = joint_cu[:-1, None] + actions + lengths[:, None] + torch.arange(tokens, device=order.device)[None]
        all_indices = torch.cat((action_indices.flatten(), point_indices, polar_indices.flatten()))
        all_qkv = torch.cat((action_qkv, point_qkv, polar_qkv), 0)
        packed = torch.empty_like(all_qkv).index_copy(0, all_indices, all_qkv)
    with torch.profiler.record_function("polar_bbox/joint_attention"):
        dropout = float(attention.attn_drop.p if hasattr(attention.attn_drop, "p") else attention.attn_drop)
        dropout = dropout if attention.training else 0.0
        if attention.enable_flash:
            if flash_function is None:
                raise RuntimeError("FlashAttention is enabled but unavailable")
            dtype = packed.dtype if packed.dtype in (torch.float16, torch.bfloat16) else torch.float16
            output = flash_function(
                packed.to(dtype), joint_cu.to(torch.int32),
                max_seqlen=int(joint_lengths.max().item()), dropout_p=dropout,
                softmax_scale=attention.scale,
            ).reshape(-1, channels).to(packed.dtype)
        else:
            # Batched reference path, including short patches; no per-group loop.
            maximum = int(joint_lengths.max().item())
            packed_groups = group_ids.repeat_interleave(joint_lengths)
            slots = torch.arange(len(packed), device=order.device) - joint_cu[packed_groups]
            padded = packed.new_zeros((groups, maximum, 3, heads, head_dim))
            padded[packed_groups, slots] = packed
            query, key, value = padded.permute(2, 0, 3, 1, 4).unbind(0)
            if attention.upcast_attention:
                query, key, value = query.float(), key.float(), value.float()
            valid = torch.arange(maximum, device=order.device)[None] < joint_lengths[:, None]
            result = F.scaled_dot_product_attention(
                query, key, value, attn_mask=valid[:, None, None],
                dropout_p=dropout, scale=attention.scale,
            ).transpose(1, 2).reshape(groups, maximum, channels)
            output = result[packed_groups, slots].to(packed.dtype)
    point_output = output[point_indices][inverse]
    action_output = output[action_indices]
    polar_output = output[polar_indices]
    batch = point.action_feat.shape[0]
    action_sum = output.new_zeros((batch, actions, channels)).index_add(0, routed.sample_ids, action_output)
    group_counts = torch.bincount(routed.sample_ids, minlength=batch).to(output.dtype)
    action_mean = action_sum / group_counts[:, None, None].clamp_min(1)
    point.feat = attention.proj_drop(attention.proj(point_output))
    point.action_feat = attention.proj_drop(attention.proj(action_mean))
    # Inspection only. The next block resamples ITS current support; these
    # group outputs are neither reordered into new groups nor written to 2D.
    point.polar_group_output = polar_output
    point.polar_group_boxes = routed.expanded_boxes
    return point


def mask_points_at_pixel_targets(points, counts, pixels, target_mask, pixel_transform=None, source_hw=None):
    """Hold out by saved observation pixel IDs, before support pooling/fusion."""
    if target_mask.ndim != 5 or target_mask.shape[1:3] != (1, 1):
        raise ValueError("BBox holdout requires target_mask [B,1,1,H,W]")
    batch = len(counts)
    torch._assert((counts > 0).all() & (counts.sum() == len(points)),
                  "BBox holdout requires positive point counts matching input rows")
    ids = torch.arange(batch, device=points.device).repeat_interleave(counts.long())
    hw = counts.new_tensor(target_mask.shape[-2:]).expand(batch, 1, 2)
    support = image_support_from_pixels(pixels, ids, hw, pixel_transform, source_hw)
    uv = torch.floor(support[:, :2] + 1e-4).long()
    keep = ~target_mask[ids, 0, 0, uv[:, 1], uv[:, 0]].bool()
    kept_counts = torch.bincount(ids[keep], minlength=batch)
    # Serialization needs >=1 point/sample. If all points were hidden, move
    # one pixel back to the observed set, rather than leaking a held-out point.
    empty = torch.nonzero(kept_counts == 0, as_tuple=False).flatten()
    first = (counts.cumsum(0) - counts)[empty].long()
    keep[first] = True
    target_mask[empty, 0, 0, uv[first, 1], uv[first, 0]] = False
    kept_counts = torch.bincount(ids[keep], minlength=batch).to(counts.dtype)
    return points[keep], kept_counts, keep


def rasterize_inherited_point_features(stage_points, feature_levels, feature_strides, feature_offsets):
    """Return deep point features to their INPUT pixels through pooling lineage."""
    if any(len(values) != 5 for values in (stage_points, feature_levels, feature_strides, feature_offsets)):
        raise ValueError("Depth rasterization requires five stages with explicit feature geometry")
    maps, masks = [], []
    for stage, reference, stride, offset in zip(stage_points, feature_levels, feature_strides, feature_offsets):
        batch, views, _, height, width = reference.shape
        if views != 1:
            raise ValueError("Inherited point rasterization currently supports frontview only")
        uv = stage["input_image_support"][:, :2]
        cells = torch.round((uv - offset) / stride).long()
        # Same border extension as dense grid_sample. Input pixels have
        # already been range checked; don't drop right/bottom-edge pixels
        # when a stride's final feature center lies inside the image edge.
        columns = cells[:, 0].clamp(0, width - 1)
        rows = cells[:, 1].clamp(0, height - 1)
        flat = stage["input_point_batch"].long() * (height * width) + rows * width + columns
        values = stage["feat"][stage["input_to_stage"]]
        channels = values.shape[-1]
        sums = values.new_zeros((batch * height * width, channels)).index_add(0, flat, values)
        counts = values.new_zeros((batch * height * width, 1)).index_add(0, flat, values.new_ones((len(values), 1)))
        dense = (sums / counts.clamp_min(1)).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        mask = (counts > 0).reshape(batch, 1, height, width)
        maps.append(dense[:, None])
        masks.append(mask[:, None])
    if len(maps) != 5:
        raise ValueError("Depth decoder requires five inherited point stages")
    return tuple(maps), tuple(masks)
