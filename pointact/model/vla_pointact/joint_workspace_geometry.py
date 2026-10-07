"""Shared geometry preparation for action training and inference."""
import torch
import torch.nn.functional as F

from .workspace_geometry import filter_normal_consistent_points
from .action_head_3d.polar_depth_self_supervision import normal_anchor_confidence


def native_cga_observation(images, intrinsics):
    """Match archived native-CGA checkpoint: 11 channels, integer-center rays.

    The first channel remains the checkpoint's supplied intensity convention;
    do not silently replace its luminance proxy with native rendered S0.
    """
    batch, views, _, height, width = images.shape
    intensity, dolp, cos2, sin2 = images[:, :, :4].unbind(2)
    q, u = intensity * dolp * cos2, intensity * dolp * sin2
    yy, xx = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=images.dtype),
        torch.arange(width, device=images.device, dtype=images.dtype), indexing="ij")
    k = intrinsics.to(images.dtype)
    x = (xx - k[..., 0, 2, None, None]) / k[..., 0, 0, None, None]
    y = (yy - k[..., 1, 2, None, None]) / k[..., 1, 1, None, None]
    rays = F.normalize(torch.stack((x, y, torch.ones_like(x)), 2), dim=2)
    return torch.cat((torch.stack((
        .5 * (intensity + q), .5 * (intensity + u),
        .5 * (intensity - q), .5 * (intensity - u),
        intensity, cos2, sin2, dolp), 2), rays), 2).reshape(batch * views, 11, height, width)


@torch.no_grad()
def prepare_workspace_points(points, counts, context, depth, valid, normal_targets):
    """Use the same observed-pixel identities in training and inference.

    Keep action targets in their existing model frame: do NOT recenter retained
    points independently of actions. Empty samples fail explicitly, rather than
    silently inserting rejected points or dropping their action labels.
    """
    if depth is None or valid is None:
        raise ValueError("Weighted workspace inference and training require sensor depth and validity")
    if depth.shape != valid.shape or depth.ndim != 5 or depth.shape[1:3] != (1, 1):
        raise ValueError("Expected single-view observed depth [B,1,1,H,W]")
    valid = valid.bool() & torch.isfinite(depth) & (depth > 0)
    valid &= context["polar_workspace_mask"].unsqueeze(2).bool()
    if "pixel_valid" in context:
        valid &= context["pixel_valid"].unsqueeze(2).bool()
    valid &= context["view_valid"][:, :, None, None, None].bool()
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    confidence = normal_anchor_confidence(
        safe_depth[:, 0].float(), context["polar_K"][:, 0].float(),
        normal_targets[:, 0].float(), valid[:, 0], pixel_center_offset=0.0)[:, None]
    confidence = torch.nan_to_num(confidence, nan=0., posinf=0., neginf=0.)
    selected, selected_counts, keep = filter_normal_consistent_points(
        points, counts, context["point_pixel_indices"], confidence)
    updated = dict(context)
    updated["point_pixel_indices"] = context["point_pixel_indices"][keep]
    updated["normal_targets"] = normal_targets.detach()
    updated["observation_confidence"] = confidence.detach()
    return selected, selected_counts, keep, updated
