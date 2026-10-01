"""Geometry routing from serialized PointACT groups to SfP-Wild feature grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class PolarRoute:
    features: Tensor
    xy: Tensor
    view_ids: Tensor
    candidate_count: int
    valid_projection_count: int
    grid_indices: Tensor


def camera_from_model(
    T_camera_from_world: Tensor, T_model_from_world: Tensor
) -> Tensor:
    """Return ``T_camera_from_world @ inverse(T_model_from_world)``."""
    if T_camera_from_world.shape[-2:] != (4, 4) or T_model_from_world.shape[-2:] != (4, 4):
        raise ValueError("Both transforms must end in [4,4]")
    return T_camera_from_world @ torch.linalg.inv(T_model_from_world).unsqueeze(-3)


def model_from_world_z_rotation(angle: Tensor, center_after_rotation: Tensor) -> Tensor:
    """Build the transform for ``x_model = Rz(angle) x_world - center``."""
    angle = torch.as_tensor(angle)
    center_after_rotation = torch.as_tensor(center_after_rotation, device=angle.device, dtype=angle.dtype)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    transform = torch.eye(4, device=angle.device, dtype=angle.dtype).expand(*angle.shape, 4, 4).clone()
    transform[..., 0, 0] = cosine
    transform[..., 0, 1] = -sine
    transform[..., 1, 0] = sine
    transform[..., 1, 1] = cosine
    transform[..., :3, 3] = -center_after_rotation
    return transform


def sfp_feature_geometry(level: int) -> tuple[int, float]:
    """Effective stride and input-pixel center offset of an SfP feature level."""
    if level not in range(5):
        raise ValueError(f"SfP feature level must be in [0,4], got {level}")
    stride = 1 << level
    # 3x3 convolutions use padding=1 and do not move centers. Each 2x2,
    # stride-2 MaxPool changes offset o to o + 0.5 * previous_stride.
    return stride, (stride - 1) / 2.0


def _farthest_2d(points: Tensor, count: int) -> Tensor:
    """Deterministic farthest-point coverage, initialized at the top-left key."""
    if count >= len(points):
        return torch.arange(len(points), device=points.device)
    chosen = [0]
    distance = ((points - points[0]) ** 2).sum(-1)
    for _ in range(1, count):
        index = int(torch.argmax(distance).item())
        chosen.append(index)
        distance = torch.minimum(distance, ((points - points[index]) ** 2).sum(-1))
    return torch.tensor(chosen, device=points.device, dtype=torch.long)


def _balanced_spatial_sample(keys: Tensor, budget: int) -> Tensor:
    if len(keys) <= budget:
        return keys
    views = torch.unique(keys[:, 0], sorted=True)
    allocations = {int(view): budget // len(views) for view in views}
    for view in views[: budget % len(views)]:
        allocations[int(view)] += 1
    selected = []
    leftovers = []
    for view in views:
        view_keys = keys[keys[:, 0] == view]
        quota = min(allocations[int(view)], len(view_keys))
        if quota:
            scale = view_keys[:, 1:].amax(0).clamp_min(1)
            indices = _farthest_2d(view_keys[:, 1:].float() / scale, quota)
            selected.append(view_keys[indices])
            keep = torch.ones(len(view_keys), dtype=torch.bool, device=keys.device)
            keep[indices] = False
            leftovers.append(view_keys[keep])
        else:
            leftovers.append(view_keys)
    result = torch.cat(selected, 0) if selected else keys[:0]
    if len(result) < budget:
        remaining = torch.cat(leftovers, 0)
        result = torch.cat((result, remaining[: budget - len(result)]), 0)
    return result


class PolarTokenRouter:
    """Select sparse feature cells for each existing serialized attention group."""

    def __init__(self, level: int, neighbor_radius: int = 1, max_tokens: int = 32, z_epsilon: float = 1e-6):
        if neighbor_radius < 0 or max_tokens <= 0:
            raise ValueError("neighbor_radius must be >=0 and max_tokens must be >0")
        self.level = level
        self.neighbor_radius = neighbor_radius
        self.max_tokens = max_tokens
        self.z_epsilon = z_epsilon

    def __call__(self, point, order: Tensor, group_lengths: Tensor) -> list[PolarRoute]:
        required = (
            "polar_features", "polar_K", "T_camera_from_model", "view_valid", "polar_image_hw"
        )
        missing = [key for key in required if key not in point]
        if missing:
            raise ValueError(f"Polar routing metadata missing from Point: {missing}")
        features = point.polar_features
        if features.ndim != 5:
            raise ValueError("polar_features must be [B,V,Hf,Wf,C]")
        batch_size, num_views, feat_h, feat_w, channels = features.shape
        if point.polar_K.shape != (batch_size, num_views, 3, 3):
            raise ValueError("polar_K shape does not match the feature bank")
        if point.T_camera_from_model.shape != (batch_size, num_views, 4, 4):
            raise ValueError("T_camera_from_model shape does not match the feature bank")
        if point.view_valid.shape != (batch_size, num_views):
            raise ValueError("view_valid shape does not match the feature bank")

        transforms = point.T_camera_from_model.float()
        intrinsics = point.polar_K.float()
        pixel_affine = point.get("polar_pixel_transform")
        if pixel_affine is not None:
            if pixel_affine.shape != (batch_size, num_views, 3, 3):
                raise ValueError("polar_pixel_transform must be [B,V,3,3]")
            pixel_affine = pixel_affine.float()
        stride, center_offset = sfp_feature_geometry(self.level)
        groups = torch.split(order, group_lengths.detach().cpu().tolist())
        routes = []
        total_unique_points = 0
        total_valid_projections = 0
        total_candidates = 0
        total_selected = 0

        for group_indices in groups:
            unique_indices = torch.unique(group_indices, sorted=True)
            sample_ids = torch.unique(point.batch[unique_indices])
            if len(sample_ids) != 1:
                raise RuntimeError("A serialized attention group crossed sample boundaries")
            sample = int(sample_ids.item())
            coords = point.coord[unique_indices].float()
            total_unique_points += len(coords)
            homogeneous = torch.cat((coords, torch.ones_like(coords[:, :1])), dim=-1)
            keys = []
            valid_for_group = 0
            for view in range(num_views):
                if not bool(point.view_valid[sample, view]):
                    continue
                camera = homogeneous @ transforms[sample, view].transpose(0, 1)
                z = camera[:, 2]
                finite = torch.isfinite(camera).all(-1)
                valid = finite & (z > self.z_epsilon)
                safe_z = z.clamp_min(self.z_epsilon)
                projected = camera[:, :3] @ intrinsics[sample, view].transpose(0, 1)
                uv = projected[:, :2] / safe_z[:, None]
                if pixel_affine is not None:
                    uv_h = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)
                    transformed = uv_h @ pixel_affine[sample, view].transpose(0, 1)
                    uv = transformed[:, :2] / transformed[:, 2:].clamp_min(self.z_epsilon)
                image_h = int(point.polar_image_hw[sample, view, 0].item())
                image_w = int(point.polar_image_hw[sample, view, 1].item())
                valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= image_w - 1)
                valid &= (uv[:, 1] >= 0) & (uv[:, 1] <= image_h - 1)
                pixel_valid = point.get("pixel_valid")
                if pixel_valid is not None:
                    pixel_col = uv[:, 0].round().long().clamp(0, image_w - 1)
                    pixel_row = uv[:, 1].round().long().clamp(0, image_h - 1)
                    valid &= pixel_valid[sample, view, pixel_row, pixel_col].bool()
                valid_for_group += int(valid.sum().item())
                feature_col = ((uv[valid, 0] - center_offset) / stride).round().long()
                feature_row = ((uv[valid, 1] - center_offset) / stride).round().long()
                for delta_row in range(-self.neighbor_radius, self.neighbor_radius + 1):
                    for delta_col in range(-self.neighbor_radius, self.neighbor_radius + 1):
                        row = feature_row + delta_row
                        col = feature_col + delta_col
                        inside = (row >= 0) & (row < feat_h) & (col >= 0) & (col < feat_w)
                        if inside.any():
                            view_col = torch.full_like(row[inside], view)
                            keys.append(torch.stack((view_col, row[inside], col[inside]), dim=-1))
            if keys:
                candidate_keys = torch.unique(torch.cat(keys, 0), sorted=True, dim=0)
                candidate_count = len(candidate_keys)
                selected_keys = _balanced_spatial_sample(candidate_keys, self.max_tokens)
                values = features[
                    sample, selected_keys[:, 0], selected_keys[:, 1], selected_keys[:, 2]
                ]
                xy = torch.stack(
                    (
                        selected_keys[:, 2].float() / max(feat_w - 1, 1) * 2 - 1,
                        selected_keys[:, 1].float() / max(feat_h - 1, 1) * 2 - 1,
                    ),
                    dim=-1,
                ).to(values.dtype)
                views = selected_keys[:, 0]
            else:
                candidate_count = 0
                selected_keys = torch.empty((0, 3), dtype=torch.long, device=features.device)
                values = features.new_empty((0, channels))
                xy = features.new_empty((0, 2))
                views = torch.empty(0, dtype=torch.long, device=features.device)
            total_valid_projections += valid_for_group
            total_candidates += candidate_count
            total_selected += len(values)
            routes.append(
                PolarRoute(values, xy, views, candidate_count, valid_for_group, selected_keys)
            )

        stats = {
            "stage": self.level,
            "groups": len(routes),
            "unique_group_points": total_unique_points,
            "valid_projections": total_valid_projections,
            "candidates": total_candidates,
            "selected": total_selected,
            "truncated": max(total_candidates - total_selected, 0),
        }
        point.setdefault("polar_route_stats", []).append(stats)
        return routes


def save_polar_route_overlay(
    image: Tensor,
    route: PolarRoute,
    level: int,
    path: str | Path,
    view: int = 0,
    projected_uv: Tensor | None = None,
) -> None:
    """Save a calibration overlay (cyan projections, red selected grid centers)."""
    from PIL import Image, ImageDraw

    value = torch.as_tensor(image).detach().cpu()
    if value.ndim != 3 or value.shape[0] not in (1, 3):
        raise ValueError("Overlay image must be [1,H,W] or [3,H,W]")
    if value.shape[0] == 1:
        value = value.expand(3, -1, -1)
    if value.is_floating_point():
        value = value.clamp(0, 1).mul(255)
    canvas = Image.fromarray(value.byte().permute(1, 2, 0).numpy())
    draw = ImageDraw.Draw(canvas)
    if projected_uv is not None:
        for u, v in torch.as_tensor(projected_uv).detach().cpu().tolist():
            draw.ellipse((u - 2, v - 2, u + 2, v + 2), outline=(0, 255, 255), width=1)
    stride, offset = sfp_feature_geometry(level)
    keys = route.grid_indices[route.grid_indices[:, 0] == view].detach().cpu()
    for _, row, col in keys.tolist():
        u, v = col * stride + offset, row * stride + offset
        draw.rectangle((u - 2, v - 2, u + 2, v + 2), outline=(255, 0, 0), width=1)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
