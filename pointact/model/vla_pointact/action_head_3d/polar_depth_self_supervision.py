"""Single-view normal/depth self-supervision for dense metric depth.

Predicted metric depth is differentiated into camera-frame normals and matched
to detached normals from the selected pretrained polar backbone.  Sparse depth
and edge-aware smoothness retain metric scale and regularize unobserved pixels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .polar_router import routed_feature_geometry
from .polar_bbox_fusion import (
    mask_points_at_pixel_targets, rasterize_inherited_point_features,
)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def rasterize_fused_point_features(
    stage_points: tuple[dict[str, Tensor], ...],
    feature_levels: tuple[Tensor, ...],
    intrinsics: Tensor,
    transforms: Tensor,
    image_hw: Tensor,
    view_valid: Tensor,
    pixel_valid: Tensor | None = None,
    pixel_transform: Tensor | None = None,
    feature_strides: tuple[float, ...] | None = None,
    feature_offsets: tuple[float, ...] | None = None,
    correspondence_mode: str = "projection",
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Rasterize each PointACT stage onto its matching polar feature grid.

    Bbox mode uses input pixel support and exact pooling lineage, without
    projecting pooled XYZ. Projection mode is the legacy ablation path.

    Colliding point features are averaged. Projection indices are discrete, but
    the scatter operation remains differentiable with respect to point features,
    which is the gradient path used by the reconstruction loss. Encoder-provided
    stride/offset metadata is used when present; otherwise the legacy SfP-Wild
    max-pooling geometry is retained.
    """
    if len(stage_points) != len(feature_levels):
        raise ValueError("stage_points and feature_levels must have the same length")
    if correspondence_mode in ("bbox", "workspace"):
        if feature_strides is None or feature_offsets is None:
            raise ValueError("Inherited depth rasterization requires feature grid geometry")
        return rasterize_inherited_point_features(
            stage_points, feature_levels, feature_strides, feature_offsets
        )
    if correspondence_mode != "projection":
        raise ValueError(f"Unknown correspondence mode: {correspondence_mode}")
    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must be [B,V,3,3]")
    batch, views = intrinsics.shape[:2]
    if transforms.shape != (batch, views, 4, 4):
        raise ValueError("transforms must be [B,V,4,4]")
    if image_hw.shape != (batch, views, 2):
        raise ValueError("image_hw must be [B,V,2]")
    if view_valid.shape != (batch, views):
        raise ValueError("view_valid must be [B,V]")
    if pixel_transform is not None and pixel_transform.shape != (batch, views, 3, 3):
        raise ValueError("pixel_transform must be [B,V,3,3]")

    geometry = {
        "polar_feature_strides": feature_strides,
        "polar_feature_offsets": feature_offsets,
    }

    maps, masks = [], []
    for level_index, (stage, polar_level) in enumerate(zip(stage_points, feature_levels)):
        point_feat = stage["feat"]
        point_coord = stage["coord"]
        point_batch = stage["batch"].long()
        if polar_level.shape[:2] != (batch, views):
            raise ValueError("Every feature level must be [B,V,C,H,W]")
        feat_h, feat_w = polar_level.shape[-2:]
        stride, center_offset = routed_feature_geometry(geometry, level_index)
        view_maps, view_masks = [], []
        for sample in range(batch):
            sample_points = point_batch == sample
            coords = point_coord[sample_points].float()
            values = point_feat[sample_points]
            homogeneous = torch.cat((coords, torch.ones_like(coords[:, :1])), dim=-1)
            for view in range(views):
                empty_map = values.new_zeros((point_feat.shape[1], feat_h, feat_w))
                empty_mask = torch.zeros((1, feat_h, feat_w), dtype=torch.bool, device=values.device)
                if not bool(view_valid[sample, view]) or coords.numel() == 0:
                    view_maps.append(empty_map)
                    view_masks.append(empty_mask)
                    continue
                camera = homogeneous @ transforms[sample, view].float().transpose(0, 1)
                z = camera[:, 2]
                valid = torch.isfinite(camera).all(-1) & (z > 1e-6)
                projected = camera[:, :3] @ intrinsics[sample, view].float().transpose(0, 1)
                uv = projected[:, :2] / z.clamp_min(1e-6)[:, None]
                if pixel_transform is not None:
                    uv_h = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)
                    transformed = uv_h @ pixel_transform[sample, view].float().transpose(0, 1)
                    uv = transformed[:, :2] / transformed[:, 2:].clamp_min(1e-6)
                height = int(image_hw[sample, view, 0].item())
                width = int(image_hw[sample, view, 1].item())
                valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                valid &= (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
                if pixel_valid is not None:
                    cols = uv[:, 0].round().long().clamp(0, width - 1)
                    rows = uv[:, 1].round().long().clamp(0, height - 1)
                    valid &= pixel_valid[sample, view, rows, cols].bool()
                cols = ((uv[:, 0] - center_offset) / stride).round().long()
                rows = ((uv[:, 1] - center_offset) / stride).round().long()
                valid &= (cols >= 0) & (cols < feat_w) & (rows >= 0) & (rows < feat_h)
                if not valid.any():
                    view_maps.append(empty_map)
                    view_masks.append(empty_mask)
                    continue
                flat_indices = rows[valid] * feat_w + cols[valid]
                sums = values.new_zeros((feat_h * feat_w, point_feat.shape[1])).index_add(
                    0, flat_indices, values[valid]
                )
                counts = values.new_zeros((feat_h * feat_w, 1)).index_add(
                    0, flat_indices, values.new_ones((int(valid.sum().item()), 1))
                )
                mean = sums / counts.clamp_min(1)
                view_maps.append(mean.transpose(0, 1).reshape(point_feat.shape[1], feat_h, feat_w))
                view_masks.append((counts > 0).transpose(0, 1).reshape(1, feat_h, feat_w))
        maps.append(torch.stack(view_maps).reshape(batch, views, point_feat.shape[1], feat_h, feat_w))
        masks.append(torch.stack(view_masks).reshape(batch, views, 1, feat_h, feat_w))
    return tuple(maps), tuple(masks)


def mask_points_at_depth_targets(
    points: Tensor,
    npoints_in_batch: Tensor,
    target_mask: Tensor,
    intrinsics: Tensor,
    transforms: Tensor,
    view_valid: Tensor,
    pixel_transform: Tensor | None = None,
    point_pixel_indices: Tensor | None = None,
    point_pixel_image_hw: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Remove points at held-out pixels using saved IDs, or legacy projection.

    RLBench V2 serializes ``point_pixel_indices`` with ``floor(uv)``.  Use the
    same raster-cell convention here so a held-out sparse-depth pixel cannot
    retain its source point merely because nearest-integer rounding selects a
    neighboring pixel.  The small epsilon matches the dataset round-trip audit
    and only absorbs floating-point drift at exact integer coordinates.
    """
    batch, views = target_mask.shape[:2]
    if target_mask.ndim != 5 or target_mask.shape[2] != 1:
        raise ValueError("target_mask must be [B,V,1,H,W]")
    if len(npoints_in_batch) != batch:
        raise ValueError("npoints_in_batch must match target_mask batch size")
    if int(npoints_in_batch.sum().item()) != len(points):
        raise ValueError("npoints_in_batch must sum to the number of points")
    if intrinsics.shape != (batch, views, 3, 3):
        raise ValueError("intrinsics must be [B,V,3,3]")
    if transforms.shape != (batch, views, 4, 4):
        raise ValueError("transforms must be [B,V,4,4]")
    if view_valid.shape != (batch, views):
        raise ValueError("view_valid must be [B,V]")
    if pixel_transform is not None and pixel_transform.shape != (batch, views, 3, 3):
        raise ValueError("pixel_transform must be [B,V,3,3]")

    if point_pixel_indices is not None:
        return mask_points_at_pixel_targets(
            points, npoints_in_batch, point_pixel_indices, target_mask,
            pixel_transform, point_pixel_image_hw,
        )

    batch_ids = torch.arange(batch, device=points.device).repeat_interleave(
        npoints_in_batch.long()
    )
    keep = torch.ones(len(points), dtype=torch.bool, device=points.device)
    for sample in range(batch):
        point_indices = torch.nonzero(batch_ids == sample, as_tuple=False).flatten()
        coords = points[point_indices, :3].float()
        homogeneous = torch.cat((coords, torch.ones_like(coords[:, :1])), dim=-1)
        for view in range(views):
            if not bool(view_valid[sample, view]) or not len(point_indices):
                continue
            camera = homogeneous @ transforms[sample, view].float().transpose(0, 1)
            z = camera[:, 2]
            valid = torch.isfinite(camera).all(-1) & (z > 1e-6)
            projected = camera[:, :3] @ intrinsics[sample, view].float().transpose(0, 1)
            uv = projected[:, :2] / z.clamp_min(1e-6)[:, None]
            if pixel_transform is not None:
                uv_h = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)
                transformed = uv_h @ pixel_transform[sample, view].float().transpose(0, 1)
                uv = transformed[:, :2] / transformed[:, 2:].clamp_min(1e-6)
            height, width = target_mask.shape[-2:]
            pixel_indices = torch.floor(uv + 1e-4).long()
            cols = pixel_indices[:, 0]
            rows = pixel_indices[:, 1]
            valid &= (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            safe_cols = cols.clamp(0, width - 1)
            safe_rows = rows.clamp(0, height - 1)
            remove = valid & target_mask[sample, view, 0, safe_rows, safe_cols].bool()
            keep[point_indices[remove]] = False
        # PointACT cannot serialize an empty sample. Keep one point in the
        # unlikely event that every point landed on a held-out target.
        if not keep[point_indices].any() and len(point_indices):
            keep[point_indices[0]] = True

    kept_batch_ids = batch_ids[keep]
    kept_counts = torch.bincount(kept_batch_ids, minlength=batch).to(npoints_in_batch.dtype)
    return points[keep], kept_counts, keep


class PolarPointDepthDecoder(nn.Module):
    """Fuse SfP-Wild image features with rasterized PointACT fused features."""

    def __init__(
        self,
        feature_channels: tuple[int, ...] = (64, 128, 256, 512, 512),
        point_feature_channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        min_depth: float = 0.05,
        max_depth: float = 4.5,
        use_polar_features: bool = True,
    ):
        super().__init__()
        if len(feature_channels) != 5 or len(point_feature_channels) != 5:
            raise ValueError("feature channels must describe five encoder stages")
        if not 0 < min_depth < max_depth:
            raise ValueError("Expected 0 < min_depth < max_depth")
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.use_polar_features = bool(use_polar_features)

        point_widths = (32, 32, 64, 64, 64)
        self.point_projections = nn.ModuleList(
            nn.Conv2d(source, target, kernel_size=1, bias=False)
            for source, target in zip(point_feature_channels, point_widths)
        )

        c1, c2, c3, c4, c5 = feature_channels
        polar_widths = (c1, c2, c3, c4, c5) if self.use_polar_features else (0, 0, 0, 0, 0)
        q1, q2, q3, q4, q5 = polar_widths
        self.fuse5 = _ConvBlock(q5 + 64 + 1, 256)
        self.fuse4 = _ConvBlock(256 + q4 + 64 + 1, 256)
        self.fuse3 = _ConvBlock(256 + q3 + 64 + 1, 128)
        self.fuse2 = _ConvBlock(128 + q2 + 32 + 1, 64)
        self.fuse1 = _ConvBlock(64 + q1 + 32 + 1, 32)
        self.depth_head = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _up_to(x: Tensor, reference: Tensor) -> Tensor:
        return F.interpolate(
            x, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self,
        feature_levels: tuple[Tensor, ...] | None,
        point_feature_levels: tuple[Tensor, ...],
        point_valid_levels: tuple[Tensor, ...],
        output_size: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return dense metric depth and the final dense completion feature."""
        if not (len(point_feature_levels) == len(point_valid_levels) == 5):
            raise ValueError("Point inputs must contain five feature levels")
        if self.use_polar_features:
            if feature_levels is None or len(feature_levels) != 5:
                raise ValueError("Polar-enabled decoder requires five feature levels")
            x1, x2, x3, x4, x5 = feature_levels
        elif feature_levels is not None:
            raise ValueError("Point-only decoder must receive feature_levels=None")
        point_levels = tuple(
            projection(points * valid.to(points.dtype))
            for projection, points, valid in zip(
                self.point_projections, point_feature_levels, point_valid_levels
            )
        )
        p1, p2, p3, p4, p5 = point_levels
        m1, m2, m3, m4, m5 = (mask.to(p1.dtype) for mask in point_valid_levels)

        def inputs(*values):
            return torch.cat(values, dim=1)

        decoded = self.fuse5(inputs(*((x5,) if self.use_polar_features else ()), p5, m5))
        decoded = self._up_to(decoded, p4)
        decoded = self.fuse4(inputs(decoded, *((x4,) if self.use_polar_features else ()), p4, m4))
        decoded = self._up_to(decoded, p3)
        decoded = self.fuse3(inputs(decoded, *((x3,) if self.use_polar_features else ()), p3, m3))
        decoded = self._up_to(decoded, p2)
        decoded = self.fuse2(inputs(decoded, *((x2,) if self.use_polar_features else ()), p2, m2))
        decoded = self._up_to(decoded, p1)
        decoded = self.fuse1(inputs(decoded, *((x1,) if self.use_polar_features else ()), p1, m1))

        # A bounded parameterization prevents invalid/negative geometry during
        # early joint training; held-out observed-depth targets provide scale.
        unit_depth = torch.sigmoid(self.depth_head(decoded))
        depth = self.min_depth + (self.max_depth - self.min_depth) * unit_depth
        if output_size is not None and depth.shape[-2:] != output_size:
            depth = F.interpolate(depth, size=output_size, mode="bilinear", align_corners=False)
        return depth, decoded


def depth_to_camera_points(depth: Tensor, intrinsics: Tensor) -> Tensor:
    """Back-project ``[N,1,H,W]`` depth to OpenCV camera points ``[N,3,H,W]``."""
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must be [N,1,H,W]")
    if intrinsics.shape != (depth.shape[0], 3, 3):
        raise ValueError("intrinsics must be [N,3,3]")
    n, _, height, width = depth.shape
    rows, cols = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    cols = cols.expand(n, -1, -1)
    rows = rows.expand(n, -1, -1)
    z = depth[:, 0]
    x = (cols - intrinsics[:, 0, 2, None, None]) * z / intrinsics[:, 0, 0, None, None]
    y = (rows - intrinsics[:, 1, 2, None, None]) * z / intrinsics[:, 1, 1, None, None]
    return torch.stack((x, y, z), dim=1)


def depth_to_normals(depth: Tensor, intrinsics: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Compute camera-facing normals and an interior finite-difference mask."""
    points = depth_to_camera_points(depth, intrinsics)
    du = points.new_zeros(points.shape)
    dv = points.new_zeros(points.shape)
    du[:, :, :, 1:-1] = points[:, :, :, 2:] - points[:, :, :, :-2]
    dv[:, :, 1:-1, :] = points[:, :, 2:, :] - points[:, :, :-2, :]
    normals = F.normalize(torch.cross(du, dv, dim=1), dim=1, eps=eps)
    view_to_camera = F.normalize(-points, dim=1, eps=eps)
    orientation = torch.where(
        (normals * view_to_camera).sum(dim=1, keepdim=True) < 0,
        -torch.ones_like(normals[:, :1]),
        torch.ones_like(normals[:, :1]),
    )
    normals = normals * orientation
    valid = torch.zeros_like(depth, dtype=torch.bool)
    valid[:, :, 1:-1, 1:-1] = True
    valid &= torch.linalg.vector_norm(du, dim=1, keepdim=True) > eps
    valid &= torch.linalg.vector_norm(dv, dim=1, keepdim=True) > eps
    return normals, valid


def _masked_mean(values: Tensor, weights: Tensor) -> Tensor:
    weights = weights.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class PolarDepthSelfSupervision(nn.Module):
    """Decode dense depth and apply normal, sparse-depth, and smoothness losses."""

    def __init__(
        self,
        feature_channels: tuple[int, ...] = (64, 128, 256, 512, 512),
        point_feature_channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        min_depth: float = 0.05,
        max_depth: float = 4.5,
        depth_keep_probability: float = 0.7,
        normal_weight: float = 1.0,
        sparse_depth_weight: float = 1.0,
        smoothness_weight: float = 0.01,
        use_polar_features: bool = True,
    ):
        super().__init__()
        if not 0 < depth_keep_probability < 1:
            raise ValueError("depth_keep_probability must be in (0,1)")
        self.decoder = PolarPointDepthDecoder(
            feature_channels, point_feature_channels, min_depth, max_depth,
            use_polar_features=use_polar_features,
        )
        self.depth_keep_probability = float(depth_keep_probability)
        self.normal_weight = float(normal_weight)
        self.sparse_depth_weight = float(sparse_depth_weight)
        self.smoothness_weight = float(smoothness_weight)

    @staticmethod
    def _cauchy(x: Tensor, scale: float = 0.1) -> Tensor:
        """Bound the influence of corrupted observed-depth outliers."""
        return 0.5 * scale * torch.log1p((x / scale).square())

    def forward(
        self,
        feature_levels: tuple[Tensor, ...] | None,
        point_feature_levels: tuple[Tensor, ...],
        point_valid_levels: tuple[Tensor, ...],
        polar_images: Tensor,
        intrinsics: Tensor,
        observed_depth: Tensor | None = None,
        observed_depth_valid: Tensor | None = None,
        pixel_valid: Tensor | None = None,
        view_valid: Tensor | None = None,
        compute_loss: bool = True,
        depth_supervision_mask: Tensor | None = None,
        normal_targets: Tensor | None = None,
        sfp_normals: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if polar_images.ndim != 5 or polar_images.shape[2] != 7:
            raise ValueError("polar_images must be [B,V,7,H,W]")
        batch, views, _, height, width = polar_images.shape
        expected_depth_shape = (batch, views, 1, height, width)
        if intrinsics.shape != (batch, views, 3, 3):
            raise ValueError("intrinsics must be [B,V,3,3]")
        if compute_loss:
            if observed_depth is None or observed_depth.shape != expected_depth_shape:
                raise ValueError(f"observed_depth must have shape {expected_depth_shape}")
            if observed_depth_valid is None or observed_depth_valid.shape != expected_depth_shape:
                raise ValueError("observed_depth_valid must match observed_depth")
            expected_normal_shape = (batch, views, 3, height, width)
            if normal_targets is not None and sfp_normals is not None:
                raise ValueError("Provide normal_targets or sfp_normals, not both")
            if normal_targets is None:
                normal_targets = sfp_normals
            if normal_targets is None or normal_targets.shape != expected_normal_shape:
                raise ValueError(
                    f"normal_targets must have shape {expected_normal_shape}"
                )

        flat_levels = None
        if feature_levels is not None:
            flat_levels = tuple(
                level.reshape(batch * views, *level.shape[2:]) for level in feature_levels
            )
        flat_point_levels = tuple(
            level.reshape(batch * views, *level.shape[2:])
            for level in point_feature_levels
        )
        flat_point_valid = tuple(
            valid.reshape(batch * views, *valid.shape[2:]).bool()
            for valid in point_valid_levels
        )

        sparse = sparse_valid = depth_supervision_mask = None
        if compute_loss:
            sparse = observed_depth.reshape(batch * views, 1, height, width).float()
            sparse_valid = observed_depth_valid.reshape(batch * views, 1, height, width).bool()
            # Hide a random subset of projected point features at observed
            # target pixels. The depth image itself is never a decoder input.
            if self.training:
                if depth_supervision_mask is None:
                    keep = (torch.rand_like(sparse) < self.depth_keep_probability) | ~sparse_valid
                    depth_supervision_mask = sparse_valid & ~keep
                else:
                    depth_supervision_mask = depth_supervision_mask.reshape_as(sparse).bool()
                masked_levels, masked_valid = [], []
                for points, valid in zip(flat_point_levels, flat_point_valid):
                    # Mask a coarse cell if any held-out target falls inside
                    # its receptive region, avoiding a multiscale copy path.
                    hidden_level = F.adaptive_max_pool2d(
                        depth_supervision_mask.float(), points.shape[-2:]
                    ).bool()
                    keep_level = ~hidden_level
                    masked_levels.append(points * keep_level.to(points.dtype))
                    masked_valid.append(valid & keep_level)
                flat_point_levels = tuple(masked_levels)
                flat_point_valid = tuple(masked_valid)
            else:
                depth_supervision_mask = sparse_valid

        prediction, decoded = self.decoder(
            flat_levels,
            flat_point_levels,
            flat_point_valid,
            output_size=(height, width),
        )
        token_mask = torch.ones_like(decoded[:, :1], dtype=torch.bool)
        if pixel_valid is not None:
            token_mask &= F.interpolate(
                pixel_valid.reshape(batch * views, 1, height, width).float(),
                size=decoded.shape[-2:], mode="nearest",
            ).bool()
        if view_valid is not None:
            token_mask &= view_valid.reshape(batch * views, 1, 1, 1).bool()
        pooled = (decoded * token_mask.to(decoded.dtype)).sum((-2, -1))
        pooled = pooled / token_mask.sum((-2, -1)).clamp_min(1).to(decoded.dtype)
        completion_token = pooled.reshape(batch, views, -1)
        if view_valid is None:
            completion_token = completion_token.mean(1)
        else:
            weights = view_valid.to(decoded.dtype).unsqueeze(-1)
            completion_token = (completion_token * weights).sum(1) / weights.sum(1).clamp_min(1)

        output = {
            "predicted_depth": prediction.reshape(batch, views, 1, height, width),
            "completion_token": completion_token,
        }
        if not compute_loss:
            return output

        flat_k = intrinsics.reshape(batch * views, 3, 3).to(prediction.dtype)
        normals, normal_valid = depth_to_normals(prediction, flat_k)
        polar = polar_images.reshape(batch * views, 7, height, width).to(prediction.dtype)
        target_normals = normal_targets.detach().reshape(
            batch * views, 3, height, width
        ).to(prediction.dtype)
        target_magnitude = torch.linalg.vector_norm(
            target_normals, dim=1, keepdim=True
        )
        target_valid = torch.isfinite(target_normals).all(dim=1, keepdim=True)
        target_valid &= target_magnitude > 1e-6
        target_normals = torch.where(
            target_valid, target_normals, torch.zeros_like(target_normals)
        )
        target_normals = F.normalize(target_normals, dim=1, eps=1e-6)

        # Polar pseudo-targets use the shared +left,+down,+forward comparison
        # frame while calibrated pinhole geometry uses +right,+down,+forward.
        predicted_sfp_normals = torch.cat(
            (-normals[:, 0:1], normals[:, 1:]), dim=1
        )
        cosine = (predicted_sfp_normals * target_normals).sum(
            dim=1, keepdim=True
        ).clamp(-1.0, 1.0)
        normal_error = 1.0 - cosine
        normal_mask = normal_valid & target_valid
        if pixel_valid is not None:
            normal_mask &= pixel_valid.reshape(batch * views, 1, height, width).bool()
        if view_valid is not None:
            normal_mask &= view_valid.reshape(batch * views, 1, 1, 1).bool()
        normal_consistency_loss = _masked_mean(normal_error, normal_mask)

        depth_mask = (
            depth_supervision_mask & torch.isfinite(sparse) & (sparse > 0)
        )
        log_depth_error = torch.log(prediction.clamp_min(1e-6)) - torch.log(sparse.clamp_min(1e-6))
        sparse_depth_loss = _masked_mean(self._cauchy(log_depth_error), depth_mask)

        inverse_depth = prediction.reciprocal()
        image = polar[:, 0:1]
        dx_depth = (inverse_depth[:, :, :, 1:] - inverse_depth[:, :, :, :-1]).abs()
        dy_depth = (inverse_depth[:, :, 1:, :] - inverse_depth[:, :, :-1, :]).abs()
        dx_image = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()
        dy_image = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs()
        smoothness_loss = (
            (dx_depth * torch.exp(-10.0 * dx_image)).mean()
            + (dy_depth * torch.exp(-10.0 * dy_image)).mean()
        )

        total = (
            self.normal_weight * normal_consistency_loss
            + self.sparse_depth_weight * sparse_depth_loss
            + self.smoothness_weight * smoothness_loss
        )
        output.update({
            "loss": total,
            "normal_consistency_loss": normal_consistency_loss,
            "sparse_depth_loss": sparse_depth_loss,
            "smoothness_loss": smoothness_loss,
            "predicted_normals": normals.reshape(batch, views, 3, height, width),
            "normal_targets": target_normals.reshape(batch, views, 3, height, width),
            # Backward-compatible alias retained for existing logs/checkpoints.
            "sfp_normals": target_normals.reshape(batch, views, 3, height, width),
        })
        return output
