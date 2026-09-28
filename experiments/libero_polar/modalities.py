"""Aligned RGB-D, polarization, and incomplete point-cloud construction.

This module has no LIBERO imports.  Keeping the numerical transformations here
allows them to be unit tested without starting MuJoCo and keeps the official
LIBERO checkout untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def polar_physics_report(polar: dict, tolerance: float = 2e-5) -> dict:
    """Validate Stokes identities and physical realizability constraints."""
    names = ("S0", "S1", "S2", "S3")
    stokes = [np.asarray(polar[name], dtype=np.float64) for name in names]
    if len({array.shape for array in stokes}) != 1 or stokes[0].ndim != 3 or stokes[0].shape[-1] != 3:
        raise ValueError("S0..S3 must share shape HxWx3")
    valid = np.asarray(polar["valid_mask"], dtype=bool)
    angle_valid = np.asarray(polar["AoLP_valid_mask"], dtype=bool)
    if valid.shape != stokes[0].shape[:2] or angle_valid.shape != valid.shape:
        raise ValueError("Polar validity masks must match the Stokes image")
    finite = np.logical_and.reduce([np.isfinite(array).all(axis=-1) for array in stokes])
    if not finite.all():
        raise ValueError(f"Polar render contains {np.count_nonzero(~finite)} nonfinite pixels")
    s0, s1, s2, s3 = stokes
    intensity, q, u, circular = [array @ LUMA.astype(np.float64)
                                 for array in (s0, s1, s2, s3)]
    dolp = np.asarray(polar["DoLP"], dtype=np.float64)
    aolp = np.asarray(polar["AoLP"], dtype=np.float64)
    cos2 = np.asarray(polar["cos2AoLP"], dtype=np.float64)
    sin2 = np.asarray(polar["sin2AoLP"], dtype=np.float64)
    if not all(array.shape == valid.shape for array in (dolp, aolp, cos2, sin2)):
        raise ValueError("Derived polar maps must share the HxW pixel grid")
    if not all(np.isfinite(array).all() for array in (dolp, aolp, cos2, sin2)):
        raise ValueError("Derived polar maps contain nonfinite values")

    reconstructed = np.divide(np.hypot(q, u), intensity,
                              out=np.zeros_like(intensity), where=intensity > 1e-8)
    dolp_error = np.abs(dolp[valid] - reconstructed[valid])
    # Raw Stokes RGB is a signed linear-sRGB spectral projection, not three
    # independent monochromatic Stokes measurements. Physical cone/analyzer
    # constraints therefore apply to the nonnegative luminance projection used
    # by the renderer's own DoLP definition, not independently to R/G/B.
    degree = np.divide(
        np.sqrt(q*q + u*u + circular*circular), intensity,
        out=np.zeros_like(intensity), where=intensity > 1e-8)
    analyzer_values = np.stack((0.5*(intensity+q), 0.5*(intensity-q),
                                0.5*(intensity+u), 0.5*(intensity-u)), axis=-1)
    analyzer_min = float(analyzer_values[valid].min()) if valid.any() else 0.0
    unit_error = np.abs(cos2[angle_valid]**2 + sin2[angle_valid]**2 - 1.0)
    report = {
        "valid_pixels": int(valid.sum()),
        "aolp_valid_pixels": int(angle_valid.sum()),
        "dolp_min": float(dolp[valid].min()) if valid.any() else 0.0,
        "dolp_max": float(dolp[valid].max()) if valid.any() else 0.0,
        "dolp_reconstruction_max_abs_error": float(dolp_error.max()) if len(dolp_error) else 0.0,
        "stokes_luminance_degree_max": float(degree[valid].max()) if valid.any() else 0.0,
        "stokes_luminance_cone_violation_pixels": int(np.count_nonzero(degree[valid] > 1+tolerance)),
        "analyzer_intensity_min": analyzer_min,
        "aolp_double_angle_unit_max_error": float(unit_error.max()) if len(unit_error) else 0.0,
        "circular_stokes_luma_abs_max": float(np.abs(circular[valid]).max()) if valid.any() else 0.0,
        "signed_linear_rgb_negative_values": int(sum(np.count_nonzero(array < 0)
                                                        for array in stokes)),
    }
    failures = []
    if not valid.any():
        failures.append("no positive-intensity polar pixels")
    if valid.any() and (report["dolp_min"] < -tolerance or report["dolp_max"] > 1+tolerance):
        failures.append("DoLP outside [0,1]")
    if report["dolp_reconstruction_max_abs_error"] > tolerance:
        failures.append("DoLP disagrees with luminance Stokes components")
    if report["stokes_luminance_cone_violation_pixels"]:
        failures.append("S0^2 < S1^2+S2^2+S3^2")
    if report["analyzer_intensity_min"] < -tolerance:
        failures.append("negative ideal-analyzer intensity")
    if report["aolp_double_angle_unit_max_error"] > tolerance:
        failures.append("AoLP doubled-angle encoding is not unit length")
    report["passed"] = not failures
    report["failures"] = failures
    if failures:
        raise ValueError("Polar physics validation failed: " + "; ".join(failures))
    return report


def polar_region_report(polar: dict, geom_ids: np.ndarray,
                        geom_names: dict[int, str]) -> dict:
    """Summarize polarization on task/material regions using MuJoCo IDs."""
    regions = {
        "target_black_bowl": ("akita_black_bowl_1",),
        "distractor_black_bowl": ("akita_black_bowl_2",),
        "plate": ("plate_1",),
        "ramekin": ("glazed_rim_porcelain_ramekin",),
        "cookie_box": ("cookies_1",),
        "wooden_cabinet": ("wooden_cabinet",),
        "table": ("table",),
        "robot_arm": ("robot0_",),
        "gripper": ("gripper0_",),
        "stove": ("flat_stove",),
    }
    ids = np.asarray(geom_ids, dtype=np.int32)
    valid = np.asarray(polar["valid_mask"], dtype=bool)
    angle_valid = np.asarray(polar["AoLP_valid_mask"], dtype=bool)
    dolp = np.asarray(polar["DoLP"], dtype=np.float32)
    result = {}
    for label, tokens in regions.items():
        matching = [geom_id for geom_id, name in geom_names.items()
                    if any(token in name.lower() for token in tokens)]
        mask = np.isin(ids, matching) & valid
        values = dolp[mask]
        result[label] = {
            "pixels": int(mask.sum()),
            "aolp_valid_fraction": (float((angle_valid & mask).sum() / mask.sum())
                                    if mask.any() else None),
            "dolp_mean": float(values.mean()) if len(values) else None,
            "dolp_median": float(np.median(values)) if len(values) else None,
            "dolp_p95": float(np.quantile(values, .95)) if len(values) else None,
            "dolp_max": float(values.max()) if len(values) else None,
        }
    return result


def unproject_depth(
    depth_m: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    """Unproject an OpenCV-style z-depth image to an HxWx3 world cloud."""
    depth = np.asarray(depth_m, dtype=np.float32)
    k = np.asarray(intrinsics, dtype=np.float32)
    pose = np.asarray(camera_to_world, dtype=np.float32)
    if depth.ndim != 2 or k.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("Expected depth HxW, intrinsics 3x3, and pose 4x4")
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float32)
    camera = np.stack(
        ((u + 0.5 - k[0, 2]) * depth / k[0, 0],
         (v + 0.5 - k[1, 2]) * depth / k[1, 1], depth),
        axis=-1,
    )
    return camera @ pose[:3, :3].T + pose[:3, 3]


def depth_normals(depth_m: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate camera-frame normals from organized z depth."""
    depth = np.asarray(depth_m, dtype=np.float32)
    identity = np.eye(4, dtype=np.float32)
    points = unproject_depth(depth, intrinsics, identity)
    dx = np.empty_like(points)
    dy = np.empty_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dx[:, 0] = points[:, 1] - points[:, 0]
    dx[:, -1] = points[:, -1] - points[:, -2]
    dy[1:-1] = points[2:] - points[:-2]
    dy[0] = points[1] - points[0]
    dy[-1] = points[-1] - points[-2]
    normals = np.cross(dx, dy)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(normals).all(axis=-1)
    valid &= (depth > 0.01) & (length[..., 0] > 1e-8)
    normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 1e-8)
    # Orient normals toward the camera.
    view = -points
    flip = np.sum(normals * view, axis=-1) < 0
    normals[flip] *= -1
    return normals.astype(np.float32), valid


def analytic_polarization(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    geom_ids: np.ndarray,
    geom_names: dict[int, str],
    *,
    default_ior: float = 1.50,
) -> dict[str, np.ndarray | dict]:
    """Return a deterministic, screen-space Fresnel polarization proxy.

    This is useful for pipeline tests and ablations.  It is deliberately
    labelled ``analytic-screen-space`` and must not be reported as a polarized
    path-traced sensor.  Use :class:`MujocoNativePolarRenderer` for that.
    """
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    depth = np.asarray(depth_m, dtype=np.float32)
    ids = np.asarray(geom_ids, dtype=np.int32)
    normals, valid = depth_normals(depth, intrinsics)
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float32)
    rays = np.stack(
        ((u + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0],
         (v + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1],
         np.ones_like(depth)), axis=-1)
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-8)
    incoming = -rays
    cos_i = np.clip(np.abs(np.sum(normals * incoming, axis=-1)), 0.0, 1.0)

    # Fresnel reflection difference supplies DoLP. Roughness dampens it.  The
    # name rules are explicit priors, not measured LIBERO material properties.
    eta = np.full(depth.shape, default_ior, dtype=np.float32)
    roughness = np.full(depth.shape, 0.30, dtype=np.float32)
    for geom_id in np.unique(ids):
        name = geom_names.get(int(geom_id), "").lower()
        mask = ids == geom_id
        if any(token in name for token in ("robot", "gripper", "metal")):
            eta[mask], roughness[mask] = 2.5, 0.18
        elif "akita_black_bowl_1" in name:
            eta[mask], roughness[mask] = 1.49, 0.08
        elif "plate_1" in name or "porcelain" in name:
            eta[mask], roughness[mask] = 1.52, 0.12
        elif "table" in name or "wood" in name:
            eta[mask], roughness[mask] = 1.48, 0.32
    sin_t2 = np.square(1.0 / eta) * np.maximum(0.0, 1.0 - cos_i**2)
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin_t2))
    rs = np.square((cos_i - eta * cos_t) / np.maximum(cos_i + eta * cos_t, 1e-7))
    rp = np.square((eta * cos_i - cos_t) / np.maximum(eta * cos_i + cos_t, 1e-7))
    dolp = np.abs(rs - rp) / np.maximum(rs + rp, 1e-7)
    dolp *= np.exp(-2.5 * roughness)
    dolp = np.clip(dolp, 0.0, 1.0)

    # The projected surface normal gives the polarization axis. Doubled-angle
    # channels avoid the pi-periodic AoLP discontinuity.
    aolp = np.arctan2(normals[..., 1], normals[..., 0]) + np.pi / 2
    valid &= np.isfinite(dolp) & (ids >= 0)
    dolp = np.where(valid, dolp, 0).astype(np.float32)
    cos2 = np.where(valid, np.cos(2 * aolp), 0).astype(np.float32)
    sin2 = np.where(valid, np.sin(2 * aolp), 0).astype(np.float32)
    intensity = image @ LUMA
    s1_scalar = intensity * dolp * cos2
    s2_scalar = intensity * dolp * sin2
    scale = np.divide(image, intensity[..., None], out=np.zeros_like(image),
                      where=intensity[..., None] > 1e-6)
    return {
        "S0": image.astype(np.float32),
        "S1": (s1_scalar[..., None] * scale).astype(np.float32),
        "S2": (s2_scalar[..., None] * scale).astype(np.float32),
        "S3": np.zeros_like(image, dtype=np.float32),
        "DoLP": dolp,
        "AoLP": np.where(valid, aolp, 0).astype(np.float32),
        "cos2AoLP": cos2,
        "sin2AoLP": sin2,
        "valid_mask": valid,
        "AoLP_valid_mask": valid & (dolp > 1e-6),
        "metadata": {
            "backend": "analytic-screen-space",
            "warning": "Fresnel proxy from MuJoCo depth normals; not a polarized path tracer",
        },
    }


def voxel_indices(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Choose one stable source row per occupied voxel."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    cells = np.floor(np.asarray(xyz, dtype=np.float64) / voxel_size).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    return np.sort(first)


@dataclass
class CorruptionResult:
    cloud: np.ndarray
    source_pixels: np.ndarray
    current_pixels: np.ndarray
    codes: np.ndarray
    stats: dict


@dataclass
class CompletionResult:
    cloud: np.ndarray
    source_pixels: np.ndarray
    current_pixels: np.ndarray
    synthetic_mask: np.ndarray
    codes: np.ndarray
    stats: dict


@dataclass
class InteractionSupervision:
    target_points: np.ndarray
    input_mask: np.ndarray
    stats: dict


def _cross_kernel(size: int) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.uint8)
    kernel[size // 2, :] = 1
    kernel[:, size // 2] = 1
    return kernel


def fill_depth_multiscale(sparse_depth: np.ndarray) -> np.ndarray:
    """HouseCat/RLBench-style deterministic morphology on sparse metric depth."""
    import cv2

    depth = np.asarray(sparse_depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D sparse depth, got {depth.shape}")
    observed = np.isfinite(depth) & (depth > 0.01)
    result = np.zeros_like(depth)
    if not observed.any():
        return result
    max_depth = max(3.0, float(depth[observed].max()) + 1.0)
    inverted = np.zeros_like(depth)
    inverted[observed] = max_depth - depth[observed]
    bands = (
        (observed & (depth > 2.0), _cross_kernel(3)),
        (observed & (depth > 1.0) & (depth <= 2.0), _cross_kernel(5)),
        (observed & (depth <= 1.0), _cross_kernel(7)),
    )
    dilated = inverted.copy()
    for band, kernel in bands:
        values = cv2.dilate(np.where(band, inverted, 0.0), kernel)
        dilated[values > 0.01] = values[values > 0.01]
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    blurred = cv2.medianBlur(closed, 5)
    closed[closed > 0.01] = blurred[closed > 0.01]
    top_rows = np.argmax(closed > 0.01, axis=0)
    fill_region = np.arange(depth.shape[0])[:, None] >= top_rows[None, :]
    large = cv2.dilate(closed, np.ones((9, 9), np.uint8))
    empty = (closed <= 0.01) & fill_region
    closed[empty] = large[empty]
    for _ in range(6):
        empty = (closed <= 0.01) & fill_region
        large = cv2.dilate(closed, np.ones((5, 5), np.uint8))
        closed[empty] = large[empty]
    valid = (closed > 0.01) & fill_region
    median = cv2.medianBlur(closed, 5)
    closed[valid] = median[valid]
    bilateral = cv2.bilateralFilter(closed, 5, 0.5, 2.0)
    closed[valid] = bilateral[valid]
    valid = closed > 0.01
    result[valid] = max_depth - closed[valid]
    result[observed] = depth[observed]
    result[~np.isfinite(result) | (result <= 0.01)] = 0
    return result


def complete_corrupted_cloud(
    corrupted: CorruptionResult,
    clean_pixel_indices: np.ndarray,
    rgb: np.ndarray,
    polar: dict,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> CompletionResult:
    """Fill known corruption holes and append aligned synthetic 9-D points.

    Candidate holes are clean voxel-source pixels absent from the corrupted
    cloud. This reproduces the RLBench filled9 offline dataset semantics; clean
    provenance is training-data generation information, not available online.
    """
    cloud = np.asarray(corrupted.cloud, dtype=np.float32)
    source = np.asarray(corrupted.source_pixels, dtype=np.int32)
    current = np.asarray(corrupted.current_pixels, dtype=np.int32)
    image = np.asarray(rgb)
    height, width = image.shape[:2]
    holes = np.setdiff1d(np.asarray(clean_pixel_indices, dtype=np.int32), source)
    pose = np.asarray(camera_to_world, dtype=np.float32)
    k = np.asarray(intrinsics, dtype=np.float32)
    camera_xyz = (cloud[:, :3] - pose[:3, 3]) @ pose[:3, :3]
    z = camera_xyz[:, 2]
    sparse = np.full(height * width, np.inf, dtype=np.float32)
    good = np.isfinite(z) & (z > 0.01)
    np.minimum.at(sparse, current[good], z[good])
    sparse[~np.isfinite(sparse)] = 0
    sparse = sparse.reshape(height, width)
    estimated = fill_depth_multiscale(sparse)
    sparse_flat, estimated_flat = sparse.reshape(-1), estimated.reshape(-1)
    new_pixels = holes[(sparse_flat[holes] <= 0.01) & (estimated_flat[holes] > 0.01)]

    if len(new_pixels):
        z_new = estimated_flat[new_pixels]
        u = new_pixels % width + 0.5
        v = new_pixels // width + 0.5
        camera_new = np.column_stack((
            (u-k[0, 2])*z_new/k[0, 0],
            (v-k[1, 2])*z_new/k[1, 1], z_new,
        )).astype(np.float32)
        world_new = camera_new @ pose[:3, :3].T + pose[:3, 3]
        flat_rgb = image.reshape(-1, 3).astype(np.float32) / 255.0
        pvalid = np.asarray(polar["valid_mask"], dtype=bool).reshape(-1)[new_pixels]
        dolp = np.asarray(polar["DoLP"], dtype=np.float32).reshape(-1)[new_pixels]
        cos2 = np.asarray(polar["cos2AoLP"], dtype=np.float32).reshape(-1)[new_pixels]
        sin2 = np.asarray(polar["sin2AoLP"], dtype=np.float32).reshape(-1)[new_pixels]
        new_rows = np.column_stack((
            world_new, flat_rgb[new_pixels],
            np.where(pvalid, dolp, 0), np.where(pvalid, cos2, 0),
            np.where(pvalid, sin2, 0),
        )).astype(np.float32)
    else:
        new_rows = np.empty((0, 9), dtype=np.float32)
    synthetic = np.concatenate((np.zeros(len(cloud), dtype=bool),
                                np.ones(len(new_rows), dtype=bool)))
    return CompletionResult(
        cloud=np.ascontiguousarray(np.concatenate((cloud, new_rows))),
        source_pixels=np.concatenate((source, np.full(len(new_rows), -1, np.int32))),
        current_pixels=np.concatenate((current, new_pixels.astype(np.int32))),
        synthetic_mask=synthetic,
        codes=np.concatenate((corrupted.codes, np.full(len(new_rows), 7, np.uint8))),
        stats={
            "candidate_hole_pixels": int(len(holes)),
            "filled_points": int(len(new_rows)),
            "input_points": int(len(cloud)),
            "output_points": int(len(cloud)+len(new_rows)),
            "method": "multiscale_morphology_on_incomplete_projected_depth",
            "candidate_semantics": "clean_voxel_source_pixels_missing_after_corruption",
        },
    )


def _sample_voxel_points(points: np.ndarray, voxel_size: float, limit: int,
                         rng: np.random.Generator) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3), dtype=np.float32)
    cells = np.floor(points/voxel_size).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    points = points[np.sort(first)]
    if len(points) > limit:
        points = points[rng.choice(len(points), limit, replace=False)]
    return np.ascontiguousarray(points, dtype=np.float32)


def interaction_reconstruction_supervision(
    input_cloud: np.ndarray,
    dense_world_points: np.ndarray,
    depth_m: np.ndarray,
    geom_image: np.ndarray,
    geom_names: dict[int, str],
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    seed: int,
    max_points: int = 512,
    voxel_size: float = 0.005,
    manipulated_tokens: Iterable[str] = ("akita_black_bowl_1",),
    related_tokens: Iterable[str] = ("plate_1",),
    workspace_low: np.ndarray | None = None,
    workspace_high: np.ndarray | None = None,
) -> InteractionSupervision:
    """Build visible interaction-surface GT and a label per input point."""
    if max_points < 1 or voxel_size <= 0:
        raise ValueError("max_points and voxel_size must be positive")
    names = {key: value.lower() for key, value in geom_names.items()}
    manipulated_ids = [key for key, name in names.items()
                       if any(token.lower() in name for token in manipulated_tokens)]
    related_ids = [key for key, name in names.items()
                   if any(token.lower() in name for token in related_tokens)]
    if not manipulated_ids or not related_ids:
        raise ValueError("Interaction geom mapping did not find manipulated/related objects")
    world = np.asarray(dense_world_points, dtype=np.float32)
    depth = np.asarray(depth_m, dtype=np.float32)
    ids = np.asarray(geom_image, dtype=np.int32)
    valid = np.isfinite(world).all(-1) & np.isfinite(depth) & (depth > 0.01)
    if workspace_low is not None:
        valid &= np.all(world >= np.asarray(workspace_low), axis=-1)
    if workspace_high is not None:
        valid &= np.all(world <= np.asarray(workspace_high), axis=-1)
    rng = np.random.default_rng(seed)
    manipulated = _sample_voxel_points(
        world[valid & np.isin(ids, manipulated_ids)], voxel_size, max_points, rng)
    related = _sample_voxel_points(
        world[valid & np.isin(ids, related_ids)], voxel_size, max_points, rng)
    manipulated_quota = min(len(manipulated), max_points//2)
    related_quota = min(len(related), max_points-manipulated_quota)
    manipulated_quota += min(len(manipulated)-manipulated_quota,
                             max_points-manipulated_quota-related_quota)
    if len(manipulated) > manipulated_quota:
        manipulated = manipulated[rng.choice(len(manipulated), manipulated_quota, replace=False)]
    if len(related) > related_quota:
        related = related[rng.choice(len(related), related_quota, replace=False)]
    target = (np.concatenate((manipulated, related)).astype(np.float32)
              if len(manipulated)+len(related) else np.empty((0, 3), np.float32))

    cloud = np.asarray(input_cloud, dtype=np.float32)
    pose = np.asarray(camera_to_world, dtype=np.float32)
    k = np.asarray(intrinsics, dtype=np.float32)
    camera = (cloud[:, :3]-pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    u = np.floor(camera[:, 0]*k[0, 0]/np.maximum(z, 1e-8)+k[0, 2]).astype(np.int64)
    v = np.floor(camera[:, 1]*k[1, 1]/np.maximum(z, 1e-8)+k[1, 2]).astype(np.int64)
    height, width = depth.shape
    inside = np.isfinite(camera).all(1) & (z > 0.01)
    inside &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    input_mask = np.zeros(len(cloud), dtype=bool)
    rows = np.flatnonzero(inside)
    union_ids = manipulated_ids + related_ids
    input_mask[rows] = np.isin(ids[v[rows], u[rows]], union_ids)
    input_mask[rows] &= np.abs(z[rows]-depth[v[rows], u[rows]]) < 0.05
    return InteractionSupervision(
        target_points=np.ascontiguousarray(target),
        input_mask=input_mask,
        stats={
            "target_points": int(len(target)),
            "manipulated_target_points": int(len(manipulated)),
            "related_target_points": int(len(related)),
            "positive_input_points": int(input_mask.sum()),
            "input_points": int(len(input_mask)),
            "target_voxel_size_m": float(voxel_size),
            "target_max_points": int(max_points),
            "depth_consistency_tolerance_m": 0.05,
            "training_only": True,
        },
    )


def realign_corrupted_features(
    result: CorruptionResult,
    rgb: np.ndarray,
    polar: dict,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> CorruptionResult:
    """Reproject moved points and sample RGB/polar at their current pixels.

    ``source_pixels`` retain pre-corruption provenance, while
    ``current_pixels`` describe the pixel where the changed XYZ now projects.
    This is the same alignment rule used by the RLBench polar9-v2 exporter.
    """
    cloud = np.asarray(result.cloud, dtype=np.float32).copy()
    k = np.asarray(intrinsics, dtype=np.float32)
    pose = np.asarray(camera_to_world, dtype=np.float32)
    image = np.asarray(rgb)
    height, width = image.shape[:2]
    camera_xyz = (cloud[:, :3] - pose[:3, 3]) @ pose[:3, :3]
    z = camera_xyz[:, 2]
    u_center = camera_xyz[:, 0] * k[0, 0] / np.maximum(z, 1e-8) + k[0, 2]
    v_center = camera_xyz[:, 1] * k[1, 1] / np.maximum(z, 1e-8) + k[1, 2]
    u, v = np.floor(u_center).astype(np.int64), np.floor(v_center).astype(np.int64)
    valid = np.isfinite(camera_xyz).all(1) & (z > 0.01)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_rows = np.flatnonzero(valid)
    current = (v[valid] * width + u[valid]).astype(np.int32)
    flat_rgb = image.reshape(-1, 3).astype(np.float32) / 255.0
    cloud = cloud[valid_rows]
    cloud[:, 3:6] = flat_rgb[current]
    polar_valid = np.asarray(polar["valid_mask"], dtype=bool).reshape(-1)[current]
    cloud[:, 6] = np.where(
        polar_valid, np.asarray(polar["DoLP"], dtype=np.float32).reshape(-1)[current], 0)
    cloud[:, 7] = np.where(
        polar_valid, np.asarray(polar["cos2AoLP"], dtype=np.float32).reshape(-1)[current], 0)
    cloud[:, 8] = np.where(
        polar_valid, np.asarray(polar["sin2AoLP"], dtype=np.float32).reshape(-1)[current], 0)
    # Geometry is governed only by MuJoCo/reprojection. Missing polar values
    # become zero features and never remove an otherwise valid simulated point.
    finite = np.isfinite(cloud).all(1)
    invalid_count = int(len(result.cloud) - np.count_nonzero(finite))
    stats = dict(result.stats)
    stats["reprojection_invalid_removed"] = invalid_count
    stats["changed_projected_pixel"] = int(np.count_nonzero(
        current[finite] != result.source_pixels[valid_rows][finite]))
    stats["output_points"] = int(np.count_nonzero(finite))
    return CorruptionResult(
        cloud=np.ascontiguousarray(cloud[finite]),
        source_pixels=np.ascontiguousarray(result.source_pixels[valid_rows][finite]),
        current_pixels=np.ascontiguousarray(current[finite]),
        codes=np.ascontiguousarray(result.codes[valid_rows][finite]),
        stats=stats,
    )


def _episode_rng(seed: int, episode_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, episode_index, 0x4C494245]))


def corrupt_libero_cloud(
    cloud: np.ndarray,
    pixel_indices: np.ndarray,
    geom_ids: np.ndarray,
    geom_names: dict[int, str],
    camera_origin: np.ndarray,
    *,
    episode_index: int,
    seed: int,
    target_tokens: Iterable[str] = ("akita_black_bowl_1",),
    support_tokens: Iterable[str] = ("plate_1",),
    robot_drop_fraction: float = 0.13,
    target_affected_fraction: float = 0.55,
    target_drop_fraction: float = 0.65,
) -> CorruptionResult:
    """Apply deterministic, episode-consistent missing-depth failure modes.

    Codes are: 0 unchanged, 1 support distortion, 2 target wrong depth,
    4 floating point, 5 robot hole. Removed target/robot rows are counted in
    metadata and intentionally have no output code.
    """
    for name, value in (
        ("robot_drop_fraction", robot_drop_fraction),
        ("target_affected_fraction", target_affected_fraction),
        ("target_drop_fraction", target_drop_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    points = np.asarray(cloud, dtype=np.float32)
    pixels = np.asarray(pixel_indices, dtype=np.int32)
    ids = np.asarray(geom_ids, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 9:
        raise ValueError(f"Expected Nx9 cloud, got {points.shape}")
    if pixels.shape != (len(points),) or ids.shape != pixels.shape:
        raise ValueError("pixel_indices and geom_ids must align with cloud rows")
    names = np.asarray([geom_names.get(int(item), "").lower() for item in ids], dtype=object)
    robot = np.asarray(["robot" in name or "gripper" in name for name in names])
    target_words = tuple(token.lower() for token in target_tokens)
    support_words = tuple(token.lower() for token in support_tokens)
    target = np.asarray([any(token in name for token in target_words) for name in names])
    support = np.asarray([any(token in name for token in support_words) for name in names])
    table = np.asarray(["table" in name for name in names])
    rng = _episode_rng(seed, episode_index)
    output = points.copy()
    keep = np.ones(len(points), dtype=bool)
    codes = np.zeros(len(points), dtype=np.uint8)

    # Ellipsoidal robot holes are fixed in normalized robot coordinates for the
    # whole episode, so the artifact moves consistently with the arm.
    robot_idx = np.flatnonzero(robot)
    robot_removed = np.empty(0, dtype=np.int64)
    if len(robot_idx) >= 20:
        xyz = points[robot_idx, :3]
        lo, span = xyz.min(0), np.maximum(np.ptp(xyz, axis=0), 1e-4)
        local = (xyz - lo) / span
        centers = rng.uniform(0.2, 0.8, size=(2, 3))
        radii = rng.uniform(0.12, 0.25, size=(2, 3))
        score = np.square((local[:, None] - centers) / radii).sum(-1).min(-1)
        count = min(round(robot_drop_fraction * len(robot_idx)), len(robot_idx))
        robot_removed = robot_idx[np.argsort(score)[:count]]
        keep[robot_removed] = False

    # Simulate a transparent/reflective target: most affected returns vanish;
    # the rest move along their camera rays by a smooth refraction-like offset.
    target_idx = np.flatnonzero(target)
    target_removed = np.empty(0, dtype=np.int64)
    target_shifted = np.empty(0, dtype=np.int64)
    if len(target_idx):
        center = points[target_idx, :3].mean(0)
        distance = np.linalg.norm(points[target_idx, :3] - center, axis=1)
        affected_count = min(round(target_affected_fraction * len(target_idx)), 320)
        affected = target_idx[np.argsort(distance)[:affected_count]]
        split = round(target_drop_fraction * len(affected))
        target_removed, target_shifted = affected[:split], affected[split:]
        keep[target_removed] = False
        vectors = output[target_shifted, :3] - np.asarray(camera_origin, dtype=np.float32)
        ranges = np.linalg.norm(vectors, axis=1)
        sign = rng.choice((-1.0, 1.0))
        error = sign * (0.012 + 0.018 * np.linspace(0, 1, len(target_shifted), dtype=np.float32))
        output[target_shifted, :3] += vectors / np.maximum(ranges[:, None], 1e-6) * error[:, None]
        codes[target_shifted] = 2

    support_idx = np.flatnonzero(support)
    if len(support_idx):
        phase = rng.uniform(0, 2 * np.pi)
        z = output[support_idx, 2]
        local_z = (z - z.min()) / max(float(np.ptp(z)), 1e-5)
        output[support_idx, 0] += 0.006 * np.sin(np.pi * local_z + phase)
        codes[support_idx] = 1

    table_idx = np.flatnonzero(table & keep)
    floating = np.empty(0, dtype=np.int64)
    if len(table_idx):
        count = min(24, len(table_idx))
        floating = rng.choice(table_idx, size=count, replace=False)
        center = points[target_idx, :3].mean(0) if len(target_idx) else points[table_idx, :3].mean(0)
        output[floating, :3] = center + rng.normal(0, (0.035, 0.035, 0.05), size=(count, 3))
        codes[floating] = 4

    retained = np.flatnonzero(keep)
    return CorruptionResult(
        cloud=np.ascontiguousarray(output[retained]),
        source_pixels=np.ascontiguousarray(pixels[retained]),
        current_pixels=np.ascontiguousarray(pixels[retained]),
        codes=np.ascontiguousarray(codes[retained]),
        stats={
            "input_points": int(len(points)), "output_points": int(len(retained)),
            "robot_hole_removed": int(len(robot_removed)),
            "target_transparent_removed": int(len(target_removed)),
            "target_wrong_depth": int(len(target_shifted)),
            "support_shape_distorted": int(len(support_idx)),
            "floating_points": int(len(floating)),
            "target_points": int(target.sum()), "support_points": int(support.sum()),
            "robot_points": int(robot.sum()),
            "robot_drop_fraction": float(robot_drop_fraction),
            "target_affected_fraction": float(target_affected_fraction),
            "target_drop_fraction": float(target_drop_fraction),
        },
    )


def make_aligned_cloud(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    world_points: np.ndarray,
    polar: dict,
    geom_image: np.ndarray,
    *,
    workspace_low: np.ndarray,
    workspace_high: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create voxelized [XYZ RGB DoLP cos2AoLP sin2AoLP] rows."""
    xyz = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    color = np.asarray(rgb, dtype=np.float32).reshape(-1, 3) / 255.0
    depth = np.asarray(depth_m, dtype=np.float32).reshape(-1)
    ids = np.asarray(geom_image, dtype=np.int32).reshape(-1)
    valid = np.isfinite(xyz).all(1) & np.isfinite(depth) & (depth > 0.01)
    valid &= np.all(xyz >= np.asarray(workspace_low), axis=1)
    valid &= np.all(xyz <= np.asarray(workspace_high), axis=1)
    rows = np.flatnonzero(valid)
    if not len(rows):
        raise ValueError("No points remain after validity and workspace filtering")
    selected = voxel_indices(xyz[rows], voxel_size)
    rows = rows[selected]
    polar_valid = np.asarray(polar["valid_mask"], dtype=bool).reshape(-1)[rows]
    dolp = np.asarray(polar["DoLP"]).reshape(-1)[rows]
    cos2 = np.asarray(polar["cos2AoLP"]).reshape(-1)[rows]
    sin2 = np.asarray(polar["sin2AoLP"]).reshape(-1)[rows]
    cloud = np.column_stack((
        xyz[rows], color[rows],
        np.where(polar_valid, dolp, 0),
        np.where(polar_valid, cos2, 0),
        np.where(polar_valid, sin2, 0),
    )).astype(np.float32)
    return np.ascontiguousarray(cloud), rows.astype(np.int32), ids[rows].astype(np.int32)
