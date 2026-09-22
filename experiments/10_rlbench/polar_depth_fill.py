"""Add camera-visible polar samples at holes in an incomplete point cloud.

This is an offline depth preprocessor, not a learned geometry-completion model.
The input cloud is rasterized with a nearest-depth z-buffer. Only pixels whose
voxel representatives were removed by the simulated corruption are candidates
for filling; other empty pixels are normal voxel downsampling, not holes.
Missing depths are estimated from nearby surviving depths, then unprojected.
RGB and polarization always come from the *same* image pixel as the new point.
"""

from __future__ import annotations

import cv2
import numpy as np


def corruption_hole_pixels(
    clean_pixel_indices: np.ndarray,
    retained_source_pixel_indices: np.ndarray,
) -> np.ndarray:
    """Return source pixels lost after corruption, before point reprojection.

    A surviving point may move to another current pixel. Its original pixel is
    still considered retained because a geometry error is not a missing point.
    """
    return np.setdiff1d(
        np.asarray(clean_pixel_indices, dtype=np.int32),
        np.asarray(retained_source_pixel_indices, dtype=np.int32),
        assume_unique=False,
    )


def _cross_kernel(size: int) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.uint8)
    kernel[size // 2, :] = 1
    kernel[:, size // 2] = 1
    return kernel


def fill_depth_multiscale(sparse_depth: np.ndarray) -> np.ndarray:
    """Fill nearby zero-depth pixels with HouseCat6D-style morphology.

    Unlike HouseCat6D's fixed 3 m inversion range, the range is chosen from
    the input because RLBench front-camera depths can exceed 3 m. Surviving
    depths are restored exactly after filtering.
    """
    depth = np.asarray(sparse_depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth image, got {depth.shape}")
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

    # Keep HouseCat6D's no-extrapolation behavior above the first supported
    # pixel in each column. An empty column may still be filled from its
    # neighbors; treating it as permanently invalid would preserve wide holes.
    top_rows = np.argmax(closed > 0.01, axis=0)
    rows = np.arange(depth.shape[0])[:, None]
    fill_region = rows >= top_rows[None, :]

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
    result[~np.isfinite(result) | (result <= 0.01)] = 0.0
    return result


def add_polar_depth_filled_points(
    cloud: np.ndarray,
    pixel_indices: np.ndarray,
    frame,
    camera_to_world: np.ndarray,
    focal: float,
    center: np.ndarray,
    *,
    hole_pixel_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Nx9 points, their pixels, and a bool mask for new filled points.

    Add points only at explicitly identified corruption holes where morphology
    estimates a depth. Existing rows stay unchanged; unavailable polar values
    use zero channels. The hole pixels come from pre-corruption voxel samples,
    never from all empty pixels of the sparse projected cloud.
    """
    cloud = np.asarray(cloud, dtype=np.float32)
    pixels = np.asarray(pixel_indices, dtype=np.int64)
    if cloud.ndim != 2 or cloud.shape[1] != 9 or pixels.shape != (len(cloud),):
        raise ValueError("Expected Nx9 cloud and one pixel index per point")
    if focal <= 0:
        raise ValueError("focal must be positive")
    rgb = np.asarray(frame["rgb"])
    dolp = np.asarray(frame["DoLP"], dtype=np.float32)
    angle = np.asarray(frame["AoLP"], dtype=np.float32)
    height, width = dolp.shape
    if rgb.shape != (height, width, 3) or angle.shape != (height, width):
        raise ValueError("RGB and polar maps must have the same pixel grid")
    if np.any((pixels < 0) | (pixels >= height * width)):
        raise ValueError("Point pixel index is outside the polar image")
    holes = np.asarray(hole_pixel_indices, dtype=np.int64)
    if holes.ndim != 1 or np.any((holes < 0) | (holes >= height * width)):
        raise ValueError("Hole pixel indices must be a 1D array inside the polar image")
    if len(cloud) == 0:
        return cloud.copy(), pixels.astype(np.int32), np.zeros(len(cloud), dtype=bool)
    if not len(holes):
        return cloud.copy(), pixels.astype(np.int32), np.zeros(len(cloud), dtype=bool)

    rotation = np.asarray(camera_to_world, dtype=np.float32)[:3, :3]
    translation = np.asarray(camera_to_world, dtype=np.float32)[:3, 3]
    camera_xyz = (cloud[:, :3] - translation) @ rotation
    z = camera_xyz[:, 2]
    observed = np.full(height * width, np.inf, dtype=np.float32)
    good = np.isfinite(z) & (z > 0.01)
    np.minimum.at(observed, pixels[good], z[good])
    observed[~np.isfinite(observed)] = 0.0
    observed = observed.reshape(height, width)
    estimated = fill_depth_multiscale(observed)

    holes = np.unique(holes).astype(np.int32)
    observed_flat = observed.reshape(-1)
    estimated_flat = estimated.reshape(-1)
    new_pixels = holes[(observed_flat[holes] <= 0.01) &
                       (estimated_flat[holes] > 0.01)]
    if not len(new_pixels):
        return cloud.copy(), pixels.astype(np.int32), np.zeros(len(cloud), dtype=bool)

    z_new = estimated_flat[new_pixels]
    u = new_pixels % width + 0.5
    v = new_pixels // width + 0.5
    center = np.asarray(center, dtype=np.float32)
    xyz_camera = np.column_stack((
        (u - center[0]) * z_new / focal,
        (v - center[1]) * z_new / focal,
        z_new,
    ))
    xyz_world = xyz_camera @ rotation.T + translation
    flat_rgb = rgb.reshape(-1, 3)[new_pixels].astype(np.float32) / 255.0
    flat_dolp = dolp.reshape(-1)[new_pixels]
    flat_angle = angle.reshape(-1)[new_pixels]
    # Earlier renders narrowed valid_mask while selecting workspace points.
    # AoLP_valid_mask still describes the rendered image, including depth holes.
    dolp_valid = np.isfinite(flat_dolp) & (flat_dolp >= 0.0) & (flat_dolp <= 1.0)
    angle_valid = (
        np.asarray(frame["AoLP_valid_mask"], dtype=bool).reshape(-1)[new_pixels]
        & np.isfinite(flat_angle)
    )
    flat_dolp = np.where(dolp_valid, flat_dolp, 0.0)
    flat_angle = np.where(angle_valid, flat_angle, 0.0)
    new_rows = np.column_stack((
        xyz_world, flat_rgb, flat_dolp,
        np.where(angle_valid, np.cos(2 * flat_angle), 0.0),
        np.where(angle_valid, np.sin(2 * flat_angle), 0.0),
    )).astype(np.float32)
    result = np.ascontiguousarray(np.concatenate((cloud, new_rows)))
    all_pixels = np.concatenate((pixels.astype(np.int32), new_pixels))
    filled_mask = np.concatenate((
        np.zeros(len(cloud), dtype=bool), np.ones(len(new_rows), dtype=bool)))
    return result, all_pixels, filled_mask
