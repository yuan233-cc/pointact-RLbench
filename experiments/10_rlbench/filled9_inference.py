"""Build training-matched filled9 inputs from a live RLBench polar frame.

The live evaluation intentionally recreates the *controlled simulated* input
pipeline used to build the training archive: clean voxel representatives are
corrupted, RGB/polar features are sampled at the corrupted projection, and
only clean source pixels removed by corruption are eligible for depth fill.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from create_rlbench_10task_realistic_failure_dataset import apply_corruption
from polar_depth_fill import corruption_hole_pixels, fill_depth_multiscale


WORKSPACE_LOW = np.array([-0.5, -1.0, 0.7505], dtype=np.float32)
WORKSPACE_HIGH = np.array([1.5, 1.0, 2.0], dtype=np.float32)
def polar_frame(observation) -> dict[str, np.ndarray]:
    """Extract the dense arrays needed by the training preprocessor."""
    polar = observation.front_polarization
    if polar is None:
        raise ValueError("RLBench observation has no front_polarization")
    required = ("DoLP", "AoLP", "valid_mask", "AoLP_valid_mask")
    missing = [key for key in required if key not in polar]
    if missing:
        raise ValueError(f"front_polarization is missing: {missing}")
    camera_keys = ("front_camera_extrinsics", "front_camera_intrinsics")
    missing_camera = [key for key in camera_keys if key not in observation.misc]
    if missing_camera:
        raise ValueError(f"RLBench observation misc is missing: {missing_camera}")
    return {
        "rgb": np.asarray(observation.front_rgb, dtype=np.uint8),
        "depth": np.asarray(observation.front_depth, dtype=np.float32),
        "point_cloud": np.asarray(observation.front_point_cloud, dtype=np.float32),
        "camera": {
            "to_world": np.asarray(
                observation.misc["front_camera_extrinsics"], dtype=np.float64
            ),
            "intrinsics": np.asarray(
                observation.misc["front_camera_intrinsics"], dtype=np.float64
            ),
        },
        **{key: np.asarray(polar[key]) for key in required},
    }


def _polar_maps(frame: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply the same validity rules as the repaired dataset builder."""
    dolp = np.asarray(frame["DoLP"], dtype=np.float32)
    angle = np.asarray(frame["AoLP"], dtype=np.float32)
    valid = (
        np.asarray(frame["valid_mask"], dtype=bool)
        & np.isfinite(dolp)
        & (dolp >= 0.0)
        & (dolp <= 1.0)
    )
    angle_valid = (
        np.asarray(frame["AoLP_valid_mask"], dtype=bool)
        & valid
        & np.isfinite(angle)
    )
    return {
        "DoLP": np.where(valid, dolp, 0.0).astype(np.float32),
        "AoLP": np.where(angle_valid, angle, 0.0).astype(np.float32),
        "valid_mask": valid,
        "AoLP_valid_mask": angle_valid,
    }


def _project_world(xyz: np.ndarray, camera: Mapping[str, np.ndarray]):
    """Project world XYZ using the live per-frame RLBench calibration."""
    to_world = np.asarray(camera["to_world"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    camera_xyz = (np.asarray(xyz, dtype=np.float64) - to_world[:3, 3]) @ to_world[:3, :3]
    depth = camera_xyz[:, 2]
    image = camera_xyz @ intrinsics.T
    uv = image[:, :2] / np.where(
        np.abs(image[:, 2:3]) > 1e-12, image[:, 2:3], np.nan
    )
    return uv, depth


def clean_voxel_points(frame: Mapping[str, np.ndarray], voxel_size: float = 0.012):
    """Return the same clean Nx9 voxel representatives and source pixels as export."""
    maps = _polar_maps(frame)
    xyz = np.asarray(frame["point_cloud"], dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(frame["rgb"], dtype=np.float32).reshape(-1, 3) / 255.0
    depth = np.asarray(frame["depth"], dtype=np.float32).reshape(-1)
    angle_valid = maps["AoLP_valid_mask"].reshape(-1)
    valid = maps["valid_mask"].reshape(-1).copy()
    valid &= np.isfinite(depth) & (depth > 0.01)
    valid &= np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1)
    valid &= np.all((xyz >= WORKSPACE_LOW) & (xyz <= WORKSPACE_HIGH), axis=1)

    dolp = maps["DoLP"].reshape(-1)
    angle = maps["AoLP"].reshape(-1)
    cos2 = np.where(angle_valid, np.cos(2 * angle), 0.0)
    sin2 = np.where(angle_valid, np.sin(2 * angle), 0.0)
    pixels = np.flatnonzero(valid).astype(np.int32)
    points = np.column_stack((xyz[valid], rgb[valid], dolp[valid], cos2[valid], sin2[valid])).astype(np.float32)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("No finite workspace points in live polar frame")

    voxel = np.floor(points[:, :3] / voxel_size).astype(np.int32)
    _, representative = np.unique(voxel, axis=0, return_index=True)
    representative.sort()
    return np.ascontiguousarray(points[representative]), pixels[representative]


def sample_projected_modalities(frame: Mapping[str, np.ndarray], xyz: np.ndarray):
    """Equivalent in-memory form of the training export sampler."""
    maps = _polar_maps(frame)
    height, width = maps["DoLP"].shape
    uv, depth = _project_world(xyz, frame["camera"])
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth)
    pixel_xy = np.floor(
        np.where(finite[:, None], uv, -1.0) + 1e-4
    ).astype(np.int64)
    in_frame = (
        finite & (depth > 0.01)
        & (pixel_xy[:, 0] >= 0) & (pixel_xy[:, 0] < width)
        & (pixel_xy[:, 1] >= 0) & (pixel_xy[:, 1] < height)
    )
    pixel_index = np.full(len(xyz), -1, dtype=np.int32)
    pixel_index[in_frame] = (pixel_xy[in_frame, 1] * width + pixel_xy[in_frame, 0]).astype(np.int32)
    valid = in_frame.copy()
    selected = pixel_index[in_frame]
    valid[in_frame] &= maps["valid_mask"].reshape(-1)[selected]
    rgb = np.zeros((len(xyz), 3), dtype=np.float32)
    polar = np.zeros((len(xyz), 3), dtype=np.float32)
    selected = pixel_index[valid]
    if len(selected):
        rgb[valid] = np.asarray(frame["rgb"]).reshape(-1, 3)[selected] / 255.0
        angle = maps["AoLP"].reshape(-1)[selected]
        angle_valid = maps["AoLP_valid_mask"].reshape(-1)[selected]
        polar[valid, 0] = maps["DoLP"].reshape(-1)[selected]
        polar[valid, 1] = np.where(angle_valid, np.cos(2 * angle), 0.0)
        polar[valid, 2] = np.where(angle_valid, np.sin(2 * angle), 0.0)
    valid &= np.isfinite(rgb).all(axis=1) & np.isfinite(polar).all(axis=1)
    return rgb, polar, pixel_index, valid


def fill_cloud(
    cloud: np.ndarray,
    pixels: np.ndarray,
    holes: np.ndarray,
    frame: Mapping[str, np.ndarray],
):
    """Match ``repair_10task_polar_rlbench9.filled_cloud`` exactly."""
    maps = _polar_maps(frame)
    camera = frame["camera"]
    height, width = maps["DoLP"].shape
    _, camera_depth = _project_world(cloud[:, :3], camera)
    sparse = np.full(height * width, np.inf, dtype=np.float32)
    good = np.isfinite(camera_depth) & (camera_depth > 0.01)
    np.minimum.at(sparse, np.asarray(pixels, dtype=np.int64)[good], camera_depth[good])
    sparse[~np.isfinite(sparse)] = 0.0
    estimated = fill_depth_multiscale(sparse.reshape(height, width)).reshape(-1)
    holes = np.unique(np.asarray(holes, dtype=np.int32))
    new_pixels = holes[(sparse[holes] <= 0.01) & (estimated[holes] > 0.01)]
    if not len(new_pixels):
        return (
            np.ascontiguousarray(cloud.copy()),
            np.asarray(pixels, dtype=np.int32).copy(),
            np.zeros(len(cloud), dtype=bool),
        )

    z = estimated[new_pixels]
    u, v = new_pixels % width, new_pixels // width
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    camera_xyz = np.column_stack((
        (u - intrinsics[0, 2]) * z / intrinsics[0, 0],
        (v - intrinsics[1, 2]) * z / intrinsics[1, 1],
        z,
    ))
    to_world = np.asarray(camera["to_world"], dtype=np.float64)
    xyz = camera_xyz @ to_world[:3, :3].T + to_world[:3, 3]
    colors = np.asarray(frame["rgb"]).reshape(-1, 3)[new_pixels].astype(np.float32) / 255.0
    dolp = maps["DoLP"].reshape(-1)[new_pixels]
    angle = maps["AoLP"].reshape(-1)[new_pixels]
    angle_valid = maps["AoLP_valid_mask"].reshape(-1)[new_pixels]
    rows = np.column_stack((
        xyz,
        colors,
        dolp,
        np.where(angle_valid, np.cos(2 * angle), 0.0),
        np.where(angle_valid, np.sin(2 * angle), 0.0),
    )).astype(np.float32)
    filled = np.ascontiguousarray(np.concatenate((cloud, rows)))
    filled_pixels = np.concatenate((np.asarray(pixels, dtype=np.int32), new_pixels))
    filled_mask = np.concatenate((
        np.zeros(len(cloud), dtype=bool), np.ones(len(rows), dtype=bool)
    ))
    return filled, filled_pixels, filled_mask


def build_incomplete9(
    frame: Mapping[str, np.ndarray],
    *,
    task_index: int,
    source_episode: int,
    corruption_seed: int,
    voxel_size: float = 0.012,
):
    """Produce the training-matched incomplete Nx9 policy input."""
    clean, clean_pixels = clean_voxel_points(frame, voxel_size)
    indexed = np.column_stack((clean[:, :6], np.arange(len(clean), dtype=np.float32)))
    corruption = apply_corruption(indexed, task_index, source_episode, corruption_seed)
    source_rows = corruption.cloud[:, 6].astype(np.int64)
    rgb, polar, projected_pixels, valid = sample_projected_modalities(frame, corruption.cloud[:, :3])
    incomplete = np.ascontiguousarray(
        np.column_stack((corruption.cloud[valid, :3], rgb[valid], polar[valid])),
        dtype=np.float32,
    )
    retained_source_pixels = clean_pixels[source_rows[valid]].astype(np.int32)
    holes = corruption_hole_pixels(clean_pixels, retained_source_pixels)
    if incomplete.ndim != 2 or incomplete.shape[1] != 9 or not np.isfinite(incomplete).all():
        raise ValueError(f"Invalid incomplete9 cloud: {incomplete.shape}")
    stats = {
        "clean_points": int(len(clean)),
        "corrupted_points": int(len(corruption.cloud)),
        "valid_incomplete_points": int(len(incomplete)),
        "hole_pixels": int(len(holes)),
        "corruption": corruption.stats,
    }
    return incomplete, projected_pixels[valid].astype(np.int32), holes, stats


def build_filled9(
    frame: Mapping[str, np.ndarray],
    *,
    task_index: int,
    source_episode: int,
    corruption_seed: int,
    voxel_size: float = 0.012,
):
    """Produce one filled Nx9 policy input and reproducibility statistics."""
    incomplete, incomplete_pixels, holes, stats = build_incomplete9(
        frame,
        task_index=task_index,
        source_episode=source_episode,
        corruption_seed=corruption_seed,
        voxel_size=voxel_size,
    )
    filled, filled_pixels, filled_mask = fill_cloud(
        incomplete, incomplete_pixels, holes, frame
    )
    if filled.ndim != 2 or filled.shape[1] != 9 or not np.isfinite(filled).all():
        raise ValueError(f"Invalid filled9 cloud: {filled.shape}")
    stats = {
        **stats,
        "depth_filled_points": int(filled_mask.sum()),
        "filled_points": int(len(filled)),
    }
    return filled, filled_pixels, filled_mask, stats
