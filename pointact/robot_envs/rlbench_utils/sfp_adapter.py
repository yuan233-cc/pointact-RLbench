"""Adapt a live RLBench polar observation to SfP-Wild model inputs.

The adapter mirrors the offline ``build_sfp_wild_inputs.py`` exporter.  In
particular, it uses same-frame RGB luminance as the current ``I_un`` proxy and
converts RLBench's negative-focal camera convention to the positive-focal
camera frame used by PointACT's polar router.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


LUMINANCE_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
CAMERA_AXIS_CONVERSION = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def _normalized_luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB must have shape HxWx3, got {rgb.shape}")
    values = rgb.astype(np.float32)
    byte_scaled = np.issubdtype(rgb.dtype, np.integer) or (
        values.size and values.max() > 1.0
    )
    upper_bound = 255.0 if byte_scaled else 1.0
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > upper_bound):
        raise ValueError("RGB must contain finite values in [0,1] or [0,255]")
    luminance = np.tensordot(values, LUMINANCE_WEIGHTS, axes=([-1], [0]))
    if byte_scaled:
        # Match build_sfp_wild_inputs.py byte for byte. The offline exporter
        # stores rounded uint8 luminance before the dataset normalizes it.
        luminance = np.clip(np.rint(luminance), 0.0, 255.0) / 255.0
    return luminance.astype(np.float32)


def _positive_focal_calibration(camera: Mapping[str, np.ndarray]):
    source_k = np.asarray(camera["intrinsics"], dtype=np.float64)
    to_world = np.asarray(camera["to_world"], dtype=np.float64)
    if source_k.shape != (3, 3) or not np.isfinite(source_k).all():
        raise ValueError("RLBench front-camera intrinsics must be a finite 3x3 matrix")
    if to_world.shape != (4, 4) or not np.isfinite(to_world).all():
        raise ValueError("RLBench front-camera to_world must be a finite 4x4 matrix")
    if source_k[0, 0] >= 0 or source_k[1, 1] >= 0:
        raise ValueError(
            "SfP adapter expects RLBench's negative-focal camera convention; "
            "do not enable a second OpenCV conversion"
        )
    k = source_k.copy()
    k[0, 0] *= -1
    k[1, 1] *= -1
    camera_from_world = CAMERA_AXIS_CONVERSION @ np.linalg.inv(to_world)
    return k.astype(np.float32), camera_from_world.astype(np.float32)


def _viewing_directions(k: np.ndarray, height: int, width: int) -> np.ndarray:
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    rays = np.stack(
        (
            (k[0, 2] - cols) / k[0, 0],
            (rows - k[1, 2]) / k[1, 1],
            np.ones((height, width), dtype=np.float32),
        )
    )
    rays /= np.linalg.norm(rays, axis=0, keepdims=True).clip(1e-8)
    return rays.astype(np.float32)


def sfp_inputs_from_polar_frame(frame: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Return one sample's view-batched SfP-Wild inputs.

    ``frame`` follows ``filled9_inference.polar_frame`` and must contain live
    RGB, DoLP/AoLP arrays, their validity masks, and front-camera calibration.
    Returned arrays retain a leading view dimension so a policy client can
    wrap each value in its outer sample-batch list.
    """
    i_un = _normalized_luminance(np.asarray(frame["rgb"]))
    height, width = i_un.shape
    expected = (height, width)
    dolp = np.asarray(frame["DoLP"], dtype=np.float32)
    aolp = np.asarray(frame["AoLP"], dtype=np.float32)
    valid = np.array(frame["valid_mask"], dtype=bool, copy=True)
    angle_valid = np.array(frame["AoLP_valid_mask"], dtype=bool, copy=True)
    for name, array in (
        ("DoLP", dolp), ("AoLP", aolp), ("valid_mask", valid),
        ("AoLP_valid_mask", angle_valid),
    ):
        if array.shape != expected:
            raise ValueError(f"{name} must match RGB image shape {expected}, got {array.shape}")

    valid &= np.isfinite(dolp) & (dolp >= 0.0) & (dolp <= 1.0)
    angle_valid &= valid & np.isfinite(aolp)
    observed_dolp = np.where(valid, dolp, 0.0).astype(np.float32)
    cos2 = np.where(angle_valid, np.cos(2.0 * aolp), 0.0).astype(np.float32)
    sin2 = np.where(angle_valid, np.sin(2.0 * aolp), 0.0).astype(np.float32)

    k, camera_from_world = _positive_focal_calibration(frame["camera"])
    rays = _viewing_directions(k, height, width)
    polar_images = np.concatenate(
        (i_un[None], observed_dolp[None], cos2[None], sin2[None], rays), axis=0
    )
    if polar_images.shape != (7, height, width) or not np.isfinite(polar_images).all():
        raise ValueError("Failed to assemble finite seven-channel SfP input")
    return {
        "polar_images": polar_images[None].astype(np.float32),
        "polar_K": k[None],
        "T_camera_from_world": camera_from_world[None],
        "view_valid": np.ones(1, dtype=bool),
        "pixel_valid": valid[None],
    }
