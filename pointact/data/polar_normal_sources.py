"""Dataset-independent polarization preprocessing for CGA normal training.

The physical prior follows Wang et al., Optics Express 2025: two specular
normal candidates, one diffuse normal candidate, unpolarized intensity, and a
3x3 local-minimum specular-confidence map.  No depth or normal ground truth is
used to build these input features.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.ndimage import minimum_filter


def _float_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float32) / float(np.iinfo(value.dtype).max)
    else:
        value = value.astype(np.float32)
    if value.ndim not in (2, 3):
        raise ValueError(f"Analyzer image must be HW or HWC, got {value.shape}")
    return value


def stack_analyzers(
    images: Iterable[np.ndarray],
    *,
    normalization: str = "dtype",
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return analyzer images as aligned ``[4,H,W,C]`` float arrays.

    Integer images are normalized by their dtype.  ``percentile`` additionally
    applies one shared robust scale, which is useful for linear HDR renders and
    preserves Stokes ratios before clipping.
    """
    values = [_float_image(image) for image in images]
    if len(values) != 4:
        raise ValueError(f"Expected four analyzer images, got {len(values)}")
    values = [value[..., None] if value.ndim == 2 else value for value in values]
    shapes = {value.shape for value in values}
    if len(shapes) != 1:
        raise ValueError(f"Analyzer images are not aligned: {sorted(shapes)}")
    analyzers = np.stack(values, axis=0)
    if normalization == "percentile":
        finite = np.isfinite(analyzers)
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape != analyzers.shape[1:3]:
                raise ValueError("valid_mask is not aligned with analyzer images")
            finite &= mask[None, ..., None]
        selected = analyzers[finite]
        if selected.size == 0:
            raise ValueError("No finite analyzer pixels available for normalization")
        scale = max(float(np.percentile(selected, 99.5)), 1e-6)
        analyzers = analyzers / scale
    elif normalization != "dtype":
        raise ValueError("normalization must be 'dtype' or 'percentile'")
    return np.clip(np.nan_to_num(analyzers), 0.0, 1.0).astype(np.float32)


def _grayscale(analyzers: np.ndarray) -> np.ndarray:
    if analyzers.shape[-1] == 1:
        return analyzers[..., 0]
    if analyzers.shape[-1] != 3:
        raise ValueError(f"Expected one or three analyzer channels, got {analyzers.shape[-1]}")
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.einsum("nhwc,c->nhw", analyzers, weights)


def viewing_directions(height: int, width: int, camera_k: np.ndarray | None = None) -> np.ndarray:
    """Build unit camera rays in ``+x right, +y down, +z forward`` coordinates."""
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    xx += 0.5
    yy += 0.5
    if camera_k is None:
        x = (xx - 0.5 * width) / max(0.5 * width, 1.0)
        y = (yy - 0.5 * height) / max(0.5 * height, 1.0)
    else:
        camera_k = np.asarray(camera_k, dtype=np.float32)
        if camera_k.shape != (3, 3):
            raise ValueError(f"camera_k must be 3x3, got {camera_k.shape}")
        x = (xx - camera_k[0, 2]) / camera_k[0, 0]
        y = (yy - camera_k[1, 2]) / camera_k[1, 1]
    rays = np.stack((x, y, np.ones_like(x)), axis=-1)
    return rays / np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-8)


def normals_from_depth(
    depth: np.ndarray,
    camera_k: np.ndarray,
    *,
    depth_scale: float = 1.0,
    relative_edge_threshold: float = 0.05,
    absolute_edge_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert clean pinhole z-depth to face-forward camera-space normals.

    A centered finite difference is used only where all four neighbors contain
    valid depth and do not cross a large depth discontinuity.  This is intended
    for independent rendered/laser GT depth, not noisy observed sensor depth.
    """
    depth = np.asarray(depth, dtype=np.float32) * float(depth_scale)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HW, got {depth.shape}")
    if depth_scale <= 0.0:
        raise ValueError("depth_scale must be positive")
    if relative_edge_threshold < 0.0 or absolute_edge_threshold < 0.0:
        raise ValueError("depth edge thresholds must be non-negative")
    rays = viewing_directions(*depth.shape, camera_k)
    points = rays * (depth / np.maximum(rays[..., 2], 1e-8))[..., None]
    normal = np.zeros((*depth.shape, 3), dtype=np.float32)
    tangent_x = points[1:-1, 2:] - points[1:-1, :-2]
    tangent_y = points[2:, 1:-1] - points[:-2, 1:-1]
    normal[1:-1, 1:-1] = np.cross(tangent_x, tangent_y)

    finite_positive = np.isfinite(depth) & (depth > 0.0)
    center = depth[1:-1, 1:-1]
    neighbors = np.stack(
        (depth[1:-1, :-2], depth[1:-1, 2:], depth[:-2, 1:-1], depth[2:, 1:-1]),
        axis=0,
    )
    threshold = np.maximum(absolute_edge_threshold, relative_edge_threshold * center)
    interior_valid = (
        finite_positive[1:-1, 1:-1]
        & finite_positive[1:-1, :-2]
        & finite_positive[1:-1, 2:]
        & finite_positive[:-2, 1:-1]
        & finite_positive[2:, 1:-1]
        & (np.max(np.abs(neighbors - center[None]), axis=0) <= threshold)
    )
    valid = np.zeros_like(finite_positive)
    valid[1:-1, 1:-1] = interior_valid
    return face_forward_normals(normal, rays, valid)


def polarization_observation(analyzers: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the 11-channel native-CGA observation from four analyzers."""
    analyzers = np.asarray(analyzers, dtype=np.float32)
    if analyzers.ndim != 4 or analyzers.shape[0] != 4:
        raise ValueError(f"analyzers must be [4,H,W,C], got {analyzers.shape}")
    gray = _grayscale(analyzers)
    i0, i45, i90, i135 = gray
    iun = 0.5 * (i0 + i45 + i90 + i135)
    q = i0 - i90
    u = i45 - i135
    amplitude = np.sqrt(np.square(q) + np.square(u))
    dolp = np.clip(amplitude / np.maximum(iun, 1e-8), 0.0, 1.0)
    angle_valid = amplitude > 1e-8
    cos2 = np.where(angle_valid, q / np.maximum(amplitude, 1e-8), 0.0)
    sin2 = np.where(angle_valid, u / np.maximum(amplitude, 1e-8), 0.0)
    aolp = np.mod(0.5 * np.arctan2(u, q), np.pi)
    contrast = gray.max(axis=0) - gray.min(axis=0)
    specular_confidence = minimum_filter(contrast, size=3, mode="nearest")
    return {
        "images": gray.astype(np.float32),
        "Iun": iun[None].astype(np.float32),
        "DoP": dolp[None].astype(np.float32),
        "cos1": cos2[None].astype(np.float32),
        "cos2": sin2[None].astype(np.float32),
        "AoLP": aolp.astype(np.float32),
        "spec": np.clip(specular_confidence, 0.0, 1.0)[None].astype(np.float32),
        "rgb": analyzers.mean(axis=0).astype(np.float32),
    }


def _rho_diffuse(theta: np.ndarray, eta: float) -> np.ndarray:
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin2 = np.square(sin_theta)
    root = np.sqrt(np.maximum(np.square(eta) - sin2, 0.0))
    numerator = np.square(eta - 1.0 / eta) * sin2
    denominator = (
        2.0
        + 2.0 * np.square(eta)
        - np.square(eta + 1.0 / eta) * sin2
        + 4.0 * cos_theta * root
    )
    return numerator / np.maximum(denominator, 1e-8)


def _rho_specular(theta: np.ndarray, eta: float) -> np.ndarray:
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin2 = np.square(sin_theta)
    root = np.sqrt(np.maximum(np.square(eta) - sin2, 0.0))
    numerator = 2.0 * sin2 * cos_theta * root
    denominator = np.square(eta) - sin2 - np.square(eta) * sin2 + 2.0 * np.square(sin2)
    return numerator / np.maximum(denominator, 1e-8)


@lru_cache(maxsize=8)
def _zenith_tables(eta: float, resolution: int) -> tuple[np.ndarray, ...]:
    theta = np.linspace(0.0, np.pi / 2.0 - 1e-5, resolution, dtype=np.float64)
    diffuse = _rho_diffuse(theta, eta)
    specular = _rho_specular(theta, eta)
    peak = int(np.argmax(specular))
    return theta, diffuse, specular[: peak + 1], theta[: peak + 1], specular[peak:][::-1], theta[peak:][::-1]


def ambiguous_normals(
    dolp: np.ndarray,
    aolp: np.ndarray,
    *,
    refractive_index: float = 1.5,
    face_camera: bool = True,
    lookup_resolution: int = 65536,
) -> np.ndarray:
    """Return ``[Ns1,Ns2,Nd]`` as a nine-channel CGA physical prior."""
    if refractive_index <= 1.0:
        raise ValueError("refractive_index must be greater than one")
    dolp = np.clip(np.asarray(dolp, dtype=np.float32), 0.0, 1.0)
    aolp = np.asarray(aolp, dtype=np.float32)
    if dolp.shape != aolp.shape:
        raise ValueError("dolp and aolp must have the same shape")
    theta, rho_d, rho_s1, theta_s1, rho_s2, theta_s2 = _zenith_tables(
        float(refractive_index), int(lookup_resolution)
    )
    theta_d = np.interp(dolp, rho_d, theta)
    theta_1 = np.interp(dolp, rho_s1, theta_s1)
    theta_2 = np.interp(dolp, rho_s2, theta_s2)

    def from_angles(azimuth: np.ndarray, zenith: np.ndarray) -> np.ndarray:
        sin_zenith = np.sin(zenith)
        normal = np.stack(
            (np.cos(azimuth) * sin_zenith, np.sin(azimuth) * sin_zenith, np.cos(zenith)),
            axis=0,
        )
        return -normal if face_camera else normal

    specular_azimuth = aolp + np.pi / 2.0
    candidates = np.concatenate(
        (
            from_angles(specular_azimuth, theta_1),
            from_angles(specular_azimuth, theta_2),
            from_angles(aolp, theta_d),
        ),
        axis=0,
    )
    return candidates.astype(np.float32)


def face_forward_normals(
    normals: np.ndarray, viewing_direction: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize normals and orient them against camera rays."""
    normals = np.asarray(normals, dtype=np.float32)
    viewing_direction = np.asarray(viewing_direction, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if normals.shape != viewing_direction.shape or normals.ndim != 3 or normals.shape[-1] != 3:
        raise ValueError("normals and viewing_direction must be aligned HWC vectors")
    if valid.shape != normals.shape[:2]:
        raise ValueError("valid_mask is not aligned with normals")
    finite = np.isfinite(normals).all(axis=-1)
    norm = np.linalg.norm(np.nan_to_num(normals), axis=-1)
    valid &= finite & (norm > 1e-6)
    normals = np.nan_to_num(normals) / np.maximum(norm[..., None], 1e-8)
    flip = np.sum(normals * viewing_direction, axis=-1) > 0.0
    normals[flip] *= -1.0
    normals[~valid] = 0.0
    return normals.astype(np.float32), valid


def build_cga_record(
    analyzers: np.ndarray,
    normal_gt: np.ndarray,
    normal_valid_mask: np.ndarray,
    *,
    camera_k: np.ndarray | None = None,
    refractive_index: float = 1.5,
    normal_orientation: str = "ray",
) -> dict[str, np.ndarray]:
    """Build one packed native-CGA record from aligned raw arrays."""
    if normal_orientation not in ("ray", "optical_axis"):
        raise ValueError("normal_orientation must be 'ray' or 'optical_axis'")
    height, width = analyzers.shape[1:3]
    viewing = viewing_directions(height, width, camera_k)
    orientation = viewing
    if normal_orientation == "optical_axis":
        viewing = np.broadcast_to(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32), (height, width, 3)
        ).copy()
        orientation = viewing
    normal_gt, normal_valid_mask = face_forward_normals(normal_gt, orientation, normal_valid_mask)
    observation = polarization_observation(analyzers)
    candidates = ambiguous_normals(
        observation["DoP"][0], observation["AoLP"], refractive_index=refractive_index
    )
    finite_observation = np.isfinite(analyzers).all(axis=(0, 3))
    normal_valid_mask &= finite_observation
    record = {
        **observation,
        "est": candidates,
        "image_coordinate": viewing.transpose(2, 0, 1).astype(np.float32),
        "label": normal_gt.transpose(2, 0, 1).astype(np.float32),
        "mask": normal_valid_mask[None].astype(np.float32),
        "normal_gt": normal_gt.astype(np.float32),
        "normal_valid_mask": normal_valid_mask.astype(np.uint8),
    }
    if camera_k is not None:
        record["K"] = np.asarray(camera_k, dtype=np.float32)
    record["physical_prior"] = np.concatenate(
        (record["est"], record["Iun"], record["spec"]), axis=0
    ).astype(np.float32)
    record["polar_observation"] = np.concatenate(
        (
            record["images"],
            record["Iun"],
            record["cos1"],
            record["cos2"],
            record["DoP"],
            record["image_coordinate"],
        ),
        axis=0,
    ).astype(np.float32)
    return record
