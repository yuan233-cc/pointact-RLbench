"""Build a one-frame, simulator-mask-aware point-cloud failure demonstration.

The input is an already captured RLBench ``stack_wine`` frame.  Its organized
front-camera XYZ image, RGB image, instance-derived wine-bottle mask, and camera
extrinsics let us model image-boundary and viewing-ray failures that cannot be
recovered from an unordered point cloud alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "PTV3_wine_umbrella_realistic_missing_20260917"
    / "attention_inputs"
    / "stack_wine"
    / "severity_0.00.npz"
)
OUTPUT = ROOT / "stack_wine_one_frame_failure_demo"
SEED = 20260918


def choose(mask: np.ndarray, score: np.ndarray, fraction: float) -> np.ndarray:
    candidates = np.flatnonzero(mask)
    count = min(len(candidates), int(round(fraction * len(candidates))))
    result = np.zeros_like(mask, dtype=bool)
    if count:
        picked = candidates[np.argpartition(score.ravel()[candidates], count - 1)[:count]]
        result.ravel()[picked] = True
    return result


def camera_geometry(points: np.ndarray, extrinsics: np.ndarray):
    origin = np.asarray(extrinsics[:3, 3], dtype=np.float32)
    rays = points - origin
    ranges = np.linalg.norm(rays, axis=-1)
    unit_rays = rays / np.maximum(ranges[..., None], 1e-8)
    return origin, ranges, unit_rays


def transparent_failure(
    points: np.ndarray,
    target: np.ndarray,
    extrinsics: np.ndarray,
    rng: np.random.Generator,
):
    """Connected dropout, background leakage, and refraction-like range errors."""
    corrupted = points.copy()
    labels = np.zeros(target.shape, dtype=np.uint8)
    edge = target & ~binary_erosion(target, iterations=3, border_value=0)
    smooth = gaussian_filter(rng.standard_normal(target.shape), sigma=6.0, mode="reflect")
    smooth[edge] -= 1.25 * max(float(smooth.std()), 1e-6)
    selected = choose(target, smooth, 0.75)

    category = gaussian_filter(rng.standard_normal(target.shape), sigma=4.0, mode="reflect")
    ordered = np.flatnonzero(selected)
    ordered = ordered[np.argsort(category.ravel()[ordered])]
    n_invalid = int(round(0.55 * len(ordered)))
    n_background = int(round(0.35 * len(ordered)))
    invalid = np.zeros_like(target)
    background = np.zeros_like(target)
    distorted = np.zeros_like(target)
    invalid.ravel()[ordered[:n_invalid]] = True
    background.ravel()[ordered[n_invalid : n_invalid + n_background]] = True
    distorted.ravel()[ordered[n_invalid + n_background :]] = True

    origin, ranges, unit_rays = camera_geometry(points, extrinsics)
    nearest_background = distance_transform_edt(
        target, return_distances=False, return_indices=True
    )
    background_points = points[nearest_background[0], nearest_background[1]]
    background_ranges = np.linalg.norm(background_points - origin, axis=-1)
    leaked_ranges = np.maximum(background_ranges, ranges + 0.025)
    leaked_ranges += rng.normal(0.0, 0.004, target.shape)
    corrupted[background] = origin + unit_rays[background] * leaked_ranges[background, None]

    delta = rng.normal(0.025, 0.018, target.shape) * rng.choice((-1.0, 1.0), target.shape)
    distorted_ranges = np.maximum(ranges + delta, 0.05)
    corrupted[distorted] = origin + unit_rays[distorted] * distorted_ranges[distorted, None]
    corrupted[invalid] = np.nan
    labels[invalid], labels[background], labels[distorted] = 1, 2, 3
    return corrupted, labels, {
        "model": "transparent_like",
        "target_fraction_affected": float(selected.sum() / target.sum()),
        "invalid_target_points": int(invalid.sum()),
        "background_leakage_points": int(background.sum()),
        "refracted_range_points": int(distorted.sum()),
    }


def boundary_confusion(
    points: np.ndarray,
    target: np.ndarray,
    extrinsics: np.ndarray,
    rng: np.random.Generator,
):
    """Warp a flat background into a continuous bridge to the object boundary.

    Mixed pixels from an RGB-D sensor often interpolate across a depth
    discontinuity.  The first background pixels outside the silhouette are
    therefore pulled strongly toward the object surface, and progressively
    return to their original background range farther from the object.
    """
    corrupted = points.copy()
    labels = np.zeros(target.shape, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=-1)
    inner = target & ~binary_erosion(target, iterations=2, border_value=0)
    distance_outside = distance_transform_edt(~target)
    band_width = 7.0
    outer_band = (~target) & finite & (distance_outside <= band_width)

    origin, ranges, unit_rays = camera_geometry(points, extrinsics)
    nearest_background = distance_transform_edt(
        target, return_distances=False, return_indices=True
    )
    nearest_target = distance_transform_edt(
        ~target, return_distances=False, return_indices=True
    )
    background_points = points[nearest_background[0], nearest_background[1]]
    target_points = points[nearest_target[0], nearest_target[1]]
    background_ranges = np.linalg.norm(background_points - origin, axis=-1)
    target_ranges = np.linalg.norm(target_points - origin, axis=-1)

    # At the silhouette, use mostly object range.  Across seven background
    # pixels, smoothly decay to the original flat background range.  Smooth
    # low-amplitude noise avoids an artificially perfect interpolation ramp.
    blend = np.clip((band_width + 1.0 - distance_outside) / band_width, 0.0, 1.0)
    blend = np.square(blend)
    smooth_noise = gaussian_filter(rng.standard_normal(target.shape), sigma=2.0)
    mixed_ranges = (
        blend * target_ranges
        + (1.0 - blend) * ranges
        + 0.0025 * smooth_noise
    )
    corrupted[outer_band] = (
        origin + unit_rays[outer_band] * mixed_ranges[outer_band, None]
    )

    # A narrow strip just inside the object is biased slightly toward the
    # background too, completing the continuous object/background connection.
    inner_ranges = 0.82 * ranges + 0.18 * background_ranges
    corrupted[inner] = origin + unit_rays[inner] * inner_ranges[inner, None]
    labels[inner], labels[outer_band] = 2, 3
    return corrupted, labels, {
        "model": "continuous_object_background_depth_bridge",
        "background_warp_band_pixels": int(outer_band.sum()),
        "background_warp_band_width_pixels": int(band_width),
        "inner_boundary_pixels": int(inner.sum()),
        "invalid_points": 0,
        "description": (
            "The flat background range is smoothly pulled toward the nearest "
            "object range, forming a connected non-flat surface at the boundary."
        ),
    }


def reflective_failure(
    points: np.ndarray,
    target: np.ndarray,
    extrinsics: np.ndarray,
    rng: np.random.Generator,
):
    """Patchy dropout, range spikes, and flying points on a reflective surface."""
    corrupted = points.copy()
    labels = np.zeros(target.shape, dtype=np.uint8)
    smooth = gaussian_filter(rng.standard_normal(target.shape), sigma=3.5, mode="reflect")
    affected = choose(target, smooth, 0.65)
    dropout = choose(affected, rng.random(target.shape), 0.35)
    remaining = affected & ~dropout
    spikes = choose(remaining, rng.random(target.shape), 0.65)
    flying = remaining & ~spikes

    origin, ranges, unit_rays = camera_geometry(points, extrinsics)
    spike_delta = rng.choice((-1.0, 1.0), spikes.sum()) * rng.uniform(0.02, 0.10, spikes.sum())
    spike_ranges = np.maximum(ranges[spikes] + spike_delta, 0.05)
    corrupted[spikes] = origin + unit_rays[spikes] * spike_ranges[:, None]
    corrupted[flying] += rng.normal(0.0, 0.025, (flying.sum(), 3)).astype(np.float32)
    corrupted[dropout] = np.nan
    labels[dropout], labels[spikes], labels[flying] = 1, 2, 3
    return corrupted, labels, {
        "model": "reflective_metal_like",
        "target_fraction_affected": float(affected.sum() / target.sum()),
        "dropout_points": int(dropout.sum()),
        "range_spike_points": int(spikes.sum()),
        "flying_points": int(flying.sum()),
    }


def apparent_shape_distortion(
    points: np.ndarray,
    target: np.ndarray,
    extrinsics: np.ndarray,
    rng: np.random.Generator,
):
    """Non-rigidly warp the measured bottle while leaving the true mesh fixed.

    A smooth range bias dents and swells different height bands, while a
    lateral drift bends the neck.  This models a coherent reconstruction error
    instead of physical deformation of the simulated object.
    """
    corrupted = points.copy()
    labels = np.zeros(target.shape, dtype=np.uint8)
    xyz = points[target]
    z_min, z_max = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    height = np.clip((xyz[:, 2] - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0)
    center = xyz.mean(axis=0)

    # Swell the lower body, pinch the shoulder, and make the neck lean.  The
    # deformation amplitudes are intentionally strong enough to inspect in one
    # frame but remain in the centimetre range of severe RGB-D artifacts.
    radial_scale = (
        1.0
        + 0.32 * np.exp(-np.square((height - 0.30) / 0.20))
        - 0.18 * np.exp(-np.square((height - 0.68) / 0.13))
    )
    warped = xyz.copy()
    warped[:, :2] = center[:2] + (xyz[:, :2] - center[:2]) * radial_scale[:, None]
    warped[:, 0] += 0.030 * np.square(height) + 0.008 * np.sin(3.0 * np.pi * height)

    origin, ranges, unit_rays = camera_geometry(points, extrinsics)
    smooth_bias = gaussian_filter(rng.standard_normal(target.shape), sigma=8.0)
    bias_values = smooth_bias[target]
    bias_values /= max(float(np.std(bias_values)), 1e-6)
    warped += unit_rays[target] * (0.012 * bias_values)[:, None]
    corrupted[target] = warped
    labels[target] = 1
    displacement = np.linalg.norm(warped - xyz, axis=-1)
    return corrupted, labels, {
        "model": "apparent_nonrigid_shape_distortion",
        "deformed_target_points": int(target.sum()),
        "mean_displacement_m": float(displacement.mean()),
        "max_displacement_m": float(displacement.max()),
        "description": (
            "Measured bottle points are coherently swollen, pinched, and bent; "
            "the simulator object mesh is unchanged."
        ),
    }


def write_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> int:
    valid = np.isfinite(points).all(axis=-1)
    xyz = points[valid].astype(np.float32)
    colors = rgb[valid].astype(np.uint8)
    with path.open("wb") as handle:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(xyz)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        handle.write(header.encode("ascii"))
        packed = np.empty(
            len(xyz),
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")],
        )
        packed["x"], packed["y"], packed["z"] = xyz.T
        packed["r"], packed["g"], packed["b"] = colors.T
        packed.tofile(handle)
    return len(xyz)


def target_crop(target: np.ndarray, pad: int = 16):
    yy, xx = np.where(target)
    return (
        slice(max(0, yy.min() - pad), min(target.shape[0], yy.max() + pad + 1)),
        slice(max(0, xx.min() - pad), min(target.shape[1], xx.max() + pad + 1)),
    )


def render_comparison(
    path: Path,
    rgb: np.ndarray,
    target: np.ndarray,
    extrinsics: np.ndarray,
    variants: dict[str, np.ndarray],
):
    crop = target_crop(target)
    origin = extrinsics[:3, 3]
    clean_ranges = np.linalg.norm(variants["clean"] - origin, axis=-1)
    lo, hi = np.percentile(clean_ranges[crop], (2, 98))
    spatial_roi = binary_dilation(target, iterations=7)
    target_xyz = variants["clean"][target]
    x_limits = (target_xyz[:, 0].min() - 0.12, target_xyz[:, 0].max() + 0.12)
    z_limits = (target_xyz[:, 2].min() - 0.10, target_xyz[:, 2].max() + 0.10)
    fig, axes = plt.subplots(2, len(variants), figsize=(4.0 * len(variants), 7.2))
    for column, (name, points) in enumerate(variants.items()):
        ranges = np.linalg.norm(points - origin, axis=-1)
        invalid = ~np.isfinite(points).all(axis=-1)
        shown = ranges[crop].copy()
        shown[invalid[crop]] = np.nan
        axes[0, column].imshow(shown, cmap="turbo", vmin=lo, vmax=hi)
        axes[0, column].set_title(name.replace("_", " "))
        axes[0, column].axis("off")

        valid = ~invalid & spatial_roi
        ids = np.flatnonzero(valid)
        if len(ids) > 10000:
            ids = ids[np.linspace(0, len(ids) - 1, 10000, dtype=int)]
        xyz = points.reshape(-1, 3)[ids]
        colors = rgb.reshape(-1, 3)[ids] / 255.0
        axes[1, column].scatter(xyz[:, 0], xyz[:, 2], c=colors, s=0.35, linewidths=0)
        axes[1, column].set_aspect("equal", adjustable="box")
        axes[1, column].set_xlim(*x_limits)
        axes[1, column].set_ylim(*z_limits)
        axes[1, column].set_xlabel("world x (m)")
        axes[1, column].set_ylabel("world z (m)")
        axes[1, column].grid(alpha=0.15)
    fig.suptitle("RLBench stack_wine: clean and simulated RGB-D point-cloud failures")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    source = np.load(SOURCE)
    rgb = source["rgb"]
    clean = source["clean_points"].astype(np.float32)
    target = source["target_mask"].astype(bool)
    extrinsics = source["camera_extrinsics"].astype(np.float32)
    if clean.shape[:2] != target.shape or not target.any():
        raise ValueError("The simulator-organized XYZ and wine-bottle mask are inconsistent")

    generators = {
        "transparent_like": transparent_failure,
        "boundary_confusion": boundary_confusion,
        "reflective_metal_like": reflective_failure,
        "apparent_shape_distorted": apparent_shape_distortion,
    }
    variants = {"clean": clean}
    labels = {"clean": np.zeros(target.shape, dtype=np.uint8)}
    stats = {
        "clean": {"model": "none", "valid_points": int(np.prod(target.shape))},
    }
    for index, (name, function) in enumerate(generators.items(), start=1):
        variants[name], labels[name], stats[name] = function(
            clean, target, extrinsics, np.random.default_rng(SEED + index)
        )

    point_counts = {}
    for name, points in variants.items():
        np.savez_compressed(
            OUTPUT / f"{name}.npz",
            rgb=rgb,
            points=points,
            target_mask=target,
            failure_labels=labels[name],
            camera_extrinsics=extrinsics,
        )
        point_counts[name] = write_ply(OUTPUT / f"{name}.ply", points, rgb)
        stats[name]["valid_points"] = point_counts[name]

    render_comparison(OUTPUT / "comparison.png", rgb, target, extrinsics, variants)
    requested = {
        name: variants[name]
        for name in (
            "clean",
            "apparent_shape_distorted",
            "transparent_like",
            "boundary_confusion",
        )
    }
    render_comparison(
        OUTPUT / "requested_three_failures.png",
        rgb,
        target,
        extrinsics,
        requested,
    )
    metadata = {
        "task": "stack_wine",
        "treated_object": "wine_bottle",
        "frame": "initial frame, variation 0, scene seed 7",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_unchanged": True,
        "organized_shape": list(clean.shape),
        "visible_target_pixels": int(target.sum()),
        "random_seed": SEED,
        "outputs": stats,
        "label_meaning": {
            "0": "unchanged",
            "1": "invalid/dropout or shape-deformed target (variant-dependent)",
            "2": "background leakage, inner-boundary bias, or range spike (variant-dependent)",
            "3": "refraction, warped background bridge, or flying point (variant-dependent)",
        },
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
