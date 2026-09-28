"""Create object-aware point-cloud failures across one full ``stack_wine`` frame."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_stack_wine_pointcloud_failure_demo import (
    boundary_confusion,
    camera_geometry,
    transparent_failure,
    write_ply,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "PTV3_wine_umbrella_realistic_missing_20260917"
    / "attention_inputs"
    / "stack_wine"
    / "severity_0.00.npz"
)
OUTPUT = ROOT / "stack_wine_full_scene_failure_demo"
SEED = 20260918

# Object roles were identified from the captured simulator instance mask,
# spatial location, color, and the explicitly saved wine-bottle target mask.
BOTTLE_ID = 82
ROBOT_IDS = (31, 34, 42, 43, 44, 45, 46)
TABLE_IDS = (48, 52)
RACK_IDS = (88, 90)


def rack_shape_distortion(
    points: np.ndarray,
    rack_mask: np.ndarray,
    extrinsics: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Coherently bend and twist the reconstructed thin rack structure."""
    corrupted = points.copy()
    labels = np.zeros(rack_mask.shape, dtype=np.uint8)
    xyz = points[rack_mask]
    z_min, z_max = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    height = np.clip((xyz[:, 2] - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0)
    warped = xyz.copy()
    warped[:, 0] += 0.030 * np.sin(np.pi * height)
    warped[:, 1] += 0.014 * np.sin(2.0 * np.pi * height)
    _origin, _ranges, unit_rays = camera_geometry(points, extrinsics)
    coherent_range_bias = 0.010 * np.sin(2.5 * np.pi * height)
    coherent_range_bias += rng.normal(0.0, 0.0015, len(xyz))
    warped += unit_rays[rack_mask] * coherent_range_bias[:, None]
    corrupted[rack_mask] = warped
    labels[rack_mask] = 1
    displacement = np.linalg.norm(warped - xyz, axis=-1)
    return corrupted, labels, {
        "model": "thin_structure_apparent_bend_and_twist",
        "deformed_target_points": int(rack_mask.sum()),
        "mean_displacement_m": float(displacement.mean()),
        "max_displacement_m": float(displacement.max()),
        "description": (
            "The reconstructed wine rack is coherently bent and twisted; "
            "the simulator rack mesh is unchanged."
        ),
    }


def floating_airborne_outliers(
    points: np.ndarray,
    candidate_mask: np.ndarray,
    rng: np.random.Generator,
    count: int = 420,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Relocate surface returns into free space as isolated and clustered outliers."""
    corrupted = points.copy()
    candidates = np.flatnonzero(candidate_mask)
    count = min(count, len(candidates))
    selected = rng.choice(candidates, size=count, replace=False)
    labels = np.zeros(candidate_mask.shape, dtype=np.uint8)
    labels.ravel()[selected] = 1

    isolated_count = int(round(0.65 * count))
    cloud = np.empty((count, 3), dtype=np.float32)
    cloud[:isolated_count, 0] = rng.uniform(-0.45, 0.65, isolated_count)
    cloud[:isolated_count, 1] = rng.uniform(-0.50, 0.35, isolated_count)
    cloud[:isolated_count, 2] = rng.uniform(0.82, 1.45, isolated_count)

    cluster_centers = np.asarray(
        [[0.18, -0.22, 1.12], [-0.12, 0.12, 1.30], [0.52, -0.08, 1.02]],
        dtype=np.float32,
    )
    cluster_ids = rng.integers(0, len(cluster_centers), count - isolated_count)
    cloud[isolated_count:] = (
        cluster_centers[cluster_ids]
        + rng.normal(0.0, [0.035, 0.035, 0.045], (count - isolated_count, 3))
    )
    corrupted.reshape(-1, 3)[selected] = cloud
    return corrupted, labels, {
        "model": "airborne_flying_outliers",
        "relocated_points": int(count),
        "isolated_points": int(isolated_count),
        "clustered_points": int(count - isolated_count),
        "description": "Table/background returns are relocated into free workspace air.",
    }


def compose_variants(
    clean: np.ndarray,
    instance: np.ndarray,
    bottle: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    robot = np.isin(instance, ROBOT_IDS)
    rack = np.isin(instance, RACK_IDS)
    table = np.isin(instance, TABLE_IDS)

    transparent, transparent_detail, transparent_stats = transparent_failure(
        clean, bottle, extrinsics, np.random.default_rng(SEED + 1)
    )
    shaped, shape_detail, shape_stats = rack_shape_distortion(
        clean, rack, extrinsics, np.random.default_rng(SEED + 2)
    )
    boundary, boundary_detail, boundary_stats = boundary_confusion(
        clean, robot, extrinsics, np.random.default_rng(SEED + 3)
    )
    floating, floating_detail, floating_stats = floating_airborne_outliers(
        clean, table, np.random.default_rng(SEED + 4)
    )

    variants = {
        "clean_full_scene": clean,
        "rack_shape_distortion": shaped,
        "bottle_transparent_failure": transparent,
        "robot_boundary_bridge": boundary,
        "floating_airborne_noise": floating,
    }
    provenance = {
        "clean_full_scene": np.zeros(instance.shape, dtype=np.uint8),
        "rack_shape_distortion": (shape_detail > 0).astype(np.uint8) * 1,
        "bottle_transparent_failure": (transparent_detail > 0).astype(np.uint8) * 2,
        "robot_boundary_bridge": (boundary_detail > 0).astype(np.uint8) * 3,
        "floating_airborne_noise": (floating_detail > 0).astype(np.uint8) * 4,
    }

    combined = clean.copy()
    combined_provenance = np.zeros(instance.shape, dtype=np.uint8)
    for name, code in (
        ("rack_shape_distortion", 1),
        ("bottle_transparent_failure", 2),
        ("robot_boundary_bridge", 3),
        ("floating_airborne_noise", 4),
    ):
        changed = provenance[name] > 0
        # Preserve the material-specific bottle and rack models if a boundary
        # band happens to overlap their image regions.
        if name == "robot_boundary_bridge":
            changed &= ~(bottle | rack)
        combined[changed] = variants[name][changed]
        combined_provenance[changed] = code
    variants["combined_object_aware_failures"] = combined
    provenance["combined_object_aware_failures"] = combined_provenance

    stats = {
        "object_masks": {
            "wine_bottle": {"instance_ids": [BOTTLE_ID], "pixels": int(bottle.sum())},
            "robot_links": {"instance_ids": list(ROBOT_IDS), "pixels": int(robot.sum())},
            "wine_rack": {"instance_ids": list(RACK_IDS), "pixels": int(rack.sum())},
            "table": {"instance_ids": list(TABLE_IDS), "pixels": int(table.sum())},
        },
        "assignments": {
            "wine_bottle": transparent_stats,
            "wine_rack": shape_stats,
            "robot_links": boundary_stats,
            "table_background_samples": floating_stats,
        },
        "combined_changed_pixels_by_failure": {
            "rack_shape_distortion": int((combined_provenance == 1).sum()),
            "bottle_transparent_failure": int((combined_provenance == 2).sum()),
            "robot_boundary_bridge": int((combined_provenance == 3).sum()),
            "floating_airborne_noise": int((combined_provenance == 4).sum()),
        },
    }
    return variants, provenance, stats


def render_full_scene(
    path: Path,
    rgb: np.ndarray,
    extrinsics: np.ndarray,
    variants: dict[str, np.ndarray],
    provenance: dict[str, np.ndarray],
) -> None:
    names = list(variants)
    origin = extrinsics[:3, 3]
    clean_range = np.linalg.norm(variants["clean_full_scene"] - origin, axis=-1)
    range_limits = np.percentile(clean_range, (1, 99))
    colors_by_code = np.asarray(
        [[0, 0, 0], [220, 50, 47], [38, 139, 210], [181, 137, 0], [211, 54, 130]],
        dtype=np.float32,
    ) / 255.0

    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 8.0))
    for column, name in enumerate(names):
        points = variants[name]
        labels = provenance[name]
        valid = np.isfinite(points).all(axis=-1)
        ranges = np.linalg.norm(points - origin, axis=-1)
        shown = ranges.copy()
        shown[~valid] = np.nan
        axes[0, column].imshow(shown, cmap="turbo", vmin=range_limits[0], vmax=range_limits[1])
        changed_y, changed_x = np.where(labels > 0)
        if len(changed_x):
            step = max(1, len(changed_x) // 1200)
            axes[0, column].scatter(
                changed_x[::step], changed_y[::step], s=1.2,
                c=colors_by_code[labels[changed_y[::step], changed_x[::step]]], alpha=0.7,
            )
        axes[0, column].set_title(name.replace("_", " "), fontsize=10)
        axes[0, column].axis("off")

        workspace = (
            valid
            & (points[..., 0] > -0.65) & (points[..., 0] < 0.75)
            & (points[..., 1] > -0.65) & (points[..., 1] < 0.55)
            & (points[..., 2] > 0.60) & (points[..., 2] < 1.55)
        )
        ids = np.flatnonzero(workspace)
        if len(ids) > 18000:
            ids = ids[np.linspace(0, len(ids) - 1, 18000, dtype=int)]
        xyz = points.reshape(-1, 3)[ids]
        point_colors = rgb.reshape(-1, 3)[ids] / 255.0
        axes[1, column].scatter(xyz[:, 0], xyz[:, 2], c=point_colors, s=0.35, linewidths=0)
        changed = labels.reshape(-1)[ids]
        selected = changed > 0
        if selected.any():
            axes[1, column].scatter(
                xyz[selected, 0], xyz[selected, 2], c=colors_by_code[changed[selected]],
                s=2.2, linewidths=0, alpha=0.75,
            )
        axes[1, column].set_xlim(-0.65, 0.75)
        axes[1, column].set_ylim(0.60, 1.55)
        axes[1, column].set_aspect("equal", adjustable="box")
        axes[1, column].set_xlabel("world x (m)")
        axes[1, column].set_ylabel("world z (m)")
        axes[1, column].grid(alpha=0.15)
    fig.suptitle("stack_wine full scene: object-aware point-cloud failure simulation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    with np.load(SOURCE) as source:
        rgb = source["rgb"]
        clean = source["clean_points"].astype(np.float32)
        instance = source["instance_mask"].astype(np.int64)
        bottle = source["target_mask"].astype(bool)
        extrinsics = source["camera_extrinsics"].astype(np.float32)

    variants, provenance, stats = compose_variants(clean, instance, bottle, extrinsics)
    for name, points in variants.items():
        np.savez_compressed(
            OUTPUT / f"{name}.npz",
            rgb=rgb,
            points=points,
            failure_provenance=provenance[name],
            instance_mask=instance,
            camera_extrinsics=extrinsics,
        )
        write_ply(OUTPUT / f"{name}.ply", points, rgb)

    render_full_scene(OUTPUT / "full_scene_comparison.png", rgb, extrinsics, variants, provenance)
    metadata = {
        "task": "stack_wine",
        "frame": "initial frame, variation 0, scene seed 7",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_unchanged": True,
        "random_seed": SEED,
        "full_scene_shape": list(clean.shape),
        "failure_provenance_codes": {
            "0": "unchanged",
            "1": "wine-rack apparent shape distortion",
            "2": "wine-bottle transparent/refraction failure",
            "3": "robot/background boundary bridge",
            "4": "floating airborne outlier",
        },
        **stats,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
