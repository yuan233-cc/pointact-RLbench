#!/usr/bin/env python3
"""Convert RLBench Mitsuba, SfPUEL, or HAMMER samples to native-CGA NPZ records."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointact.data.polar_normal_sources import (  # noqa: E402
    ambiguous_normals,
    build_cga_record,
    normals_from_depth,
    polarization_observation,
    stack_analyzers,
)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Failed to read {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_record(
    destination: Path,
    record: dict[str, np.ndarray],
    *,
    dataset_id: str,
    source_id: str,
) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    payload = {key: np.asarray(value) for key, value in record.items()}
    payload["dataset_id"] = np.asarray(dataset_id)
    payload["source_id"] = np.asarray(source_id)
    np.savez_compressed(destination, **payload)


def write_manifest(output: Path, entries: list[dict], metadata: dict) -> None:
    (output / "manifest.json").write_text(json.dumps({"samples": entries}, indent=2) + "\n")
    (output / "conversion.json").write_text(json.dumps(metadata, indent=2) + "\n")


def prepare_rlbench(args: argparse.Namespace) -> None:
    sources = sorted(args.input_root.glob("frame_*/polar_data.npz"))
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        raise ValueError(f"No frame_*/polar_data.npz files below {args.input_root}")
    args.output.mkdir(parents=True, exist_ok=False)
    entries = []
    skipped = []
    for source in sources:
        with np.load(source, allow_pickle=False) as data:
            required = (
                "I0", "I45", "I90", "I135", "normal_gt", "normal_valid_mask", "K", "depth_z"
            )
            missing = [key for key in required if key not in data]
            if missing:
                skipped.append({"path": str(source), "missing": missing})
                continue
            camera_k = np.asarray(data["K"], dtype=np.float32).copy()
            if camera_k.shape != (3, 3) or not np.isfinite(camera_k).all():
                raise ValueError(f"Invalid RLBench camera intrinsics in {source}")
            if camera_k[0, 0] >= 0 or camera_k[1, 1] >= 0:
                raise ValueError(f"Expected the source RLBench/Mitsuba negative focal convention: {source}")
            camera_k[0, 0] = abs(camera_k[0, 0])
            camera_k[1, 1] = abs(camera_k[1, 1])
            normal_gt = np.asarray(data["normal_gt"], dtype=np.float32).copy()
            normal_gt[..., :2] *= -1.0
            depth_normal, depth_mask = normals_from_depth(
                data["depth_z"],
                camera_k,
                relative_edge_threshold=0.01,
                absolute_edge_threshold=0.003,
            )
            agreement = np.sum(normal_gt * depth_normal, axis=-1)
            mask = (
                np.asarray(data["normal_valid_mask"], dtype=bool)
                & depth_mask
                & (agreement >= np.cos(np.deg2rad(args.max_geometry_angle)))
            )
            analyzers = stack_analyzers(
                [data["I0"], data["I45"], data["I90"], data["I135"]],
                normalization="percentile",
                valid_mask=mask,
            )
            record = build_cga_record(
                analyzers,
                normal_gt,
                mask,
                camera_k=camera_k,
                refractive_index=args.refractive_index,
            )
        destination = args.output / f"{len(entries):06d}_{source.parent.name}.npz"
        write_record(
            destination,
            record,
            dataset_id="rlbench_mitsuba",
            source_id=str(source),
        )
        entries.append(
            {
                "path": destination.name,
                "group": args.group,
                "id": source.parent.name,
                "dataset": "rlbench_mitsuba",
            }
        )
    if not entries:
        raise ValueError(f"No RLBench source contained normal supervision; skipped={skipped}")
    write_manifest(
        args.output,
        entries,
        {
            "adapter": "rlbench_mitsuba",
            "source": str(args.input_root.resolve()),
            "coordinate_frame": "+x right, +y down, +z forward; normals face camera",
            "source_normal_xy_to_camera": [-1, -1],
            "source_negative_focal_to_camera": "take positive fx and fy",
            "normal_aov_depth_consistency_degrees": args.max_geometry_angle,
            "normal_supervision": "Mitsuba shading-normal AOV after camera-axis conversion and clean-depth geometry check",
            "analyzer_order_degrees": [0, 45, 90, 135],
            "refractive_index": args.refractive_index,
            "records": len(entries),
            "skipped": skipped,
        },
    )


def sfpuel_group(stem: str, group_prefix_components: int) -> str:
    parts = stem.split("_")
    count = min(group_prefix_components, len(parts))
    return "sfpuel_" + "_".join(parts[:count])


def parse_hammer_layout(value: str) -> tuple[int, int, int, int]:
    try:
        layout = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layout must contain comma-separated integer angles") from exc
    if len(layout) != 4 or set(layout) != {0, 45, 90, 135}:
        raise argparse.ArgumentTypeError(
            "layout must assign 0,45,90,135 to TL,TR,BL,BR exactly once"
        )
    return layout


def hammer_sources(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.glob("scene*_traj*/polarization/pol/*.png")
        if "naked" not in path.parents[2].name
    )


def hammer_quadrants(path: Path) -> list[np.ndarray]:
    mosaic = read_rgb(path)
    height, width = mosaic.shape[:2]
    if height % 2 or width % 2:
        raise ValueError(f"HAMMER polarization mosaic has odd dimensions: {path} {mosaic.shape}")
    return [
        mosaic[: height // 2, : width // 2],
        mosaic[: height // 2, width // 2 :],
        mosaic[height // 2 :, : width // 2],
        mosaic[height // 2 :, width // 2 :],
    ]


def order_hammer_analyzers(quadrants: list[np.ndarray], layout: tuple[int, ...]) -> list[np.ndarray]:
    by_angle = {angle: quadrants[index] for index, angle in enumerate(layout)}
    return [by_angle[angle] for angle in (0, 45, 90, 135)]


def hammer_intrinsics(pol_path: Path) -> np.ndarray:
    path = pol_path.parent.parent / "intrinsics.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing HAMMER polarization intrinsics: {path}")
    values = np.loadtxt(path, dtype=np.float32)
    if values.shape == (3, 3):
        camera_k = values
    elif values.size == 4:
        fx, fy, cx, cy = values.reshape(-1)
        camera_k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    else:
        raise ValueError(f"Unsupported HAMMER intrinsics shape {values.shape}: {path}")
    if not np.isfinite(camera_k).all() or camera_k[0, 0] <= 0 or camera_k[1, 1] <= 0:
        raise ValueError(f"Invalid HAMMER intrinsics: {path}")
    return camera_k.astype(np.float32)


def hammer_supervision(pol_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    polarization = pol_path.parent.parent
    depth_path = polarization / "_gt" / pol_path.name
    if not depth_path.is_file():
        raise FileNotFoundError(f"Missing HAMMER clean GT depth image: {depth_path}")
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise OSError(f"Failed to read {depth_path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    camera_k = hammer_intrinsics(pol_path)
    normal, mask = normals_from_depth(depth, camera_k, depth_scale=0.001)
    return normal, mask, camera_k


def strided_camera_intrinsics(camera_k: np.ndarray, stride: int) -> np.ndarray:
    """Keep rays aligned with source pixel centers selected at 0, stride, 2*stride, ..."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    result = np.asarray(camera_k, dtype=np.float32).copy()
    result[0] /= stride
    result[1] /= stride
    result[0, 2] += 0.5 - 0.5 / stride
    result[1, 2] += 0.5 - 0.5 / stride
    return result


def hammer_group(pol_path: Path) -> str:
    # Different trajectories of the same scene share geometry and must not
    # cross the train/validation boundary.
    return "hammer_" + pol_path.parents[2].name.split("_", 1)[0]


def select_per_group(sources: list[Path], maximum: int | None) -> list[Path]:
    if maximum is None:
        return sources
    counts: dict[str, int] = {}
    selected = []
    for source in sources:
        group = hammer_group(source)
        count = counts.get(group, 0)
        if count < maximum:
            selected.append(source)
            counts[group] = count + 1
    return selected


def prepare_hammer(args: argparse.Namespace) -> None:
    sources = select_per_group(hammer_sources(args.input_root), args.max_per_group)
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        raise ValueError(
            f"No non-naked scene*_traj*/polarization/pol/*.png files below {args.input_root}"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    entries = []
    for source in sources:
        quadrants = hammer_quadrants(source)
        normal, mask, camera_k = hammer_supervision(source)
        if args.spatial_stride > 1:
            quadrants = [image[:: args.spatial_stride, :: args.spatial_stride] for image in quadrants]
            normal = normal[:: args.spatial_stride, :: args.spatial_stride]
            mask = mask[:: args.spatial_stride, :: args.spatial_stride]
            camera_k = strided_camera_intrinsics(camera_k, args.spatial_stride)
        analyzers = stack_analyzers(
            order_hammer_analyzers(quadrants, args.quadrant_layout),
            normalization="dtype",
        )
        if normal.shape[:2] != analyzers.shape[1:3] or mask.shape != analyzers.shape[1:3]:
            raise ValueError(
                f"HAMMER modalities are not aligned for {source}: "
                f"analyzers={analyzers.shape}, normal={normal.shape}, mask={mask.shape}"
            )
        record = build_cga_record(
            analyzers,
            normal,
            mask,
            camera_k=camera_k,
            refractive_index=args.refractive_index,
        )
        destination = args.output / f"{len(entries):06d}_{source.stem}.npz"
        write_record(destination, record, dataset_id="hammer", source_id=str(source))
        entries.append(
            {
                "path": destination.name,
                "group": hammer_group(source),
                "id": f"{source.parent.parent.name}_{source.stem}",
                "dataset": "hammer",
            }
        )
    write_manifest(
        args.output,
        entries,
        {
            "adapter": "hammer",
            "source": str(args.input_root.resolve()),
            "coordinate_frame": "+x right, +y down, +z forward; normals face camera",
            "quadrant_positions": ["top_left", "top_right", "bottom_left", "bottom_right"],
            "quadrant_layout_degrees": list(args.quadrant_layout),
            "layout_status": "explicitly selected after empirical calibration",
            "normal_supervision": "centered finite-difference normals from clean _gt depth",
            "depth_scale_to_meters": 0.001,
            "spatial_stride": args.spatial_stride,
            "refractive_index": args.refractive_index,
            "max_per_group": args.max_per_group,
            "records": len(entries),
        },
    )


def calibrate_hammer(args: argparse.Namespace) -> None:
    sources = select_per_group(hammer_sources(args.input_root), args.max_per_group)
    sources = sources[: args.limit]
    if not sources:
        raise ValueError(f"No HAMMER samples below {args.input_root}")
    samples = []
    for source in sources:
        normal, mask, _ = hammer_supervision(source)
        quadrants = hammer_quadrants(source)
        if args.stride > 1:
            normal = normal[:: args.stride, :: args.stride]
            mask = mask[:: args.stride, :: args.stride]
            quadrants = [image[:: args.stride, :: args.stride] for image in quadrants]
        samples.append((source, quadrants, normal, mask))
    scores = []
    for layout in itertools.permutations((0, 45, 90, 135)):
        errors = []
        for _, quadrants, normal, mask in samples:
            analyzers = stack_analyzers(order_hammer_analyzers(quadrants, layout))
            observation = polarization_observation(analyzers)
            candidates = ambiguous_normals(
                observation["DoP"][0],
                observation["AoLP"],
                refractive_index=args.refractive_index,
            ).reshape(3, 3, *mask.shape)
            dots = np.einsum("kchw,hwc->khw", candidates, normal)
            best = np.clip(dots.max(axis=0), -1.0, 1.0)
            valid = mask & (observation["DoP"][0] >= args.minimum_dolp)
            if np.any(valid):
                errors.append(np.degrees(np.arccos(best[valid])))
        merged = np.concatenate(errors)
        scores.append(
            {
                "layout_tl_tr_bl_br": list(layout),
                "mean_candidate_error_degrees": float(np.mean(merged)),
                "median_candidate_error_degrees": float(np.median(merged)),
                "valid_pixels": int(merged.size),
            }
        )
    scores.sort(key=lambda item: item["mean_candidate_error_degrees"])
    print(json.dumps({"samples": [str(item[0]) for item in samples], "scores": scores}, indent=2))


def prepare_sfpuel(args: argparse.Namespace) -> None:
    sources = sorted((args.input_root / "pol000").glob("*.png"))
    if args.max_per_group is not None:
        counts: dict[str, int] = {}
        selected = []
        for source in sources:
            group = sfpuel_group(source.stem, args.group_prefix_components)
            count = counts.get(group, 0)
            if count < args.max_per_group:
                selected.append(source)
                counts[group] = count + 1
        sources = selected
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        raise ValueError(f"No pol000/*.png files below {args.input_root}")
    args.output.mkdir(parents=True, exist_ok=False)
    entries = []
    for pol000 in sources:
        name = pol000.name
        paths = {
            angle: args.input_root / f"pol{angle}" / name
            for angle in ("000", "045", "090", "135")
        }
        normal_path = args.input_root / "normal" / name
        mask_path = args.input_root / "mask" / name
        missing = [path for path in (*paths.values(), normal_path, mask_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete SfPUEL sample {name}: {missing}")
        analyzer_images = [read_rgb(paths[angle]) for angle in ("000", "045", "090", "135")]
        normal_image = read_rgb(normal_path)
        if not np.issubdtype(normal_image.dtype, np.integer):
            raise TypeError(f"Expected integer-encoded SfPUEL normal image: {normal_path}")
        normal = normal_image.astype(np.float32) / float(np.iinfo(normal_image.dtype).max)
        normal = normal * 2.0 - 1.0
        # SfPUEL's PNG uses image-right x, image-up y, and camera-facing +z.
        # Convert to image-right x, image-down y, and forward +z.
        normal[..., 1:] *= -1.0
        mask_image = read_rgb(mask_path)
        mask = mask_image[..., 0] > 0 if mask_image.ndim == 3 else mask_image > 0
        if args.spatial_stride > 1:
            analyzer_images = [
                image[:: args.spatial_stride, :: args.spatial_stride] for image in analyzer_images
            ]
            normal = normal[:: args.spatial_stride, :: args.spatial_stride]
            mask = mask[:: args.spatial_stride, :: args.spatial_stride]
        analyzers = stack_analyzers(analyzer_images, normalization="dtype")
        record = build_cga_record(
            analyzers,
            normal,
            mask,
            refractive_index=args.refractive_index,
            normal_orientation="optical_axis",
        )
        destination = args.output / f"{len(entries):06d}_{pol000.stem}.npz"
        write_record(destination, record, dataset_id="sfpuel", source_id=str(pol000))
        entries.append(
            {
                "path": destination.name,
                "group": sfpuel_group(pol000.stem, args.group_prefix_components),
                "id": pol000.stem,
                "dataset": "sfpuel",
            }
        )
    write_manifest(
        args.output,
        entries,
        {
            "adapter": "sfpuel",
            "source": str(args.input_root.resolve()),
            "coordinate_frame": "+x right, +y down, +z forward; normals face camera",
            "analyzer_order_degrees": [0, 45, 90, 135],
            "normal_decode": "official RGB image / dtype_max * 2 - 1; (x,y,z) to (x,-y,-z)",
            "normal_orientation": "optical axis because camera intrinsics are unavailable",
            "refractive_index": args.refractive_index,
            "group_prefix_components": args.group_prefix_components,
            "max_per_group": args.max_per_group,
            "spatial_stride": args.spatial_stride,
            "records": len(entries),
        },
    )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refractive-index", type=float, default=1.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    rlbench = subparsers.add_parser("rlbench")
    add_common(rlbench)
    rlbench.add_argument("--group", required=True, help="Complete RLBench task/episode/seed group")
    rlbench.add_argument("--max-geometry-angle", type=float, default=15.0)
    rlbench.set_defaults(function=prepare_rlbench)
    sfpuel = subparsers.add_parser("sfpuel")
    add_common(sfpuel)
    sfpuel.add_argument("--group-prefix-components", type=int, default=1)
    sfpuel.add_argument("--max-per-group", type=int)
    sfpuel.add_argument("--spatial-stride", type=int, default=1)
    sfpuel.set_defaults(function=prepare_sfpuel)
    hammer = subparsers.add_parser("hammer")
    add_common(hammer)
    hammer.add_argument(
        "--quadrant-layout",
        type=parse_hammer_layout,
        required=True,
        help="Angles at top-left,top-right,bottom-left,bottom-right, e.g. 0,45,90,135",
    )
    hammer.add_argument("--max-per-group", type=int)
    hammer.add_argument("--spatial-stride", type=int, default=4)
    hammer.set_defaults(function=prepare_hammer)
    hammer_calibrate = subparsers.add_parser("hammer-calibrate")
    hammer_calibrate.add_argument("--input-root", type=Path, required=True)
    hammer_calibrate.add_argument("--limit", type=int, default=8)
    hammer_calibrate.add_argument("--max-per-group", type=int, default=1)
    hammer_calibrate.add_argument("--minimum-dolp", type=float, default=0.02)
    hammer_calibrate.add_argument("--stride", type=int, default=4)
    hammer_calibrate.add_argument("--refractive-index", type=float, default=1.5)
    hammer_calibrate.set_defaults(function=calibrate_hammer)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.refractive_index <= 1.0:
        parser.error("--refractive-index must be greater than one")
    if getattr(args, "max_geometry_angle", 1.0) <= 0.0 or getattr(args, "max_geometry_angle", 1.0) > 180.0:
        parser.error("--max-geometry-angle must be in (0, 180]")
    if getattr(args, "group_prefix_components", 1) <= 0:
        parser.error("--group-prefix-components must be positive")
    if getattr(args, "max_per_group", None) is not None and args.max_per_group <= 0:
        parser.error("--max-per-group must be positive")
    if not 0.0 <= getattr(args, "minimum_dolp", 0.0) <= 1.0:
        parser.error("--minimum-dolp must be between zero and one")
    if getattr(args, "stride", 1) <= 0:
        parser.error("--stride must be positive")
    if getattr(args, "spatial_stride", 1) <= 0:
        parser.error("--spatial-stride must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
