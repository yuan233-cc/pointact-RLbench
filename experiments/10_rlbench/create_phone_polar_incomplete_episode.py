"""Build one aligned phone episode with incomplete XYZRGB and three polar channels.

The already-collected successful RLBench episode supplies the RGB/state/action
trajectory and high-quality polar keyframes. Geometry is corrupted first;
each retained point then receives RGB and polar from its current projected
image pixel. Thus image modalities remain correct while 3D positions can be
wrong. Points projecting outside valid image data are omitted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np

from create_rlbench_10task_realistic_failure_dataset import apply_corruption
from create_stack_wine_10episode_failure_dataset import (
    CAMERA_CENTER, CAMERA_EXTRINSICS, CAMERA_FOCAL, project_world,
)
from polar_depth_fill import add_polar_depth_filled_points, corruption_hole_pixels


msgpack_numpy.patch()
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_LOW = np.array([-0.5, -1.0, 0.7505], dtype=np.float32)
WORKSPACE_HIGH = np.array([1.5, 1.0, 2.0], dtype=np.float32)
DEFAULT_SOURCE = REPO_ROOT / "robot_data/rlbench/lerobot_point_lmdb/hybridvla_phone_on_base_1episode_polar_aligned_seed24"
DEFAULT_FRAMES = REPO_ROOT.parent / "rlbench_custom_render/RLBench/output/material_profile_v3_aligned/phone_on_base_episode1/frames_spp512"
DEFAULT_OUTPUT = REPO_ROOT / "robot_data/rlbench/lerobot_point_lmdb/phone_on_base_1episode_polar_incomplete9_seed24"


def source_points(frame_path: Path, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    with np.load(frame_path) as frame:
        points = np.asarray(frame["points_xyzrgbpolar"], dtype=np.float32)
        pixels = np.asarray(frame["point_pixel_index"], dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 10 or len(points) != len(pixels):
            raise ValueError(f"{frame_path}: invalid source point/pixel alignment")
        image_shape = frame["DoLP"].shape
        if np.any(pixels < 0) or np.any(pixels >= np.prod(image_shape)):
            raise ValueError(f"{frame_path}: pixel index outside polar image")
        angle_valid = np.asarray(frame["AoLP_valid_mask"], dtype=bool).reshape(-1)[pixels]
        angle = np.asarray(frame["AoLP"], dtype=np.float32).reshape(-1)[pixels]
        dolp = np.asarray(frame["DoLP"], dtype=np.float32).reshape(-1)[pixels]
        if not np.allclose(points[:, 8], dolp, atol=1e-5):
            raise ValueError(f"{frame_path}: dense and point DoLP disagree")
        # Angle is undefined for a few low-signal pixels. The (0, 0) pair
        # cannot be a valid unit-angle encoding, so it serves as an implicit
        # invalid marker without adding a tenth channel.
        polar = np.column_stack((dolp, np.where(angle_valid, np.cos(2 * angle), 0.0),
                                 np.where(angle_valid, np.sin(2 * angle), 0.0)))
        features = np.concatenate((points[:, :6], polar), axis=1).astype(np.float32)

    inside = np.all((features[:, :3] >= WORKSPACE_LOW) & (features[:, :3] <= WORKSPACE_HIGH), axis=1)
    features, pixels = features[inside], pixels[inside]
    if not len(features) or not np.isfinite(features).all():
        raise ValueError(f"{frame_path}: no finite workspace points")

    # Keep an actual camera sample per voxel so its polar value still belongs
    # to the retained 3D point. Averaging XYZ independently would break this.
    voxel = np.floor(features[:, :3] / voxel_size).astype(np.int32)
    _, representative = np.unique(voxel, axis=0, return_index=True)
    representative.sort()
    return np.ascontiguousarray(features[representative]), pixels[representative]


def sample_projected_modalities(
    frame_path: Path, xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the true RGB/polar maps at each corrupted point's current pixel."""
    with np.load(frame_path) as frame:
        height, width = frame["DoLP"].shape
        uv, depth = project_world(xyz)
        finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth)
        pixel_xy = np.floor(np.where(finite[:, None], uv, -1.0)).astype(np.int64)
        in_frame = (
            finite & (depth > 0)
            & (pixel_xy[:, 0] >= 0) & (pixel_xy[:, 0] < width)
            & (pixel_xy[:, 1] >= 0) & (pixel_xy[:, 1] < height)
        )
        pixel_index = np.full(len(xyz), -1, dtype=np.int32)
        pixel_index[in_frame] = (
            pixel_xy[in_frame, 1] * width + pixel_xy[in_frame, 0]
        ).astype(np.int32)
        valid = in_frame.copy()
        selected = pixel_index[in_frame]
        valid[in_frame] &= (
            frame["valid_mask"].reshape(-1)[selected]
            & frame["AoLP_valid_mask"].reshape(-1)[selected]
        )
        rgb = np.zeros((len(xyz), 3), dtype=np.float32)
        polar = np.zeros((len(xyz), 3), dtype=np.float32)
        selected = pixel_index[valid]
        if len(selected):
            rgb[valid] = frame["rgb"].reshape(-1, 3)[selected] / 255.0
            angle = frame["AoLP"].reshape(-1)[selected]
            polar[valid] = np.column_stack((
                frame["DoLP"].reshape(-1)[selected],
                np.cos(2 * angle), np.sin(2 * angle),
            ))
        valid &= np.isfinite(rgb).all(axis=1) & np.isfinite(polar).all(axis=1)
    return rgb, polar, pixel_index, valid


def link_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link)


def write_clouds(path: Path, clouds: list[np.ndarray]) -> None:
    environment = lmdb.open(str(path), map_size=1 << 29)
    try:
        with environment.begin(write=True) as transaction:
            for index, cloud in enumerate(clouds):
                transaction.put(f"0-{index}".encode("ascii"), msgpack.packb(cloud))
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--voxel-size", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--fill-depth-holes", action="store_true",
                        help="also export a HouseCat-style depth-filled polar point cloud")
    args = parser.parse_args()
    if args.voxel_size <= 0:
        parser.error("--voxel-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {args.output}")

    frames = sorted(args.frames.glob("[0-9][0-9][0-9][0-9][0-9][0-9].npz"))
    info = json.loads((args.source / "meta/info.json").read_text())
    if len(frames) != info["total_frames"] or info["total_episodes"] != 1:
        raise ValueError("Rendered frames do not match the one-episode LeRobot source")

    clean_clouds: list[np.ndarray] = []
    incomplete_clouds: list[np.ndarray] = []
    filled_clouds: list[np.ndarray] = []
    filled_pixel_indices: list[np.ndarray] = []
    filled_source_indices: list[np.ndarray] = []
    depth_filled_masks: list[np.ndarray] = []
    records: list[dict] = []
    pixel_indices: list[np.ndarray] = []
    source_pixel_indices: list[np.ndarray] = []
    for index, frame_path in enumerate(frames):
        clean, pixels = source_points(frame_path, args.voxel_size)
        # The seventh column carries the original row index through the v2
        # corruption function. Its masks and geometry use only XYZRGB.
        indexed = np.column_stack((clean[:, :6], np.arange(len(clean), dtype=np.float32)))
        result = apply_corruption(indexed, task_index=5, source_episode=0, base_seed=args.seed)
        source_rows = result.cloud[:, 6].astype(np.int64)
        if np.any(source_rows < 0) or np.any(source_rows >= len(clean)):
            raise ValueError(f"{frame_path}: lost source-point correspondence")
        rgb, polar, projected_pixels, valid = sample_projected_modalities(
            frame_path, result.cloud[:, :3]
        )
        incomplete = np.ascontiguousarray(
            np.column_stack((result.cloud[valid, :3], rgb[valid], polar[valid])),
            dtype=np.float32,
        )
        if len(incomplete) == len(clean) or not np.isfinite(incomplete).all():
            raise ValueError(f"{frame_path}: incomplete cloud is invalid")
        if args.fill_depth_holes:
            retained_source_pixels = pixels[source_rows[valid]].astype(np.int32)
            hole_pixels = corruption_hole_pixels(pixels, retained_source_pixels)
            with np.load(frame_path) as frame:
                filled, filled_pixels, filled_mask = add_polar_depth_filled_points(
                    incomplete, projected_pixels[valid], frame,
                    CAMERA_EXTRINSICS, CAMERA_FOCAL, CAMERA_CENTER,
                    hole_pixel_indices=hole_pixels,
                )
            filled_clouds.append(filled)
            filled_pixel_indices.append(filled_pixels)
            filled_source_indices.append(np.concatenate((
                retained_source_pixels,
                np.full(int(filled_mask.sum()), -1, dtype=np.int32),
            )))
            depth_filled_masks.append(filled_mask)
        clean_clouds.append(clean)
        incomplete_clouds.append(incomplete)
        pixel_indices.append(projected_pixels[valid])
        source_pixel_indices.append(pixels[source_rows[valid]].astype(np.int32))
        records.append({"frame_index": index, "source_frame": frame_path.name,
                        "stats": result.stats,
                        "invalid_projected_pixel_removed": int((~valid).sum()),
                        "final_output_points": len(incomplete),
                        "changed_projected_pixel": int(np.count_nonzero(
                            projected_pixels[valid] != pixels[source_rows[valid]])),
                        "modalities_sampled_at": "current_projected_pixel",
                        **({"depth_fill_target_pixels": len(hole_pixels),
                            "depth_filled_points": int(depth_filled_masks[-1].sum()),
                            "filled_output_points": len(filled_clouds[-1])}
                           if args.fill_depth_holes else {})})

    staging = args.output.with_name(args.output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    try:
        staging.mkdir(parents=True)
        for name in ("data", "meta", "videos", "points_frontview_polar"):
            if (args.source / name).exists():
                link_tree(args.source / name, staging / name)
        write_clouds(staging / "points_frontview_polar_clean9", clean_clouds)
        write_clouds(staging / "points_frontview_polar_incomplete9", incomplete_clouds)
        if args.fill_depth_holes:
            write_clouds(staging / "points_frontview_polar_filled9", filled_clouds)
        indices_dir = staging / "point_pixel_indices"
        indices_dir.mkdir()
        source_indices_dir = staging / "point_source_pixel_indices"
        source_indices_dir.mkdir()
        if args.fill_depth_holes:
            filled_indices_dir = staging / "point_pixel_indices_filled"
            filled_indices_dir.mkdir()
            filled_source_dir = staging / "point_source_pixel_indices_filled"
            filled_source_dir.mkdir()
            fill_mask_dir = staging / "point_depth_filled_mask"
            fill_mask_dir.mkdir()
        for index, pixels in enumerate(pixel_indices):
            np.save(indices_dir / f"{index:06d}.npy", pixels)
            np.save(source_indices_dir / f"{index:06d}.npy", source_pixel_indices[index])
            if args.fill_depth_holes:
                np.save(filled_indices_dir / f"{index:06d}.npy", filled_pixel_indices[index])
                np.save(filled_source_dir / f"{index:06d}.npy", filled_source_indices[index])
                np.save(fill_mask_dir / f"{index:06d}.npy", depth_filled_masks[index])
        metadata = {
            "source_dataset": str(args.source.resolve()),
            "source_render_frames": str(args.frames.resolve()),
            "point_cloud_dirname": "points_frontview_polar_incomplete9",
            "clean_point_cloud_dirname": "points_frontview_polar_clean9",
            "point_feature_mode": "xyzrgb_polar",
            "features": ["x", "y", "z", "r", "g", "b", "DoLP", "cos2AoLP", "sin2AoLP"],
            "task": "phone_on_base", "episode_count": 1, "frame_count": len(frames),
            "voxel_size_m": args.voxel_size, "corruption_seed": args.seed,
            "polar_semantics": "RGB and polar are sampled at the corrupted point's current projected pixel; invalid projected pixels are removed",
            "filled_point_cloud_dirname": "points_frontview_polar_filled9" if args.fill_depth_holes else None,
            "depth_fill": ({"method": "multiscale_morphology_on_incomplete_projected_depth",
                            "point_selection": "lost_pre_corruption_voxel_pixels_with_estimated_depth",
                            "unavailable_polar_components": "zero",
                            "mask_dirname": "point_depth_filled_mask",
                            "source_pixel_index_for_filled_points": -1}
                           if args.fill_depth_holes else None),
        }
        (staging / "meta/polar_incomplete_features.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (staging / "frame_corruption_stats.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records))
        if args.fill_depth_holes:
            (staging / "README.md").write_text(
                "# Phone polar depth-filled point cloud\n\n"
                "Use `points_frontview_polar_filled9` for training. Each row is "
                "[XYZ, RGB01, DoLP, cos(2 AoLP), sin(2 AoLP)]. The original "
                "incomplete cloud remains in `points_frontview_polar_incomplete9`. "
                "Missing depths are estimated from the incomplete cloud by "
                "multiscale morphology, then RGB and polar are sampled at each "
                "filled pixel. Candidate pixels are limited to source voxel "
                "samples lost to corruption, not every empty image pixel. "
                "This offline target mask uses the clean source cloud and is "
                "not available at live inference. Only "
                "unavailable polar components are zero. The "
                "renderer's true depth is never used to fill "
                "holes. `point_depth_filled_mask` marks new rows; the model does "
                "not read this sidecar. `point_source_pixel_indices_filled` is -1 "
                "for generated rows. Use the filled-cloud normalization JSON "
                "with `experiments/10_rlbench/data_configs/"
                "data-phone-polar-filled9-one-episode.yaml`.\n",
                encoding="utf-8",
            )
        if args.fill_depth_holes:
            norm_path = staging / "robot_state_action_stats/euler_points_frontview_filled_clf.json"
            norm_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                sys.executable, str(REPO_ROOT / "data_prep/prepare_robot_state_action_stats.py"),
                "--dataset_dirs", str(staging), "--output_file", str(norm_path),
                "--point_cloud_dir", "points_frontview_polar_filled9",
                "--state_xyz_slice", "0", "3", "--action_xyz_slice", "0", "3",
                "--state_rotation_slice", "3", "7", "--action_rotation_slice", "3", "7",
                "--rotation_type", "quat", "--target_rotation_type", "euler",
                "--replace_zero_std",
            ], cwd=REPO_ROOT, check=True)
        staging.rename(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output": str(args.output), "frames": len(frames),
                      "clean_points": sum(map(len, clean_clouds)),
                      "incomplete_points": sum(map(len, incomplete_clouds)),
                      **({"filled_points": sum(map(len, filled_clouds))}
                         if args.fill_depth_holes else {})}, indent=2))


if __name__ == "__main__":
    main()
