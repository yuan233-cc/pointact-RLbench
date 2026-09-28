#!/usr/bin/env python3
"""Replay official LIBERO-Spatial demos and export aligned polar point frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from modalities import (
    analytic_polarization, complete_corrupted_cloud, corrupt_libero_cloud,
    interaction_reconstruction_supervision, make_aligned_cloud,
    polar_physics_report, polar_region_report, realign_corrupted_features,
    unproject_depth,
)


DEFAULT_MATERIALS = Path(__file__).with_name("libero_spatial_materials.json")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_noop(action, previous, threshold=1e-4):
    return np.linalg.norm(action[:-1]) < threshold and (
        previous is None or action[-1] == previous[-1])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Debug limit per episode; omit for complete trajectories")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--state-source", choices=("replay", "recorded"), default="replay",
                        help="execute actions, or restore each recorded MuJoCo state directly")
    parser.add_argument("--voxel-size", type=float, default=0.012)
    parser.add_argument("--target-voxel-size", type=float, default=0.005)
    parser.add_argument("--target-max-points", type=int, default=512)
    parser.add_argument("--workspace-low", type=float, nargs=3, default=(-0.5, -0.5, 0.85))
    parser.add_argument("--workspace-high", type=float, nargs=3, default=(0.5, 0.5, 1.7))
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--robot-drop-fraction", type=float, default=0.13,
                        help="fraction of robot/gripper points removed in structured holes")
    parser.add_argument("--target-affected-fraction", type=float, default=0.55,
                        help="fraction of manipulated-object points affected by missing/wrong depth")
    parser.add_argument("--target-drop-fraction", type=float, default=0.65,
                        help="fraction of affected target points removed rather than depth-shifted")
    parser.add_argument("--keep-noops", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip complete episodes and regenerate only partial episodes")
    parser.add_argument("--omit-redundant-aolp", action="store_true",
                        help="omit dense AoLP; recover it as 0.5*atan2(sin2AoLP, cos2AoLP) mod pi")
    parser.add_argument("--min-free-gib", type=float, default=5.0,
                        help="stop before starting an episode if less free space remains")
    parser.add_argument("--save-dense-debug", action="store_true",
                        help="also save dense world XYZ and full Stokes S0..S3")
    parser.add_argument("--polar-backend", choices=("analytic", "native"), default="analytic")
    parser.add_argument("--native-renderer-repo", type=Path,
                        default=Path(__file__).resolve().parents[3] / "rlbench_custom_render/RLBench")
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS,
                        help="geom-name/glob to native-BSDF JSON overrides")
    parser.add_argument("--spp", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error(f"output already exists: {args.output}")
    if (args.episodes_per_task < 1 or args.resolution < 16 or args.voxel_size <= 0
            or args.target_voxel_size <= 0 or args.target_max_points < 1
            or args.min_free_gib < 0):
        parser.error("episode count, resolution, and voxel size must be positive")
    for name in ("robot_drop_fraction", "target_affected_fraction", "target_drop_fraction"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def save_frame(path, *, rgb, depth, geom_image, world, intrinsics, extrinsics,
               polar, clean, pixels, point_geom_ids, corrupted, filled,
               supervision, action, state,
               save_dense_debug=False, store_aolp=True):
    arrays = dict(
        rgb=rgb, depth_m=depth, geom_id=geom_image,
        camera_intrinsics=intrinsics.astype(np.float32),
        camera_to_world=extrinsics.astype(np.float32),
        DoLP=np.asarray(polar["DoLP"], dtype=np.float32),
        cos2AoLP=np.asarray(polar["cos2AoLP"], dtype=np.float32),
        sin2AoLP=np.asarray(polar["sin2AoLP"], dtype=np.float32),
        polar_valid_mask=np.asarray(polar["valid_mask"], dtype=bool),
        aolp_valid_mask=np.asarray(polar["AoLP_valid_mask"], dtype=bool),
        clean9=clean, clean_pixel_index=pixels, clean_geom_id=point_geom_ids,
        incomplete9=corrupted.cloud,
        incomplete_source_pixel_index=corrupted.source_pixels,
        incomplete_current_pixel_index=corrupted.current_pixels,
        incomplete_corruption_code=corrupted.codes,
        filled9=filled.cloud,
        filled_source_pixel_index=filled.source_pixels,
        filled_current_pixel_index=filled.current_pixels,
        filled_synthetic_mask=filled.synthetic_mask,
        filled_corruption_code=filled.codes,
        interaction_target_points=supervision.target_points,
        interaction_input_mask=supervision.input_mask,
        action=np.asarray(action, dtype=np.float32),
        state=np.asarray(state, dtype=np.float32),
    )
    if store_aolp:
        arrays["AoLP"] = np.asarray(polar["AoLP"], dtype=np.float32)
    if save_dense_debug:
        arrays.update(
            point_cloud_world=world.astype(np.float32),
            S0=np.asarray(polar["S0"], dtype=np.float32),
            S1=np.asarray(polar["S1"], dtype=np.float32),
            S2=np.asarray(polar["S2"], dtype=np.float32),
            S3=np.asarray(polar["S3"], dtype=np.float32),
        )
    np.savez_compressed(path, **arrays)


def main():
    args = parse_args()
    # Imports occur after argument validation so --help does not require LIBERO.
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if any(task_id < 0 or task_id >= suite.n_tasks for task_id in args.task_ids):
        raise ValueError(f"task ids must be in [0, {suite.n_tasks})")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    dataset_records = []
    global_episode = 0
    try:
        for task_id in args.task_ids:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            source_path = args.raw_dir / f"{task.name}_demo.hdf5"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            env = OffScreenRenderEnv(
                bddl_file_name=bddl,
                camera_heights=args.resolution, camera_widths=args.resolution,
                camera_depths=True, camera_segmentations="element",
            )
            env.seed(args.seed + task_id)
            native = None
            if args.polar_backend == "native":
                from mujoco_native_polar import MujocoNativePolarRenderer
                materials = (json.loads(args.materials.read_text(encoding="utf-8"))
                             if args.materials is not None else {})
                native = MujocoNativePolarRenderer(
                    env.sim, args.native_renderer_repo, spp=args.spp,
                    max_depth=args.max_depth, seed=args.seed, device=args.device,
                    material_overrides=materials)
            try:
                with h5py.File(source_path, "r") as source:
                    demos = source["data"]
                    count = min(args.episodes_per_task, len(demos))
                    for episode_index in range(count):
                        global_episode = task_id * args.episodes_per_task + episode_index
                        episode_dir = args.output / f"task_{task_id:02d}" / f"episode_{episode_index:06d}"
                        summary_path = episode_dir / "summary.json"
                        if args.resume and summary_path.is_file():
                            prior = json.loads(summary_path.read_text(encoding="utf-8"))
                            if prior.get("complete", False) and prior.get("success", False):
                                dataset_records.append(prior)
                                print(f"task {task_id} episode {episode_index}: already complete", flush=True)
                                continue
                        if episode_dir.exists():
                            if not args.resume:
                                raise FileExistsError(episode_dir)
                            shutil.rmtree(episode_dir)
                        free_gib = shutil.disk_usage(args.output).free / 1024**3
                        if free_gib < args.min_free_gib:
                            raise RuntimeError(
                                f"Only {free_gib:.2f} GiB free; refusing to start another episode "
                                f"(--min-free-gib={args.min_free_gib})"
                            )
                        demo = demos[f"demo_{episode_index}"]
                        actions = np.asarray(demo["actions"])
                        states = np.asarray(demo["states"])
                        frames_dir = episode_dir / "frames"
                        frames_dir.mkdir(parents=True)
                        raw_obs = env.reset()
                        if native is not None:
                            native.set_sim(env.sim)
                        if args.state_source == "replay":
                            raw_obs = env.set_init_state(states[0])
                            for _ in range(args.settle_steps):
                                raw_obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                        intrinsics = camera_utils.get_camera_intrinsic_matrix(
                            env.sim, args.camera, args.resolution, args.resolution)
                        saved, previous = 0, None
                        frame_stats = []
                        done = False
                        for source_step, action in enumerate(actions):
                            if args.max_steps is not None and source_step >= args.max_steps:
                                break
                            if args.state_source == "recorded":
                                raw_obs = env.set_init_state(states[source_step])
                            skip = not args.keep_noops and is_noop(action, previous)
                            if not skip:
                                rgb = np.ascontiguousarray(raw_obs[f"{args.camera}_image"][::-1])
                                normalized_depth = np.ascontiguousarray(raw_obs[f"{args.camera}_depth"][::-1])
                                depth = camera_utils.get_real_depth_map(env.sim, normalized_depth)[..., 0]
                                geom_image = np.ascontiguousarray(
                                    raw_obs[f"{args.camera}_segmentation_element"][::-1, :, 0],
                                    dtype=np.int32)
                                extrinsics = camera_utils.get_camera_extrinsic_matrix(env.sim, args.camera)
                                geom_names = {gid: (env.sim.model.geom_id2name(gid) or f"geom_{gid}")
                                              for gid in range(env.sim.model.ngeom)}
                                world = unproject_depth(depth, intrinsics, extrinsics)
                                if native is None:
                                    polar = analytic_polarization(
                                        rgb, depth, intrinsics, geom_image, geom_names)
                                else:
                                    polar = native.render(
                                        args.camera, args.resolution, args.resolution,
                                        intrinsics, extrinsics,
                                        seed=(args.seed + global_episode * 100000 + saved) % 2**32)
                                polar["metadata"]["physics_qa"] = polar_physics_report(polar)
                                polar["metadata"]["region_qa"] = polar_region_report(
                                    polar, geom_image, geom_names)
                                clean, pixels, point_geom_ids = make_aligned_cloud(
                                    rgb, depth, world, polar, geom_image,
                                    workspace_low=np.asarray(args.workspace_low),
                                    workspace_high=np.asarray(args.workspace_high),
                                    voxel_size=args.voxel_size)
                                corrupted = corrupt_libero_cloud(
                                    clean, pixels, point_geom_ids, geom_names, extrinsics[:3, 3],
                                    episode_index=global_episode, seed=args.seed,
                                    robot_drop_fraction=args.robot_drop_fraction,
                                    target_affected_fraction=args.target_affected_fraction,
                                    target_drop_fraction=args.target_drop_fraction)
                                corrupted = realign_corrupted_features(
                                    corrupted, rgb, polar, intrinsics, extrinsics)
                                filled = complete_corrupted_cloud(
                                    corrupted, pixels, rgb, polar, intrinsics, extrinsics)
                                supervision = interaction_reconstruction_supervision(
                                    filled.cloud, world, depth, geom_image, geom_names,
                                    intrinsics, extrinsics,
                                    seed=args.seed+global_episode*100000+saved,
                                    max_points=args.target_max_points,
                                    voxel_size=args.target_voxel_size,
                                    workspace_low=np.asarray(args.workspace_low),
                                    workspace_high=np.asarray(args.workspace_high))
                                state = np.concatenate((raw_obs["robot0_eef_pos"],
                                                        raw_obs["robot0_eef_quat"],
                                                        raw_obs["robot0_gripper_qpos"]))
                                save_frame(
                                    frames_dir / f"{saved:06d}.npz", rgb=rgb, depth=depth,
                                    geom_image=geom_image, world=world, intrinsics=intrinsics,
                                    extrinsics=extrinsics, polar=polar, clean=clean, pixels=pixels,
                                    point_geom_ids=point_geom_ids, corrupted=corrupted,
                                    filled=filled, supervision=supervision,
                                    action=action, state=state,
                                    save_dense_debug=args.save_dense_debug,
                                    store_aolp=not args.omit_redundant_aolp)
                                frame_stats.append({
                                    "frame": saved, "source_step": source_step,
                                    "polar": jsonable(polar["metadata"]),
                                    "corruption": corrupted.stats,
                                    "completion": filled.stats,
                                    "interaction_supervision": supervision.stats,
                                })
                                saved += 1
                                previous = action
                            if args.state_source == "replay":
                                raw_obs, reward, done, info = env.step(action.tolist())
                        summary = {
                            "complete": args.max_steps is None,
                            "debug_truncated": args.max_steps is not None,
                            "suite": "libero_spatial", "task_id": task_id,
                            "task": task.name, "language": task.language,
                            "source": str(source_path), "source_episode": episode_index,
                            "global_episode": global_episode, "source_steps": len(actions),
                            "saved_frames": saved,
                            "success": bool(done) if args.state_source == "replay" else None,
                            "state_source": args.state_source,
                            "polar_backend": args.polar_backend,
                            "material_overrides": (str(args.materials.resolve())
                                                   if args.polar_backend == "native" else None),
                            "save_dense_debug": args.save_dense_debug,
                            "omitted_redundant_fields": (
                                ["AoLP"] if args.omit_redundant_aolp else []
                            ),
                            "aolp_reconstruction": (
                                "mod(0.5 * atan2(sin2AoLP, cos2AoLP), pi)"
                                if args.omit_redundant_aolp else None
                            ),
                            "point_channels": ["x", "y", "z", "r", "g", "b",
                                               "DoLP", "cos2AoLP", "sin2AoLP"],
                            "corruption_config": {
                                "robot_drop_fraction": args.robot_drop_fraction,
                                "target_affected_fraction": args.target_affected_fraction,
                                "target_drop_fraction": args.target_drop_fraction,
                            },
                            "geom_names": {str(key): value for key, value in geom_names.items()},
                            "corruption_codes": {"0": "unchanged", "1": "support distortion",
                                                 "2": "target wrong depth", "4": "floating point",
                                                 "5": "robot hole (removed rows are counted only)",
                                                 "7": "morphology-filled synthetic point"},
                            "frames": frame_stats,
                        }
                        summary_path.write_text(
                            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                        dataset_records.append(summary)
                        print(f"task {task_id} episode {episode_index}: {saved} frames", flush=True)
                        if (args.state_source == "replay" and args.max_steps is None
                                and not summary["success"]):
                            raise RuntimeError(
                                f"Replay did not reach success for task {task_id}, "
                                f"episode {episode_index}; refusing an incomplete training set"
                            )
            finally:
                if native is not None:
                    native.close()
                env.close()
    except Exception:
        (args.output / "FAILED").write_text("generation stopped before completion\n")
        raise
    manifest = {
        "complete": args.max_steps is None,
        "suite": "libero_spatial", "episodes": len(dataset_records),
        "polar_backend": args.polar_backend, "resolution": args.resolution,
        "state_source": args.state_source,
        "material_overrides": (str(args.materials.resolve())
                               if args.polar_backend == "native" else None),
        "save_dense_debug": args.save_dense_debug,
        "omitted_redundant_fields": (["AoLP"] if args.omit_redundant_aolp else []),
        "aolp_reconstruction": (
            "mod(0.5 * atan2(sin2AoLP, cos2AoLP), pi)"
            if args.omit_redundant_aolp else None
        ),
        "voxel_size_m": args.voxel_size, "seed": args.seed,
        "corruption_config": {
            "robot_drop_fraction": args.robot_drop_fraction,
            "target_affected_fraction": args.target_affected_fraction,
            "target_drop_fraction": args.target_drop_fraction,
        },
        "target_voxel_size_m": args.target_voxel_size,
        "target_max_points": args.target_max_points,
        "official_libero_modified": False,
        "records": [{k: item[k] for k in ("task_id", "task", "source_episode", "saved_frames", "success")}
                    for item in dataset_records],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failed_marker = args.output / "FAILED"
    if failed_marker.exists():
        failed_marker.unlink()


if __name__ == "__main__":
    main()
