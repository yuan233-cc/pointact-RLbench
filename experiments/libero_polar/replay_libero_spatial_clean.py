#!/usr/bin/env python3
"""Export the full clean LIBERO-Spatial GeoVLA inputs without polarization."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import h5py
import numpy as np


def is_noop(action: np.ndarray, previous: np.ndarray | None, threshold: float = 1e-4) -> bool:
    return np.linalg.norm(action[:-1]) < threshold and (
        previous is None or action[-1] == previous[-1]
    )


def state8(raw_state: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_state, dtype=np.float32)
    finger_width = float(raw[7] - raw[8])
    gripper_closed = 1.0 if finger_width < 0.04 else -1.0
    return np.ascontiguousarray(np.concatenate((raw[:7], [gripper_closed])), dtype=np.float32)


def openvla_action(raw_action: np.ndarray) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32).copy()
    if action.shape != (7,) or not np.isin(action[-1], (-1.0, 1.0)):
        raise ValueError(f"Unexpected LIBERO action: shape={action.shape}, gripper={action[-1]}")
    action[-1] *= -1.0
    return action


def unproject(depth: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float32)
    camera = np.stack(
        (
            (u - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (v - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    )
    world = camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    world = np.ascontiguousarray(world[:, ::-1], dtype=np.float32)
    if not np.isfinite(world).all():
        raise ValueError("base_pc contains non-finite values")
    return world


def save_frame(
    path: Path,
    image: np.ndarray,
    base_pc: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    source_step: int,
) -> None:
    np.savez_compressed(
        path,
        image=np.ascontiguousarray(image, dtype=np.uint8),
        base_pc=np.ascontiguousarray(base_pc, dtype=np.float32),
        state=np.ascontiguousarray(state, dtype=np.float32),
        action=np.ascontiguousarray(action, dtype=np.float32),
        source_step=np.int32(source_step),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--writer-workers", type=int, default=4)
    parser.add_argument("--max-pending-writes", type=int, default=16)
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--worker-fragment",
        action="store_true",
        help="Write a task-specific manifest/FAILED marker for safe multi-process generation.",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error(f"output already exists: {args.output}")
    if min(args.episodes_per_task, args.resolution, args.writer_workers, args.max_pending_writes) < 1:
        parser.error("episode count, resolution, and writer settings must be positive")
    return args


def main() -> None:
    args = parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if any(task_id < 0 or task_id >= suite.n_tasks for task_id in args.task_ids):
        raise ValueError(f"task ids must be in [0, {suite.n_tasks})")

    args.output.mkdir(parents=True, exist_ok=args.resume)
    fragment_suffix = "_".join(f"{task_id:02d}" for task_id in args.task_ids)
    failed_path = (
        args.output / f"FAILED_{fragment_suffix}"
        if args.worker_fragment
        else args.output / "FAILED"
    )
    records: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=args.writer_workers) as writers:
            for task_id in args.task_ids:
                task = suite.get_task(task_id)
                bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                source_path = args.raw_dir / f"{task.name}_demo.hdf5"
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl,
                    camera_heights=args.resolution,
                    camera_widths=args.resolution,
                    camera_depths=True,
                )
                env.seed(args.seed + task_id)
                try:
                    with h5py.File(source_path, "r") as source:
                        demos = source["data"]
                        count = min(args.episodes_per_task, len(demos))
                        for episode_index in range(count):
                            episode_dir = (
                                args.output / f"task_{task_id:02d}" / f"episode_{episode_index:06d}"
                            )
                            summary_path = episode_dir / "summary.json"
                            if args.resume and summary_path.is_file():
                                prior = json.loads(summary_path.read_text(encoding="utf-8"))
                                if (
                                    prior.get("complete", False)
                                    and prior.get("success", False)
                                    and prior.get("observation_source") == "recorded_sim_state"
                                    and prior.get("unprojection")
                                    == "robosuite_integer_pixel_coordinates"
                                    and prior.get("state_source")
                                    == "official_hdf5_robot_states"
                                    and "camera_intrinsics" in prior
                                    and "camera_to_world" in prior
                                ):
                                    records.append(prior)
                                    print(f"task {task_id} episode {episode_index}: already complete", flush=True)
                                    continue
                            if episode_dir.exists():
                                if not args.resume:
                                    raise FileExistsError(episode_dir)
                                shutil.rmtree(episode_dir)
                            free_gib = shutil.disk_usage(args.output).free / 1024**3
                            if free_gib < args.min_free_gib:
                                raise RuntimeError(
                                    f"Only {free_gib:.2f} GiB free; refusing another episode "
                                    f"(--min-free-gib={args.min_free_gib})"
                                )

                            demo = demos[f"demo_{episode_index}"]
                            actions = np.asarray(demo["actions"])
                            states = np.asarray(demo["states"])
                            dones = np.asarray(demo["dones"])
                            rewards = np.asarray(demo["rewards"])
                            robot_states = np.asarray(demo["robot_states"])
                            if not (
                                len(actions)
                                == len(states)
                                == len(dones)
                                == len(rewards)
                                == len(robot_states)
                            ):
                                raise ValueError(
                                    f"Source length mismatch for task {task_id}, "
                                    f"episode {episode_index}"
                                )
                            source_success = bool(dones[-1] and rewards[-1] > 0)
                            if not source_success:
                                raise RuntimeError(
                                    f"Official source demo is not terminal/successful: "
                                    f"task {task_id}, episode {episode_index}"
                                )
                            frames_dir = episode_dir / "frames"
                            frames_dir.mkdir(parents=True)

                            env.reset()
                            observation = env.set_init_state(states[0])
                            intrinsics = camera_utils.get_camera_intrinsic_matrix(
                                env.sim, args.camera, args.resolution, args.resolution
                            ).astype(np.float32)
                            camera_to_world = camera_utils.get_camera_extrinsic_matrix(
                                env.sim, args.camera
                            ).astype(np.float32)

                            saved = 0
                            previous = None
                            pending = deque()
                            for source_step, raw_action in enumerate(actions):
                                # Render the exact state stored in the official demonstration.
                                # Re-executing actions accumulates version-dependent controller
                                # drift and can turn a successful source demo into a failed replay.
                                observation = env.set_init_state(states[source_step])
                                skip = is_noop(raw_action, previous)
                                if not skip:
                                    # LIBERO/OpenVLA uses a 180-degree image convention. MuJoCo
                                    # depth is vertically corrected before unprojection, then the
                                    # organized XYZ grid receives the matching horizontal flip.
                                    image = np.ascontiguousarray(
                                        observation[f"{args.camera}_image"][::-1, ::-1]
                                    )
                                    normalized_depth = np.ascontiguousarray(
                                        observation[f"{args.camera}_depth"][::-1]
                                    )
                                    depth = camera_utils.get_real_depth_map(
                                        env.sim, normalized_depth
                                    )[..., 0].astype(np.float32)
                                    base_pc = unproject(depth, intrinsics, camera_to_world)
                                    recorded_robot_state = robot_states[source_step]
                                    if recorded_robot_state.shape != (9,):
                                        raise ValueError(
                                            f"Expected official robot_state (9,), got "
                                            f"{recorded_robot_state.shape}"
                                        )
                                    # Official order is gripper_qpos[2], eef_xyz[3], quat_xyzw[4].
                                    raw_state = np.concatenate(
                                        (recorded_robot_state[2:], recorded_robot_state[:2])
                                    )
                                    pending.append(
                                        writers.submit(
                                            save_frame,
                                            frames_dir / f"{saved:06d}.npz",
                                            image,
                                            base_pc,
                                            state8(raw_state),
                                            openvla_action(raw_action),
                                            source_step,
                                        )
                                    )
                                    if len(pending) >= args.max_pending_writes:
                                        pending.popleft().result()
                                    saved += 1
                                    previous = raw_action
                            while pending:
                                pending.popleft().result()
                            summary = {
                                "complete": True,
                                "suite": "libero_spatial",
                                "task_id": task_id,
                                "task": task.name,
                                "language": task.language,
                                "source": str(source_path),
                                "source_episode": episode_index,
                                "source_steps": len(actions),
                                "saved_frames": saved,
                                "success": source_success,
                                "observation_source": "recorded_sim_state",
                                "success_source": "official_hdf5_terminal_reward",
                                "state_source": "official_hdf5_robot_states",
                                "format": "geovla_clean_npz_v1",
                                "image_convention": "libero_openvla_rot180",
                                "point_cloud_frame": "world",
                                "unprojection": "robosuite_integer_pixel_coordinates",
                                "camera_intrinsics": intrinsics.tolist(),
                                "camera_to_world": camera_to_world.tolist(),
                                "action_convention": "eef_delta_gripper_open_positive",
                                "state_encoding": "eef_xyz_quat_binary_gripper_closed_positive",
                            }
                            summary_path.write_text(
                                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                            )
                            records.append(summary)
                            print(f"task {task_id} episode {episode_index}: {saved} frames", flush=True)
                finally:
                    env.close()
    except Exception:
        failed_path.write_text("generation stopped before completion\n")
        raise

    expected = sum(
        min(args.episodes_per_task, 50) for _ in args.task_ids
    )
    manifest = {
        "complete": len(records) == expected and all(record["success"] for record in records),
        "suite": "libero_spatial",
        "episodes": len(records),
        "expected_episodes": expected,
        "resolution": args.resolution,
        "format": "geovla_clean_npz_v1",
        "contains_polar": False,
        "contains_corrupted_or_filled_clouds": False,
        "official_libero_modified": False,
        "records": [
            {key: record[key] for key in ("task_id", "task", "source_episode", "saved_frames", "success")}
            for record in records
        ],
    }
    manifest_path = (
        args.output / f"partial_manifest_{fragment_suffix}.json"
        if args.worker_fragment
        else args.output / "manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if failed_path.exists():
        failed_path.unlink()


if __name__ == "__main__":
    main()
