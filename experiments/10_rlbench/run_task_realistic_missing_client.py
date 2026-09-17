"""Evaluate one RLBench task under structured depth failures on its target object.

This is an additive diagnostic client.  It reuses the trained PointACT policy and
RLBench environment without modifying either implementation.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects.shape import Shape
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.utils import task_file_to_task_class
import tyro

from pointact.robot_envs.rlbench_utils.environments import Mover, RLBenchEnv
from pointact.robot_envs.rlbench_utils.eval_utils import (
    get_rlbench_robot_workspace,
    set_random_seed,
)
from pointact.utils.server_client import PolicyClient
from run_close_fridge_realistic_missing_client import state_vector, structured_corruption


@dataclasses.dataclass
class Args:
    task: str
    target_object: str
    output_dir: str
    host: str = "127.0.0.1"
    port: int = 15507
    severity: float = 0.0
    variation: int = 0
    seed: int = 7
    num_episodes: int = 10
    max_steps: int = 25


def target_handles(root: Shape) -> np.ndarray:
    objects = root.get_objects_in_tree(exclude_base=False)
    return np.asarray([obj.get_handle() for obj in objects], dtype=np.int64)


def corrupt_visible_target(
    points: np.ndarray,
    target: np.ndarray,
    camera_extrinsics: np.ndarray,
    severity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Corrupt visible target pixels; an occluded target is a valid no-op frame."""
    if target.any():
        return structured_corruption(points, target, camera_extrinsics, severity, seed)
    finite = np.isfinite(points).all(axis=-1)
    return points.copy(), np.zeros(target.shape, dtype=np.uint8), {
        "target_pixels": 0,
        "target_corrupted_pixels": 0,
        "target_invalid_pixels": 0,
        "target_background_pixels": 0,
        "target_distorted_pixels": 0,
        "non_target_invalid_pixels": 0,
        "correct_target_fraction": 1.0,
        "target_invalid_fraction": 0.0,
        "target_wrong_depth_fraction": 0.0,
        "valid_xyz_fraction": float(finite.mean()),
        "raw_point_center_shift_m": 0.0,
        "target_fully_occluded": True,
    }


def main(args: Args) -> None:
    if not 0.0 <= args.severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "episode_results.jsonl"
    results_path.write_text("", encoding="utf-8")

    client = PolicyClient(args.host, args.port)
    if not client.ping():
        raise RuntimeError(f"PointACT server is unavailable on {args.host}:{args.port}")

    env = RLBenchEnv(
        apply_rgb=True,
        apply_depth=True,
        apply_pc=True,
        apply_mask=True,
        apply_cameras=("front",),
        headless=True,
        image_size=(256, 256),
        cam_params_to_opencv=True,
        use_metric_depth=True,
    )
    env.obs_config.front_camera.masks_as_one_channel = True
    env.env.launch()
    records: list[dict] = []
    try:
        task = env.env.get_task(task_file_to_task_class(args.task))
        task.set_variation(args.variation)
        target_root = Shape(args.target_object)
        handles = target_handles(target_root)
        workspace = get_rlbench_robot_workspace()

        for episode in range(args.num_episodes):
            episode_seed = args.seed + 10_007 * episode
            set_random_seed(episode_seed)
            instructions, observation = task.reset()
            data = env.get_observation(observation)
            signature = {
                "target_object": args.target_object,
                "target_pose": np.asarray(target_root.get_pose(), dtype=float).tolist(),
                "instruction": instructions[0],
            }
            move = Mover(task, max_tries=10)
            move.reset(data["gripper"])
            client.reset(options={"seed": episode_seed})
            first_prediction = None
            first_stats = None
            success = False
            error = None

            for step in range(args.max_steps):
                points = data["pc"][0]
                instance = np.rint(data["gt_mask"][0]).astype(np.int64)
                target = np.isin(instance, handles)
                corruption_seed = 701 + 1_000_003 * episode + 10_007 * step
                corrupted, labels, stats = corrupt_visible_target(
                    points,
                    target,
                    data["camera_extrinsics"]["front"],
                    args.severity,
                    corruption_seed,
                )
                batch = {
                    "task": [instructions[0]],
                    "repo_id": ["hybridvla_10tasks_train_keysteps"],
                    "observation.state": [state_vector(data["gripper"])],
                    "observation.images.front_image": [data["rgb"][0]],
                    "observation.points.front": [corrupted],
                }
                output_action = client.get_action(
                    batch, options={"pred_rot_type": "euler", "remove_arm": False}
                )
                action = np.asarray(output_action.action[0, 0]).copy()
                if first_prediction is None:
                    first_prediction = action[:3].astype(float).tolist()
                    first_stats = stats
                    np.savez_compressed(
                        output / f"episode_{episode:03d}_initial.npz",
                        rgb=data["rgb"][0],
                        clean_points=points,
                        corrupted_points=corrupted,
                        target_mask=target,
                        corruption_labels=labels,
                        instance_mask=instance,
                    )

                action[0] = np.clip(action[0], *workspace["X_BBOX"])
                action[1] = np.clip(action[1], *workspace["Y_BBOX"])
                action[2] = np.clip(action[2], *workspace["Z_BBOX"])
                try:
                    observation, reward, _terminate, _ = move(action, verbose=False)
                    data = env.get_observation(observation)
                except (IKError, ConfigurationPathError, InvalidActionError) as exc:
                    reward = 0.0
                    error = f"{type(exc).__name__}: {exc}"
                    break
                if reward == 1:
                    success = True
                    break

            record = {
                "task": args.task,
                "variation": args.variation,
                "target_object": args.target_object,
                "episode": episode,
                "episode_seed": episode_seed,
                "severity": args.severity,
                "success": success,
                "policy_steps": step + 1,
                "error": error,
                "initial_scene": signature,
                "first_predicted_position_world": first_prediction,
                "first_corruption": first_stats,
            }
            records.append(record)
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"task={args.task} severity={args.severity:.2f} episode={episode} "
                f"success={success} steps={step + 1}",
                flush=True,
            )
    finally:
        env.env.shutdown()

    summary = {
        "task": args.task,
        "variation": args.variation,
        "target_object": args.target_object,
        "severity": args.severity,
        "num_episodes": len(records),
        "successes": sum(item["success"] for item in records),
        "success_rate": float(np.mean([item["success"] for item in records])),
        "corruption_definition": {
            "target_region": f"visible descendants of {args.target_object} in front instance mask",
            "target_mix": {
                "invalid_holes": 0.55,
                "background_depth": 0.35,
                "range_distortion": 0.10,
            },
            "edge_bias": True,
            "non_target_hole_fraction": 0.10 * args.severity,
            "rgb_instruction_robot_state_unchanged": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    tyro.cli(main)
