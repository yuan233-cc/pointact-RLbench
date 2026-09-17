"""Evaluate close_fridge with structured, transparent-object-like depth failures.

The checkpoint-selected front RGB image remains clean.  Corruption is applied
to the organized front-camera XYZ image before the existing PointACT server:
connected invalid holes, edge-biased failures, background-depth leakage, and
smaller non-target holes.  No model or RLBench source code is changed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.utils import task_file_to_task_class
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
import tyro

from pointact.robot_envs.rlbench_utils.environments import Mover, RLBenchEnv
from pointact.robot_envs.rlbench_utils.eval_utils import (
    get_rlbench_robot_workspace,
    set_random_seed,
)
from pointact.utils.rotation import convert_rotation
from pointact.utils.server_client import PolicyClient


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 15505
    pretrained_path: str = ""
    output_dir: str = ""
    severity: float = 0.0
    seed: int = 7
    num_episodes: int = 10
    max_steps: int = 25


def choose_exact(mask: np.ndarray, score: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    candidates = np.flatnonzero(mask)
    count = min(max(int(count), 0), len(candidates))
    if count:
        selected = candidates[np.argpartition(score.ravel()[candidates], count - 1)[:count]]
        result.ravel()[selected] = True
    return result


def structured_corruption(
    points: np.ndarray,
    target: np.ndarray,
    camera_extrinsics: np.ndarray,
    severity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return corrupted XYZ, a categorical corruption map, and measurements.

    map labels: 0 clean, 1 target invalid, 2 target background leakage,
    3 target range distortion, 4 non-target invalid hole.
    """
    corrupted = points.copy()
    labels = np.zeros(target.shape, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=-1)
    target = target & finite
    if not target.any():
        raise RuntimeError("No visible fridge pixels were found in the front instance mask")
    if severity <= 0:
        center = points[finite].mean(axis=0)
        return corrupted, labels, {
            "target_pixels": int(target.sum()),
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
        }

    rng = np.random.default_rng(seed)
    height, width = target.shape
    smooth = gaussian_filter(rng.standard_normal((height, width)), sigma=7.0, mode="reflect")
    edge = target & ~binary_erosion(target, iterations=3, border_value=0)
    # Lower score is corrupted.  The edge bias reproduces boundary dropout.
    score = smooth.copy()
    score[edge] -= 1.25 * max(float(smooth.std()), 1e-6)
    total_count = int(round(severity * target.sum()))
    selected = choose_exact(target, score, total_count)

    category_score = gaussian_filter(
        rng.standard_normal((height, width)), sigma=5.0, mode="reflect"
    )
    selected_indices = np.flatnonzero(selected)
    category_order = selected_indices[np.argsort(category_score.ravel()[selected_indices])]
    invalid_count = int(round(0.55 * len(category_order)))
    background_count = int(round(0.35 * len(category_order)))
    invalid = np.zeros_like(target)
    background = np.zeros_like(target)
    distorted = np.zeros_like(target)
    invalid.ravel()[category_order[:invalid_count]] = True
    background.ravel()[category_order[invalid_count:invalid_count + background_count]] = True
    distorted.ravel()[category_order[invalid_count + background_count:]] = True

    # Estimate the surface behind the target from the nearest non-target pixel,
    # then place the erroneous point at that range along the original camera ray.
    nearest = distance_transform_edt(target, return_distances=False, return_indices=True)
    camera_origin = np.asarray(camera_extrinsics[:3, 3], dtype=np.float32)
    rays = points - camera_origin
    ranges = np.linalg.norm(rays, axis=-1)
    unit_rays = rays / np.maximum(ranges[..., None], 1e-8)
    neighbor_points = points[nearest[0], nearest[1]]
    neighbor_ranges = np.linalg.norm(neighbor_points - camera_origin, axis=-1)
    replacement_ranges = np.maximum(neighbor_ranges, ranges + 0.03)
    replacement_ranges += rng.normal(0.0, 0.005, target.shape)
    corrupted[background] = (
        camera_origin + unit_rays[background] * replacement_ranges[background, None]
    )

    # Refraction-like range errors stay on the viewing ray.
    range_delta = rng.normal(0.0, 0.04, target.shape)
    small = np.abs(range_delta) < 0.01
    range_delta[small] += np.where(range_delta[small] >= 0, 0.015, -0.015)
    distorted_ranges = np.maximum(ranges + range_delta, 0.05)
    corrupted[distorted] = (
        camera_origin + unit_rays[distorted] * distorted_ranges[distorted, None]
    )
    corrupted[invalid] = np.nan

    # Small connected holes elsewhere model ordinary RGB-D dropouts.
    non_target = finite & ~target
    scene_score = gaussian_filter(
        rng.standard_normal((height, width)), sigma=4.0, mode="reflect"
    )
    scene_count = int(round(0.10 * severity * non_target.sum()))
    scene_holes = choose_exact(non_target, scene_score, scene_count)
    corrupted[scene_holes] = np.nan

    labels[invalid] = 1
    labels[background] = 2
    labels[distorted] = 3
    labels[scene_holes] = 4
    valid_after = np.isfinite(corrupted).all(axis=-1)
    clean_center = points[finite].mean(axis=0)
    corrupt_center = corrupted[valid_after].mean(axis=0)
    stats = {
        "target_pixels": int(target.sum()),
        "target_edge_pixels": int(edge.sum()),
        "target_corrupted_pixels": int(selected.sum()),
        "target_invalid_pixels": int(invalid.sum()),
        "target_background_pixels": int(background.sum()),
        "target_distorted_pixels": int(distorted.sum()),
        "non_target_invalid_pixels": int(scene_holes.sum()),
        "correct_target_fraction": float(1.0 - selected.sum() / target.sum()),
        "target_invalid_fraction": float(invalid.sum() / target.sum()),
        "target_wrong_depth_fraction": float((background | distorted).sum() / target.sum()),
        "valid_xyz_fraction": float(valid_after.mean()),
        "raw_point_center_shift_m": float(np.linalg.norm(corrupt_center - clean_center)),
    }
    return corrupted, labels, stats


def state_vector(gripper: np.ndarray) -> np.ndarray:
    rotation = convert_rotation(
        gripper[3:7], "quat", "euler",
        quat_order_src="xyzw", euler_order_dst="xyz",
    )
    return np.concatenate([gripper[:3], rotation, gripper[7:]])


def scene_signature(fridge_root: Shape, top_joint: Joint) -> dict:
    return {
        "fridge_root_pose": np.asarray(fridge_root.get_pose(), dtype=float).tolist(),
        "top_joint_position": float(top_joint.get_joint_position()),
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
        apply_rgb=True, apply_depth=True, apply_pc=True, apply_mask=True,
        apply_cameras=("front",), headless=True, image_size=(256, 256),
        cam_params_to_opencv=True, use_metric_depth=True,
    )
    env.obs_config.front_camera.masks_as_one_channel = True
    env.env.launch()
    records = []
    try:
        task = env.env.get_task(task_file_to_task_class("close_fridge"))
        task.set_variation(0)
        fridge_root = Shape("fridge_root")
        top_joint = Joint("top_joint")
        fridge_objects = fridge_root.get_objects_in_tree(exclude_base=False)
        fridge_handles = np.asarray([obj.get_handle() for obj in fridge_objects], dtype=np.int64)
        workspace = get_rlbench_robot_workspace()

        for episode in range(args.num_episodes):
            # Episode-local seeding prevents earlier trajectory length from changing
            # the next episode's initial placement.
            episode_seed = args.seed + 10_007 * episode
            set_random_seed(episode_seed)
            instructions, observation = task.reset()
            data = env.get_observation(observation)
            signature = scene_signature(fridge_root, top_joint)
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
                target = np.isin(instance, fridge_handles)
                corruption_seed = 701 + 1_000_003 * episode + 10_007 * step
                corrupted, labels, stats = structured_corruption(
                    points, target, data["camera_extrinsics"]["front"],
                    args.severity, corruption_seed,
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
                        rgb=data["rgb"][0], clean_points=points,
                        corrupted_points=corrupted, target_mask=target,
                        corruption_labels=labels,
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
                "task": "close_fridge", "variation": 0,
                "episode": episode, "episode_seed": episode_seed,
                "severity": args.severity, "success": success,
                "policy_steps": step + 1, "error": error,
                "initial_scene": signature,
                "first_predicted_position_world": first_prediction,
                "first_corruption": first_stats,
            }
            records.append(record)
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"severity={args.severity:.2f} episode={episode} "
                f"success={success} steps={step + 1}"
            )
    finally:
        env.env.shutdown()

    summary = {
        "task": "close_fridge", "variation": 0, "severity": args.severity,
        "num_episodes": len(records),
        "successes": sum(item["success"] for item in records),
        "success_rate": float(np.mean([item["success"] for item in records])),
        "corruption_definition": {
            "target_region": "all visible descendants of fridge_root in the front instance mask",
            "target_mix": {"invalid_holes": 0.55, "background_depth": 0.35, "range_distortion": 0.10},
            "edge_bias": True,
            "non_target_hole_fraction": 0.10 * args.severity,
            "rgb_instruction_robot_state_unchanged": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
