"""Capture matched PTV3 action attention for two trained RLBench tasks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyrep.objects.shape import Shape
from rlbench.backend.utils import task_file_to_task_class

from pointact.robot_envs.rlbench_utils.environments import RLBenchEnv
from pointact.robot_envs.rlbench_utils.eval_utils import set_random_seed
from pointact.utils.server_client import PolicyClient
from run_close_fridge_realistic_missing_client import state_vector, structured_corruption
from run_task_realistic_missing_client import corrupt_visible_target, target_handles


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_wine_umbrella_realistic_missing_20260917"
INPUTS = OUTPUT / "attention_inputs"
PORT = 15508
SEVERITIES = (0.0, 0.25, 0.50, 0.75)
TASKS = (
    ("stack_wine", "wine_bottle"),
    ("take_umbrella_out_of_umbrella_stand", "umbrella"),
)
SCENE_SEED = 7
CORRUPTION_SEED = 701


def make_batch(data: dict, instruction: str, points: np.ndarray) -> dict:
    return {
        "task": [instruction],
        "repo_id": ["hybridvla_10tasks_train_keysteps"],
        "observation.state": [state_vector(data["gripper"])],
        "observation.images.front_image": [data["rgb"][0]],
        "observation.points.front": [points],
    }


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    if next(INPUTS.rglob("severity_*.npz"), None) is not None:
        raise FileExistsError(f"Refusing to overwrite existing inputs in {INPUTS}")
    client = PolicyClient("127.0.0.1", PORT)
    if not client.ping():
        raise RuntimeError(f"PointACT attention server is unavailable on port {PORT}")

    set_random_seed(SCENE_SEED)
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
    trials = []
    capture_index = 0
    try:
        for task_name, target_name in TASKS:
            set_random_seed(SCENE_SEED)
            task = env.env.get_task(task_file_to_task_class(task_name))
            task.set_variation(0)
            instructions, observation = task.reset()
            data = env.get_observation(observation)
            instruction = instructions[0]
            target_root = Shape(target_name)
            handles = target_handles(target_root)
            clean_points = data["pc"][0]
            instance = np.rint(data["gt_mask"][0]).astype(np.int64)
            target = np.isin(instance, handles)
            if not target.any():
                raise RuntimeError(f"No visible {target_name} pixels for {task_name}")
            task_input_dir = INPUTS / task_name
            task_input_dir.mkdir(parents=True, exist_ok=True)

            for severity in SEVERITIES:
                corrupted, labels, stats = corrupt_visible_target(
                    clean_points,
                    target,
                    data["camera_extrinsics"]["front"],
                    severity,
                    CORRUPTION_SEED,
                )
                client.reset(options={"seed": SCENE_SEED})
                result = client.get_action(
                    make_batch(data, instruction, corrupted),
                    options={"pred_rot_type": "euler", "remove_arm": False},
                )
                predicted = np.asarray(result.action[0, 0, :3], dtype=np.float32)
                target_file = task_input_dir / f"severity_{severity:.2f}.npz"
                np.savez_compressed(
                    target_file,
                    rgb=data["rgb"][0],
                    clean_points=clean_points,
                    corrupted_points=corrupted,
                    target_mask=target,
                    corruption_labels=labels,
                    instance_mask=instance,
                    camera_extrinsics=data["camera_extrinsics"]["front"],
                    gripper=data["gripper"],
                    predicted_position_world=predicted,
                )
                trials.append({
                    "capture_index": capture_index,
                    "task": task_name,
                    "target_object": target_name,
                    "variation": 0,
                    "severity": severity,
                    "file": str(target_file.relative_to(INPUTS)),
                    "instruction": instruction,
                    "target_visible_pixels": int(target.sum()),
                    "predicted_position_world": predicted.astype(float).tolist(),
                    "target_pose": np.asarray(target_root.get_pose(), dtype=float).tolist(),
                    "corruption": stats,
                })
                print(
                    f"capture={capture_index} task={task_name} severity={severity:.0%} "
                    f"target_pixels={target.sum()} predicted={predicted.tolist()}",
                    flush=True,
                )
                capture_index += 1
    finally:
        env.env.shutdown()

    metadata = {
        "scene_seed": SCENE_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "actions_executed": 0,
        "same_scene_rgb_instruction_robot_state_within_each_task": True,
        "attention_definition": (
            "Direct action-query to point-key softmax attention, averaged over "
            "heads and non-state action queries."
        ),
        "trials": trials,
    }
    (OUTPUT / "attention_trials.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
