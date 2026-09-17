"""Capture matched close_fridge inputs and true PointACT action attention.

The same reset scene, RGB, instruction, and robot state are used at all four
severity levels. Only the front-camera XYZ image is corrupted. No action is
executed in RLBench.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from rlbench.backend.utils import task_file_to_task_class

from pointact.robot_envs.rlbench_utils.environments import RLBenchEnv
from pointact.robot_envs.rlbench_utils.eval_utils import set_random_seed
from pointact.utils.server_client import PolicyClient
from run_close_fridge_realistic_missing_client import (
    scene_signature,
    state_vector,
    structured_corruption,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_close_fridge_realistic_missing_20260917"
INPUTS = OUTPUT / "attention_inputs"
PORT = 15506
SEVERITIES = (0.0, 0.25, 0.50, 0.75)
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
    if next(INPUTS.glob("severity_*.npz"), None) is not None:
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
    try:
        task = env.env.get_task(task_file_to_task_class("close_fridge"))
        task.set_variation(0)
        instructions, observation = task.reset()
        data = env.get_observation(observation)
        instruction = instructions[0]

        fridge_root = Shape("fridge_root")
        fridge_objects = fridge_root.get_objects_in_tree(exclude_base=False)
        fridge_handles = np.asarray(
            [obj.get_handle() for obj in fridge_objects], dtype=np.int64
        )
        clean_points = data["pc"][0]
        instance = np.rint(data["gt_mask"][0]).astype(np.int64)
        target = np.isin(instance, fridge_handles)

        for capture_index, severity in enumerate(SEVERITIES):
            corrupted, labels, stats = structured_corruption(
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
            target_file = INPUTS / f"severity_{severity:.2f}.npz"
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
            valid = np.isfinite(corrupted).all(axis=-1)
            trials.append(
                {
                    "capture_index": capture_index,
                    "severity": severity,
                    "file": target_file.name,
                    "valid_raw_points": int(valid.sum()),
                    "predicted_position_world": predicted.astype(float).tolist(),
                    "corruption": stats,
                }
            )
            print(
                f"capture={capture_index} severity={severity:.0%} "
                f"valid_raw={valid.sum()}/{valid.size} predicted={predicted.tolist()}"
            )

        metadata = {
            "task": "close_fridge",
            "variation": 0,
            "scene_seed": SCENE_SEED,
            "corruption_seed": CORRUPTION_SEED,
            "instruction": instruction,
            "actions_executed": 0,
            "same_scene_rgb_instruction_robot_state": True,
            "scene_signature": scene_signature(
                fridge_root, Joint("top_joint"),
            ),
            "attention_definition": (
                "Direct action-query to point-key softmax attention, averaged over "
                "heads and non-state action queries."
            ),
            "trials": trials,
        }
        (OUTPUT / "attention_trials.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    finally:
        env.env.shutdown()


if __name__ == "__main__":
    main()
