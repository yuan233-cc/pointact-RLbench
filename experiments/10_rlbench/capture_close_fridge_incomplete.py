"""Capture PointACT features/attention for close_fridge under point dropout.

This is an additive diagnostic: RGB, instruction, and robot state stay fixed;
only XYZ pixels are replaced by NaN before sending requests to the existing
attention-capture policy server.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_close_fridge_incomplete_study_20260916"
sys.path.insert(0, str(ROOT))

from rlbench.backend.utils import task_file_to_task_class
from pointact.robot_envs.rlbench_utils.environments import RLBenchEnv
from pointact.robot_envs.rlbench_utils.eval_utils import set_random_seed
from pointact.utils.rotation import convert_rotation
from pointact.utils.server_client import PolicyClient


CAMERAS = ("left_shoulder", "right_shoulder", "wrist", "front")
PORT = 15493
MISSING_RATES = (0.0, 0.25, 0.50, 0.75)
REPEAT_SEEDS = (1701, 1702, 1703)


def make_batch(data: dict, instruction: str, corrupted: list[np.ndarray]) -> dict:
    gripper = data["gripper"]
    rotation = convert_rotation(
        gripper[3:7], "quat", "euler",
        quat_order_src="xyzw", euler_order_dst="xyz",
    )
    state = np.concatenate([gripper[:3], rotation, gripper[7:]])
    batch = {
        "task": [instruction],
        "repo_id": ["hybridvla_10tasks_train_keysteps"],
        "observation.state": [state],
    }
    # The checkpoint processor_config selects only the front camera. Keep the
    # diagnostic input identical to rollout evaluation.
    index = CAMERAS.index("front")
    batch["observation.images.front_image"] = [data["rgb"][index]]
    batch["observation.points.front"] = [corrupted[index]]
    return batch


def main() -> None:
    scene_dir = OUTPUT / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    client = PolicyClient("127.0.0.1", PORT)
    if not client.ping():
        raise RuntimeError(f"PointACT attention server is not available on port {PORT}")

    set_random_seed(7)
    env = RLBenchEnv(
        apply_rgb=True, apply_depth=True, apply_pc=True, apply_mask=False,
        apply_cameras=CAMERAS, headless=True, image_size=(256, 256),
        cam_params_to_opencv=True, use_metric_depth=True,
    )
    env.env.launch()
    try:
        task = env.env.get_task(task_file_to_task_class("close_fridge"))
        task.set_variation(0)
        instructions, observation = task.reset()
        data = env.get_observation(observation)
        instruction = instructions[0]

        arrays = {"gripper": data["gripper"]}
        for index, camera in enumerate(CAMERAS):
            arrays[f"{camera}_rgb"] = data["rgb"][index]
            arrays[f"{camera}_points"] = data["pc"][index]
        np.savez_compressed(scene_dir / "observation.npz", **arrays)
        metadata = {
            "task": "close_fridge", "variation": 0, "seed": 7,
            "instruction": instruction, "actions_executed": 0,
            "camera_order": list(CAMERAS),
        }
        (scene_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        height, width = data["pc"][0].shape[:2]
        trials, masks = [], []

        def capture(rate: float, repeat: int, seed: int, keep: np.ndarray) -> None:
            corrupted = []
            for camera_index in range(len(CAMERAS)):
                points = data["pc"][camera_index].copy()
                points[~keep[camera_index]] = np.nan
                corrupted.append(points)
            client.get_action(
                make_batch(data, instruction, corrupted),
                options={"pred_rot_type": "euler"},
            )
            trials.append({
                "capture_index": len(trials),
                "missing_rate_requested": rate,
                "keep_rate_requested": 1.0 - rate,
                "repeat": repeat, "seed": seed,
                "kept_image_pixels": int(keep.sum()),
                "total_image_pixels": int(keep.size),
            })
            masks.append(keep.copy())
            print(f"capture {len(trials)-1}: missing={rate:.0%}, repeat={repeat}, kept={keep.mean():.1%}")

        capture(0.0, 0, 0, np.ones((len(CAMERAS), height, width), dtype=bool))
        for repeat, seed in enumerate(REPEAT_SEEDS):
            score = np.random.default_rng(seed).random((len(CAMERAS), height, width))
            for rate in MISSING_RATES[1:]:
                capture(rate, repeat, seed, score >= rate)

        np.savez_compressed(OUTPUT / "corruption_masks.npz", keep_masks=np.asarray(masks))
        study = {
            **metadata,
            "checkpoint_training_task": True,
            "model_input_camera": "front",
            "corruption": "Independent pixel-level XYZ dropout in the checkpoint-selected front camera; RGB is unchanged.",
            "missing_rates": list(MISSING_RATES),
            "repeat_seeds": list(REPEAT_SEEDS),
            "nested_within_each_repeat": True,
            "trials": trials,
        }
        (OUTPUT / "trials.json").write_text(json.dumps(study, indent=2), encoding="utf-8")
    finally:
        env.env.shutdown()


if __name__ == "__main__":
    main()
