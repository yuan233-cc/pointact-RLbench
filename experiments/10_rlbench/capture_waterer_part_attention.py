"""Capture the initial water_plants scene with masks and PointACT attention."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PTV3_waterer_part_attention_20260916"
sys.path.insert(0, str(ROOT))

from pyrep.objects.shape import Shape
from rlbench.backend.utils import task_file_to_task_class
from pointact.robot_envs.rlbench_utils.environments import RLBenchEnv
from pointact.robot_envs.rlbench_utils.eval_utils import set_random_seed
from pointact.utils.rotation import convert_rotation
from pointact.utils.server_client import PolicyClient


CAMERAS = ("left_shoulder", "right_shoulder", "wrist", "front")
PORT = 15494


def main() -> None:
    scene_dir = OUTPUT / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    client = PolicyClient("127.0.0.1", PORT)
    if not client.ping():
        raise RuntimeError(f"PointACT attention server is not available on port {PORT}")

    set_random_seed(7)
    env = RLBenchEnv(
        apply_rgb=True,
        apply_depth=True,
        apply_pc=True,
        apply_mask=True,
        apply_cameras=CAMERAS,
        headless=True,
        image_size=(256, 256),
        cam_params_to_opencv=True,
        use_metric_depth=True,
    )
    for camera in CAMERAS:
        getattr(env.obs_config, f"{camera}_camera").masks_as_one_channel = True
    env.env.launch()
    try:
        task = env.env.get_task(task_file_to_task_class("water_plants"))
        task.set_variation(0)
        instructions, observation = task.reset()
        data = env.get_observation(observation)
        instruction = instructions[0]

        waterer = Shape("waterer")
        objects = waterer.get_objects_in_tree(exclude_base=False)
        metadata = {
            "task": "water_plants",
            "variation": 0,
            "seed": 7,
            "instruction": instruction,
            "actions_executed": 0,
            "camera_order": list(CAMERAS),
            "waterer": {
                "handle": waterer.get_handle(),
                "position": np.asarray(waterer.get_position()).tolist(),
                "matrix": np.asarray(waterer.get_matrix()).reshape(4, 4).tolist(),
                "bbox": np.asarray(waterer.get_bounding_box()).tolist(),
                "objects": [
                    {
                        "handle": item.get_handle(),
                        "name": item.get_name(),
                        "type": str(item.get_type()),
                    }
                    for item in objects
                ],
            },
        }
        (scene_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        arrays = {"gripper": data["gripper"]}
        for index, camera in enumerate(CAMERAS):
            arrays[f"{camera}_rgb"] = data["rgb"][index]
            arrays[f"{camera}_points"] = data["pc"][index]
            arrays[f"{camera}_mask"] = data["gt_mask"][index].astype(np.int64)
        np.savez_compressed(scene_dir / "observation.npz", **arrays)

        gripper = data["gripper"]
        rotation = convert_rotation(
            gripper[3:7], "quat", "euler",
            quat_order_src="xyzw", euler_order_dst="xyz",
        )
        state = np.concatenate([gripper[:3], rotation, gripper[7:]])
        batch = {
            "observation.state": [state],
            "task": [instruction],
            "repo_id": ["hybridvla_10tasks_train_keysteps"],
        }
        for index, camera in enumerate(CAMERAS):
            batch[f"observation.images.{camera}_image"] = [data["rgb"][index]]
            batch[f"observation.points.{camera}"] = [data["pc"][index]]
        client.get_action(batch, options={"pred_rot_type": "euler"})
        print(json.dumps(metadata, indent=2))
    finally:
        env.env.shutdown()


if __name__ == "__main__":
    main()
