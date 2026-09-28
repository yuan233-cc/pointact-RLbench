"""Replay actions recovered from an existing RLBench attention evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.utils import task_file_to_task_class

from pointact.robot_envs.rlbench_utils.environments import CAMERA_ATTR, Mover, RLBenchEnv
from pointact.robot_envs.rlbench_utils.eval_utils import set_random_seed
from run_water_plants_geometry_client import WaterPlantsGeometryTracker


CAMERA_NAMES = ("left_shoulder", "right_shoulder", "wrist", "front")


def append_jsonl(path: Path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episodes", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    action_data = json.loads(args.actions.read_text())
    episodes = action_data["episodes"][: args.max_episodes]
    set_random_seed(args.seed)
    env = RLBenchEnv(
        data_path="",
        apply_rgb=True,
        apply_depth=True,
        apply_pc=True,
        apply_mask=False,
        apply_cameras=CAMERA_NAMES,
        headless=True,
        image_size=[256, 256],
        cam_rand_factor=0,
        cam_params_to_opencv=True,
        use_metric_depth=True,
    )
    env.env.launch()
    task = env.env.get_task(task_file_to_task_class("water_plants"))
    task.set_variation(0)
    mover = Mover(task, max_tries=10)

    tracker = None
    recording = False
    callback_tick = 0
    camera_stride = max(
        1,
        round(1.0 / (2.0 * task._scene.pyrep.get_simulation_timestep())),
    )

    def diagnostic_callback():
        nonlocal callback_tick
        if recording:
            callback_tick += 1
            if callback_tick % camera_stride == 0:
                camera = getattr(task._scene, CAMERA_ATTR["front"])
                camera.handle_explicitly()
                camera.capture_rgb()
        if tracker is not None:
            tracker.simulator_step()

    task._scene.register_step_callback(diagnostic_callback)
    try:
        for source_episode in episodes:
            episode_id = int(source_episode["episode"])
            _instructions, obs = task.reset()
            if tracker is None:
                tracker = WaterPlantsGeometryTracker(task)
            tracker.start_episode(episode_id)
            callback_tick = 0
            recording = True
            obs_state = env.get_observation(obs)
            mover.reset(obs_state["gripper"])
            reward = 0.0
            terminate = False
            for step_id, action_list in enumerate(source_episode["actions"]):
                action = np.asarray(action_list, dtype=np.float64)
                tracker.start_action(step_id, action, obs_state["gripper"])
                try:
                    obs, reward, terminate, _ = mover(action, verbose=False)
                    obs_state = env.get_observation(obs)
                    tracker.finish_action(obs_state["gripper"], reward, terminate)
                except (IKError, ConfigurationPathError, InvalidActionError) as error:
                    reward = 0.0
                    terminate = True
                    tracker.finish_action(
                        None,
                        reward,
                        terminate,
                        error=f"{type(error).__name__}: {error}",
                    )
                    break
                if reward == 1:
                    break
            recording = False
            for record in tracker.episode_records:
                append_jsonl(args.output_dir / "water_plants_geometry_steps.jsonl", record)
            summary = tracker.episode_summary(reward == 1)
            summary.update(
                {
                    "source_success": bool(source_episode["source_success"]),
                    "replay_success": bool(reward == 1),
                    "source_policy_steps": int(source_episode["source_policy_steps"]),
                    "replayed_policy_steps": len(tracker.episode_records),
                }
            )
            append_jsonl(
                args.output_dir / "water_plants_geometry_episodes.jsonl", summary
            )
            print(
                action_data["label"],
                "episode",
                episode_id,
                "source_success",
                source_episode["source_success"],
                "replay_success",
                bool(reward == 1),
                "steps",
                len(tracker.episode_records),
                flush=True,
            )
    finally:
        env.env.shutdown()


if __name__ == "__main__":
    main()
