"""Run local RLBench rollouts with training-matched polar filled9 inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError

# The existing local environments split simulator and preprocessing packages:
# RLBench owns PyRep/cffi, while PointACT owns cv2/av. Append (do not prepend)
# the latter so the active RLBench environment keeps precedence.
WORKSPACE = Path(__file__).resolve().parents[3]
POINTACT_SITE = WORKSPACE / ".conda/envs/pointact/lib/python3.10/site-packages"
if POINTACT_SITE.is_dir() and str(POINTACT_SITE) not in sys.path:
    sys.path.append(str(POINTACT_SITE))

from filled9_inference import build_filled9, polar_frame
from pointact.robot_envs.rlbench_utils.environments import Mover
from pointact.robot_envs.rlbench_utils.eval_utils import set_random_seed
from pointact.utils.rotation import convert_rotation
from pointact.utils.server_client import PolicyClient


TASKS = (
    "close_box", "close_laptop_lid", "toilet_seat_down", "sweep_to_dustpan",
    "close_fridge", "phone_on_base", "take_umbrella_out_of_umbrella_stand",
    "take_frame_off_hanger", "stack_wine", "water_plants",
)
REPO_ID = "hybridvla_10tasks_train_keysteps_polar_rlbench9_v2"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATERIALS = (
    ROOT / "robot_data/rlbench/lerobot_point_lmdb" / REPO_ID
    / "material_profiles_10tasks.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--corruption-seed", type=int, default=20260923)
    parser.add_argument("--eval-episode-offset", type=int, default=10000)
    parser.add_argument("--checkpoint-step", type=int, default=7500)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/filled9_checkpoint7500_local.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    return args


def observation_config(task_name: str, materials_path: Path, seed: int):
    from pyrep.const import RenderMode
    from rlbench.material_profiles import load_task_material_overrides
    from rlbench.native_config import NativePolarizationConfig
    from rlbench.observation_config import CameraConfig, ObservationConfig

    native = NativePolarizationConfig(
        spp=512,
        max_depth=8,
        seed=seed,
        device=0,
        rgb_source="coppeliasim",
        lighting="reference",
        geometry_source="rlbench",
        material_overrides=load_task_material_overrides(materials_path, task_name),
    )
    unused = CameraConfig()
    unused.set_all(False)
    front = CameraConfig(
        rgb=True,
        depth=True,
        point_cloud=True,
        mask=False,
        image_size=(256, 256),
        render_mode=RenderMode.OPENGL,
        depth_in_meters=True,
        polarization=True,
        polarization_backend="native",
        polarization_config=native,
    )
    return ObservationConfig(
        front_camera=front,
        left_shoulder_camera=unused,
        right_shoulder_camera=unused,
        wrist_camera=unused,
        overhead_camera=unused,
        joint_forces=False,
        joint_positions=False,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=True,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )


def make_environment(task_name: str, materials: Path, seed: int):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.backend.utils import task_file_to_task_class
    from rlbench.environment import Environment

    mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
        gripper_action_mode=Discrete(),
    )
    environment = Environment(
        mode, obs_config=observation_config(task_name, materials, seed), headless=True
    )
    environment.launch()
    return environment, environment.get_task(task_file_to_task_class(task_name))


def state_euler(observation):
    gripper = np.concatenate(
        (np.asarray(observation.gripper_pose, dtype=np.float32),
         [observation.gripper_open])
    )
    euler = convert_rotation(
        gripper[3:7], "quat", "euler", quat_order_src="xyzw", euler_order_dst="xyz"
    )
    return gripper, np.concatenate((gripper[:3], euler, gripper[7:])).astype(np.float32)


def append_progress(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()


def run_task(
    args, policy: PolicyClient, task_name: str, progress_path: Path, start_episode: int = 0
):
    task_index = TASKS.index(task_name)
    # Match the stock RLBench evaluator: each task worker starts from the same
    # seed once, then consumes one continuous RNG stream across its episodes.
    # A task-per-process launcher makes this independent of worker scheduling.
    set_random_seed(args.seed)
    environment, task = make_environment(task_name, args.materials, args.seed)
    records = []
    try:
        task.set_variation(0)
        mover = Mover(task, max_tries=10)
        for episode in range(start_episode, args.episodes):
            instructions, observation = task.reset()
            gripper, _ = state_euler(observation)
            mover.reset(gripper)
            reward = 0.0
            steps = 0
            episode_error = None
            input_stats = []
            for step in range(args.max_steps):
                frame = polar_frame(observation)
                source_episode = args.eval_episode_offset + task_index * args.episodes + episode
                points, _, _, stats = build_filled9(
                    frame,
                    task_index=task_index,
                    source_episode=source_episode,
                    corruption_seed=args.corruption_seed,
                )
                _, state = state_euler(observation)
                output = policy.get_action(
                    {
                        "observation.state": [state],
                        "observation.points": [points],
                        "task": [instructions[0]],
                        "repo_id": [REPO_ID],
                    },
                    options={"pred_rot_type": "euler", "remove_arm": False},
                )
                action_chunk = np.asarray(output.action[0])
                if action_chunk.ndim != 2 or action_chunk.shape[0] != 1:
                    raise ValueError(f"Expected one-action chunk, got {action_chunk.shape}")
                steps = step + 1
                input_stats.append(stats)
                try:
                    observation, reward, terminate, _ = mover(action_chunk[0], verbose=False)
                except (IKError, ConfigurationPathError, InvalidActionError) as error:
                    episode_error = f"{type(error).__name__}: {error}"
                    reward = 0.0
                    break
                if reward == 1 or terminate:
                    break
            record = {
                "task": task_name,
                "episode": episode,
                "success": bool(reward == 1),
                "reward": float(reward),
                "steps": steps,
                "source_episode": args.eval_episode_offset + task_index * args.episodes + episode,
                "input_stats": input_stats,
            }
            if episode_error is not None:
                record["error"] = episode_error
            records.append(record)
            append_progress(progress_path, record)
            print(json.dumps({key: value for key, value in record.items() if key != "input_stats"}))
    finally:
        environment.shutdown()
    return records


def main():
    args = parse_args()
    progress_path = args.output.with_name(args.output.stem + ".progress.jsonl")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite final output: {args.output}")
    if progress_path.exists() and not args.resume:
        raise FileExistsError(
            f"Progress exists; pass --resume to continue: {progress_path}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if progress_path.exists():
        records = [
            json.loads(line) for line in progress_path.read_text().splitlines()
            if line.strip()
        ]
        expected_prefix = []
        for task_name in args.tasks:
            expected_prefix.extend((task_name, episode) for episode in range(args.episodes))
        actual_prefix = [(record["task"], record["episode"]) for record in records]
        if actual_prefix != expected_prefix[:len(actual_prefix)]:
            raise ValueError("Progress rows are not the expected ordered task/episode prefix")
    policy = PolicyClient(args.host, args.port)
    if not policy.ping():
        raise RuntimeError(f"filled9 policy server is not available at {args.host}:{args.port}")
    for task_name in args.tasks:
        completed = sum(record["task"] == task_name for record in records)
        if completed < args.episodes:
            records.extend(
                run_task(args, policy, task_name, progress_path, start_episode=completed)
            )
    summary = {
        "checkpoint_step": args.checkpoint_step,
        "repo_id": REPO_ID,
        "polar": {"backend": "native", "spp": 512, "max_depth": 8,
                  "lighting": "reference", "geometry_source": "rlbench"},
        "corruption_seed": args.corruption_seed,
        "randomization": {
            "mode": "original_rlbench_worker_stream",
            "seed": args.seed,
            "reseed_each_episode": False,
        },
        "progress_file": str(progress_path),
        "episodes": records,
        "successes": sum(record["success"] for record in records),
        "total": len(records),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary={args.output} success={summary['successes']}/{summary['total']}")


if __name__ == "__main__":
    main()
