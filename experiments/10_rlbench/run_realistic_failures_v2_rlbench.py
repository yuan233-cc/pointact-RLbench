"""Run local RLBench rollouts with realistic-failures-v2 point inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.utils import task_file_to_task_class

# Keep the RLBench environment's PyRep/cffi first, while reusing already-installed
# PointACT packages. This does not install or alter either environment.
WORKSPACE = Path(__file__).resolve().parents[3]
POINTACT_SITE = WORKSPACE / ".conda/envs/pointact/lib/python3.10/site-packages"
if POINTACT_SITE.is_dir() and str(POINTACT_SITE) not in sys.path:
    sys.path.append(str(POINTACT_SITE))

from pointact.robot_envs.rlbench_utils.environments import Mover, RLBenchEnv
from pointact.robot_envs.rlbench_utils.eval_utils import set_random_seed
from pointact.utils.rotation import convert_rotation
from pointact.utils.server_client import PolicyClient


TASKS = (
    "close_box", "close_laptop_lid", "toilet_seat_down", "sweep_to_dustpan",
    "close_fridge", "phone_on_base", "take_umbrella_out_of_umbrella_stand",
    "take_frame_off_hanger", "stack_wine", "water_plants",
)
REPO_ID = "hybridvla_10tasks_train_keysteps_realistic_failures_v2"
ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--corruption-seed", type=int, default=20260923)
    parser.add_argument("--eval-episode-offset", type=int, default=10000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15556)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/realistic_failures_v2_checkpoint7500_local_10task_25ep.json",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    return args


def state_euler(gripper: np.ndarray) -> np.ndarray:
    euler = convert_rotation(
        gripper[3:7], "quat", "euler", quat_order_src="xyzw", euler_order_dst="xyz"
    )
    return np.concatenate((gripper[:3], euler, gripper[7:])).astype(np.float32)


def append_progress(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        stream.flush()


def make_environment():
    env = RLBenchEnv(
        apply_rgb=True,
        apply_depth=True,
        apply_pc=True,
        apply_mask=False,
        apply_cameras=("front",),
        headless=True,
        image_size=(256, 256),
        cam_rand_factor=0,
        cam_params_to_opencv=True,
        use_metric_depth=True,
    )
    env.env.launch()
    return env


def run_task(args, policy, task_name, progress_path, start_episode=0):
    task_index = TASKS.index(task_name)
    env = make_environment()
    records = []
    try:
        task = env.env.get_task(task_file_to_task_class(task_name))
        task.set_variation(0)
        mover = Mover(task, max_tries=10)
        for episode in range(start_episode, args.episodes):
            episode_seed = args.seed + task_index * 1000 + episode
            set_random_seed(episode_seed)
            instructions, observation = task.reset()
            data = env.get_observation(observation)
            mover.reset(data["gripper"])
            source_episode = args.eval_episode_offset + task_index * args.episodes + episode
            reward = 0.0
            steps = 0
            episode_error = None
            action_chunk_shape = None
            for step in range(args.max_steps):
                batch = {
                    "observation.state": [state_euler(data["gripper"])],
                    "observation.images.front_image": [data["rgb"][0]],
                    "observation.points.front": [data["pc"][0]],
                    "task": [instructions[0]],
                    "repo_id": [REPO_ID],
                    "realistic_failure_task_index": [task_index],
                    "realistic_failure_source_episode": [source_episode],
                    "realistic_failure_base_seed": [args.corruption_seed],
                    "realistic_failure_step": [step],
                }
                output = policy.get_action(
                    batch, options={"pred_rot_type": "euler", "remove_arm": False}
                )
                action_chunk = np.asarray(output.action[0])
                if action_chunk.ndim != 2 or action_chunk.shape[0] < 1:
                    raise ValueError(f"Expected a nonempty action chunk, got {action_chunk.shape}")
                action_chunk_shape = list(action_chunk.shape)
                steps = step + 1
                try:
                    observation, reward, terminate, _ = mover(action_chunk[0], verbose=False)
                    data = env.get_observation(observation)
                except (IKError, ConfigurationPathError, InvalidActionError) as error:
                    episode_error = f"{type(error).__name__}: {error}"
                    reward = 0.0
                    break
                if reward == 1 or terminate:
                    break
            record = {
                "task": task_name,
                "episode": episode,
                "episode_seed": episode_seed,
                "source_episode": source_episode,
                "success": bool(reward == 1),
                "reward": float(reward),
                "steps": steps,
                "action_chunk_shape": action_chunk_shape,
            }
            if episode_error is not None:
                record["error"] = episode_error
            records.append(record)
            append_progress(progress_path, record)
            print(json.dumps(record), flush=True)
    finally:
        env.env.shutdown()
    return records


def main():
    args = parse_args()
    progress_path = args.output.with_name(args.output.stem + ".progress.jsonl")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite final output: {args.output}")
    if progress_path.exists() and not args.resume:
        raise FileExistsError(f"Progress exists; pass --resume: {progress_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if progress_path.exists():
        records = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
        expected = [(task, episode) for task in args.tasks for episode in range(args.episodes)]
        actual = [(row["task"], row["episode"]) for row in records]
        if actual != expected[:len(actual)]:
            raise ValueError("Progress rows are not the expected ordered prefix")

    policy = PolicyClient(args.host, args.port, timeout_ms=120000)
    if not policy.ping():
        raise RuntimeError(f"Policy server unavailable at {args.host}:{args.port}")
    for task_name in args.tasks:
        completed = sum(row["task"] == task_name for row in records)
        if completed < args.episodes:
            records.extend(run_task(args, policy, task_name, progress_path, completed))

    summary = {
        "checkpoint_step": 7500,
        "repo_id": REPO_ID,
        "input": "front XYZRGB, 1cm voxel, realistic-failures-v2 corruption",
        "corruption_seed": args.corruption_seed,
        "eval_episode_offset": args.eval_episode_offset,
        "progress_file": str(progress_path),
        "episodes": records,
        "successes": sum(row["success"] for row in records),
        "total": len(records),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary={args.output} success={summary['successes']}/{summary['total']}")


if __name__ == "__main__":
    main()
