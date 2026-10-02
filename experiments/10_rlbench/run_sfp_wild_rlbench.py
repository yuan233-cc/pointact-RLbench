"""Evaluate SfP-Wild + PointACT with the live RLBench polar renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError

# The simulator environment owns PyRep while the PointACT environment provides
# av/cv2 for the shared corruption and point-cloud utilities.
WORKSPACE = Path(__file__).resolve().parents[3]
POINTACT_SITE = WORKSPACE / ".conda/envs/pointact/lib/python3.10/site-packages"
if POINTACT_SITE.is_dir() and str(POINTACT_SITE) not in sys.path:
    sys.path.append(str(POINTACT_SITE))

from filled9_inference import build_incomplete9, polar_frame
from run_filled9_rlbench import (
    DEFAULT_MATERIALS,
    REPO_ID,
    ROOT,
    TASKS,
    append_progress,
    make_environment,
    state_euler,
)
from pointact.robot_envs.rlbench_utils.environments import Mover
from pointact.robot_envs.rlbench_utils.eval_utils import set_random_seed
from pointact.robot_envs.rlbench_utils.sfp_adapter import sfp_inputs_from_polar_frame
from pointact.utils.server_client import PolicyClient


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--corruption-seed", type=int, default=20260923)
    parser.add_argument("--eval-episode-offset", type=int, default=10000)
    parser.add_argument("--checkpoint-step", type=int, default=-1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15570)
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/sfp_wild_rlbench_local.json"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    return args


def policy_batch(observation, instruction, *, task_index, source_episode, corruption_seed):
    """Build one inference batch with aligned point and SfP camera inputs."""
    frame = polar_frame(observation)
    points, _, _, stats = build_incomplete9(
        frame,
        task_index=task_index,
        source_episode=source_episode,
        corruption_seed=corruption_seed,
    )
    sfp = sfp_inputs_from_polar_frame(frame)
    batch = {
        "observation.images.front_image": [frame["rgb"]],
        "observation.points": [points],
        "task": [instruction],
        "repo_id": [REPO_ID],
    }
    # sfp values already contain the one-view dimension. The outer list is the
    # policy batch dimension expected by VLAEncDec3DProcessor.select_action.
    batch.update({key: [value] for key, value in sfp.items()})
    return batch, stats


def run_task(
    args, policy: PolicyClient, task_name: str, progress_path: Path, start_episode: int = 0
):
    task_index = TASKS.index(task_name)
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
            source_episode = args.eval_episode_offset + task_index * args.episodes + episode
            reward = 0.0
            steps = 0
            episode_error = None
            input_stats = []
            for step in range(args.max_steps):
                batch, stats = policy_batch(
                    observation,
                    instructions[0],
                    task_index=task_index,
                    source_episode=source_episode,
                    corruption_seed=args.corruption_seed,
                )
                _, state = state_euler(observation)
                batch["observation.state"] = [state]
                output = policy.get_action(
                    batch,
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
                "source_episode": source_episode,
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
        raise FileExistsError(f"Progress exists; pass --resume to continue: {progress_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if progress_path.exists():
        records = [
            json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()
        ]
        expected_prefix = [
            (task_name, episode)
            for task_name in args.tasks
            for episode in range(args.episodes)
        ]
        actual_prefix = [(record["task"], record["episode"]) for record in records]
        if actual_prefix != expected_prefix[: len(actual_prefix)]:
            raise ValueError("Progress rows are not the expected ordered task/episode prefix")

    policy = PolicyClient(args.host, args.port)
    if not policy.ping():
        raise RuntimeError(f"SfP policy server is unavailable at {args.host}:{args.port}")
    for task_name in args.tasks:
        completed = sum(record["task"] == task_name for record in records)
        if completed < args.episodes:
            records.extend(run_task(args, policy, task_name, progress_path, completed))

    summary = {
        "checkpoint_step": args.checkpoint_step,
        "repo_id": REPO_ID,
        "point_input": "training-matched incomplete XYZRGB+polar",
        "sfp_input": "live RGB luminance proxy + DoLP/AoLP + calibrated viewing rays",
        "polar": {
            "backend": "native",
            "spp": 512,
            "max_depth": 8,
            "lighting": "reference",
            "geometry_source": "rlbench",
        },
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
