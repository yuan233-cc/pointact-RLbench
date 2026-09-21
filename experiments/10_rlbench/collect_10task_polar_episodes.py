"""Collect aligned RLBench demonstrations and render polar at keyframes.

Run in an environment with a CUDA GPU and the local custom RLBench/PyRep
installations. The script is resumable: completed episode and render summaries
are checked before either expensive stage is started again.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent
RLBENCH = WORKSPACE / "rlbench_custom_render/RLBench"
PYREP = WORKSPACE / "rlbench_custom_render/PyRep"
RLBENCH_PYTHON = WORKSPACE / ".conda/envs/rlbench/bin/python"
COPPELIASIM = WORKSPACE / ".deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MATERIALS = WORKSPACE / "rlbench_custom_render/material_plan_10tasks/material_profiles.json"
TASKS = (
    "close_box", "close_laptop_lid", "toilet_seat_down", "sweep_to_dustpan",
    "close_fridge", "phone_on_base", "take_umbrella_out_of_umbrella_stand",
    "take_frame_off_hanger", "stack_wine", "water_plants",
)


def completed(path: Path, task: str | None = None) -> dict | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("complete") is not True:
        raise ValueError(f"Incomplete result exists: {path}")
    if task is not None and record.get("task") != task:
        raise ValueError(f"Wrong task in {path}: {record.get('task')!r}")
    return record


def run_logged(command: list[str], env: dict[str, str], log: Path, timeout: int) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        stream.write("command: " + " ".join(command) + "\n")
        stream.flush()
        try:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)
        except Exception:
            tail = log.read_text(errors="replace")[-3000:]
            raise RuntimeError(f"Command failed; see {log}\n{tail}") from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RLBENCH / "output/ten_tasks_polar_train_20260921")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=20260921)
    parser.add_argument("--render-spp", type=int, default=512)
    parser.add_argument("--render-max-depth", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--episode-retries", type=int, default=3)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    args = parser.parse_args()
    if min(args.episodes_per_task, args.render_spp, args.max_attempts,
           args.episode_retries) < 1:
        parser.error("episode count, render spp, and attempts must be positive")
    for path in (RLBENCH_PYTHON, COPPELIASIM, MATERIALS):
        if not path.exists():
            raise FileNotFoundError(path)

    env = os.environ.copy()
    env.update({
        "COPPELIASIM_ROOT": str(COPPELIASIM),
        "LD_LIBRARY_PATH": str(COPPELIASIM) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
        "QT_QPA_PLATFORM_PLUGIN_PATH": str(COPPELIASIM),
        "QT_QPA_PLATFORM": "xcb",
        "DISPLAY": env.get("DISPLAY", ":1"),
        "PYTHONPATH": str(RLBENCH) + ":" + str(PYREP),
        "NATIVE_POLAR_NVCC": "/usr/local/cuda-11.8/bin/nvcc",
        "PYTHONNOUSERSITE": "1",
    })
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output / "progress.jsonl"
    for task in args.tasks:
        task_index = TASKS.index(task)
        for episode_index in range(args.episodes_per_task):
            output = args.output / task / f"episode_{episode_index:06d}"
            summary = completed(output / "summary.json", task)
            if summary is None:
                if output.exists():
                    raise FileExistsError(f"Partial episode needs inspection: {output}")
                free_gb = shutil.disk_usage(args.output).free / 1024**3
                if free_gb < args.min_free_gb:
                    raise RuntimeError(f"Only {free_gb:.1f} GiB free; stopping before {output}")
                for retry in range(args.episode_retries):
                    seed = (args.base_seed + task_index * 100000 + episode_index
                            + retry * 10000000)
                    command = [
                        str(RLBENCH_PYTHON), str(RLBENCH / "tools/collect_phone_polar_episode.py"),
                        "--task", task, "--capture-only", "--output", str(output),
                        "--materials", str(MATERIALS), "--resolution", "256",
                        "--spp", "1", "--max-depth", "4", "--seed", str(seed),
                        "--max-attempts", str(args.max_attempts),
                    ]
                    print(f"collect {task} {episode_index + 1}/{args.episodes_per_task} "
                          f"trial {retry + 1}/{args.episode_retries}", flush=True)
                    try:
                        run_logged(command, env, args.output / "logs" / task /
                                   f"episode_{episode_index:06d}_collect_trial{retry + 1}.log",
                                   timeout=1800)
                    except RuntimeError:
                        if output.exists() or retry + 1 == args.episode_retries:
                            raise
                        continue
                    break
                summary = completed(output / "summary.json", task)
                if summary is None:
                    raise RuntimeError(f"Collector did not produce {output / 'summary.json'}")

            render_path = output / "frames_spp512"
            render = completed(render_path / "render_summary.json")
            if render is None:
                if render_path.exists():
                    raise FileExistsError(f"Partial render needs inspection: {render_path}")
                seed = args.base_seed + task_index * 100000 + episode_index + 7000
                command = [
                    str(RLBENCH_PYTHON), str(RLBENCH / "tools/rerender_polar_keyframes.py"),
                    "--episode", str(output), "--output", str(render_path),
                    "--spp", str(args.render_spp), "--max-depth", str(args.render_max_depth),
                    "--seed", str(seed),
                ]
                print(f"render {task} {episode_index + 1}/{args.episodes_per_task}", flush=True)
                run_logged(command, env, args.output / "logs" / task /
                           f"episode_{episode_index:06d}_render.log", timeout=1800)
                render = completed(render_path / "render_summary.json")
                if render is None:
                    raise RuntimeError(f"Renderer did not produce {render_path / 'render_summary.json'}")
            if render["frame_count"] != summary["keyframe_count"]:
                raise ValueError(f"Frame mismatch in {output}")
            with progress.open("a") as stream:
                stream.write(json.dumps({"task": task, "episode_index": episode_index,
                                         "frames": render["frame_count"], "output": str(output)}) + "\n")
            print(f"complete {task} {episode_index + 1}/{args.episodes_per_task}: "
                  f"{render['frame_count']} keyframes", flush=True)


if __name__ == "__main__":
    main()
