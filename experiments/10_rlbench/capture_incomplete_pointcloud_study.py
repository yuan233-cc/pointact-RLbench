"""Run a reproducible point-dropout robustness study on one RLBench scene.

The RGB image, instruction, and robot state remain unchanged. Missing XYZ
pixels are set to NaN, which the existing workspace comparison filters out.
No environment action is executed and no model code is modified.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "PTV3_reach_target_action_attention_20260916" / "scene"
OUTPUT = ROOT / "PTV3_incomplete_pointcloud_study_20260916"
sys.path.insert(0, str(ROOT))

from pointact.utils.rotation import convert_rotation
from pointact.utils.server_client import PolicyClient


PORT = 15493
MISSING_RATES = (0.0, 0.25, 0.50, 0.75)
REPEAT_SEEDS = (1701, 1702, 1703)


def make_batch(scene, instruction: str, corrupted_points: np.ndarray) -> dict:
    gripper = scene["gripper"]
    rotation = convert_rotation(
        gripper[3:7], "quat", "euler",
        quat_order_src="xyzw", euler_order_dst="xyz",
    )
    state = np.concatenate([gripper[:3], rotation, gripper[7:]])
    return {
        "task": [instruction],
        "repo_id": ["hybridvla_10tasks_train_keysteps"],
        "observation.state": [state],
        "observation.images.front_image": [scene["rgb"]],
        "observation.points.front": [corrupted_points],
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scene = np.load(SOURCE / "observation.npz")
    scene_metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    points = scene["world_points"].copy()
    height, width = points.shape[:2]
    instance_mask = scene["instance_mask"]
    object_handles = {
        item["name"]: np.asarray(item["handles"], dtype=np.int64)
        for item in scene_metadata["objects"]
    }

    client = PolicyClient("127.0.0.1", PORT)
    if not client.ping():
        raise RuntimeError(f"PointACT attention server is not available on port {PORT}")

    trials = []
    keep_masks = []
    trial_index = 0

    def run_trial(missing_rate: float, repeat: int, seed: int, keep: np.ndarray):
        nonlocal trial_index
        corrupted = points.copy()
        corrupted[~keep] = np.nan
        client.get_action(
            make_batch(scene, scene_metadata["instruction"], corrupted),
            options={"pred_rot_type": "euler"},
        )
        object_visibility = {}
        for name, handles in object_handles.items():
            object_pixels = np.isin(instance_mask, handles)
            denominator = int(object_pixels.sum())
            numerator = int((object_pixels & keep).sum())
            object_visibility[name] = {
                "kept_pixels": numerator,
                "total_pixels": denominator,
                "kept_fraction": numerator / max(denominator, 1),
            }
        trials.append({
            "capture_index": trial_index,
            "missing_rate_requested": missing_rate,
            "keep_rate_requested": 1.0 - missing_rate,
            "repeat": repeat,
            "seed": seed,
            "kept_image_pixels": int(keep.sum()),
            "total_image_pixels": int(keep.size),
            "object_visibility": object_visibility,
        })
        keep_masks.append(keep)
        print(
            f"capture {trial_index}: missing={missing_rate:.0%}, "
            f"repeat={repeat}, kept={keep.mean():.1%}"
        )
        trial_index += 1

    run_trial(0.0, 0, 0, np.ones((height, width), dtype=bool))
    for repeat, seed in enumerate(REPEAT_SEEDS):
        # One score field per repeat makes dropout masks nested by severity.
        score = np.random.default_rng(seed).random((height, width))
        for missing_rate in MISSING_RATES[1:]:
            run_trial(missing_rate, repeat, seed, score >= missing_rate)

    np.savez_compressed(
        OUTPUT / "corruption_masks.npz",
        keep_masks=np.asarray(keep_masks, dtype=bool),
    )
    study = {
        "task": scene_metadata["task"],
        "instruction": scene_metadata["instruction"],
        "source_scene": str(SOURCE),
        "corruption": (
            "Independent pixel-level XYZ dropout. Dropped XYZ values are NaN and "
            "are removed by the unmodified workspace filter. RGB/VLM input is unchanged."
        ),
        "missing_rates": list(MISSING_RATES),
        "repeat_seeds": list(REPEAT_SEEDS),
        "nested_within_each_repeat": True,
        "actions_executed": 0,
        "trials": trials,
    }
    (OUTPUT / "trials.json").write_text(
        json.dumps(study, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
