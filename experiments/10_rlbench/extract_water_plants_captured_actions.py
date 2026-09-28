"""Recover full rollout actions from saved PointACT attention captures."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_server import MODEL_MAP


RUNS = {
    "old_complete": (
        "baseline_job25521_complete",
        "checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000",
    ),
    "new_incomplete25_matched": (
        "incomplete25_clf_concerto_matched",
        "checkpoints/rlbench/pointact-rlbench-incomplete25-clf-concerto-hf/checkpoint-40000",
    ),
}


@torch.no_grad()
def extract_run(project_root: Path, source_root: Path, output_root: Path, label: str):
    source_name, checkpoint_rel = RUNS[label]
    source_dir = source_root / source_name
    checkpoint = project_root / checkpoint_rel
    model_config = json.loads((checkpoint / "config.json").read_text())
    model_class, processor_class = MODEL_MAP[model_config["architectures"][0]]
    model = model_class.from_pretrained(
        checkpoint, device_map={"": "cuda"}, local_files_only=True
    ).eval()
    processor = processor_class.from_pretrained(checkpoint, local_files_only=True)
    repo_id = next(iter(processor.robot_config["state_action_norm"]))
    source_summary = json.loads((source_dir / "summary.json").read_text())
    captures_dir = source_dir / "attention_captures"
    episodes = []
    recovery_errors = []
    for episode in source_summary["episodes_detail"]:
        actions = []
        for capture_index in range(
            episode["attention_capture_start"],
            episode["attention_capture_end"] + 1,
        ):
            capture_path = captures_dir / f"capture_{capture_index:06d}.npz"
            with np.load(capture_path) as capture:
                features = torch.from_numpy(capture["features"]).float().cuda()
                coordinates = torch.from_numpy(capture["coordinates"]).float().cuda()
                action_features = torch.from_numpy(capture["action_features"]).float().cuda()
                if model.config.use_robot_state:
                    action_features = action_features[:, 1:]
                npoints = torch.tensor([len(coordinates)], dtype=torch.long, device="cuda")
                output = model.action_head(
                    action_features,
                    features,
                    coordinates,
                    npoints,
                    return_cont_actions=True,
                )
                recovered = output[3]
                recovered_position = recovered[0, 0, :3].detach().float().cpu().numpy()
                captured_position = capture["predicted_position"].astype(np.float32)
                recovery_errors.append(float(np.max(np.abs(recovered_position - captured_position))))
                # The stored predicted_position is the exact position selected in
                # the original forward pass. Use it to avoid any float16 capture
                # quantization changing the winning point/bin during recovery.
                recovered = recovered.detach().float().cpu()
                recovered[0, 0, :3] = torch.from_numpy(captured_position)
                action = processor._build_action_output(
                    [repo_id], recovered, pred_rot_type="euler"
                ).action[0, 0]
                action[:3] += capture["scene_center"].astype(np.float32)
                actions.append(action.astype(np.float64).tolist())
        episodes.append(
            {
                "episode": episode["episode"],
                "source_success": episode["success"],
                "source_policy_steps": episode["policy_steps"],
                "capture_start": episode["attention_capture_start"],
                "capture_end": episode["attention_capture_end"],
                "actions": actions,
            }
        )
    result = {
        "label": label,
        "checkpoint": str(checkpoint),
        "source_run": str(source_dir),
        "episodes": episodes,
        "position_recovery_max_abs_error_m": max(recovery_errors),
        "position_recovery_median_abs_error_m": float(np.median(recovery_errors)),
        "note": (
            "Position uses the exact captured prediction plus scene center; "
            "rotation and gripper are recovered from captured final PTV3 features "
            "with the checkpoint action head."
        ),
    }
    output_path = output_root / f"{label}_captured_actions.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    for label in RUNS:
        path = extract_run(
            args.project_root, args.source_root, args.output_root, label
        )
        print(path)


if __name__ == "__main__":
    main()
