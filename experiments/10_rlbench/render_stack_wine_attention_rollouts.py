"""Render per-episode PointACT attention videos and summarize RLBench rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


ATTENTION_KEY = "action_attention_stage4_block2_point_weights"
ATTENTION_COORD_KEY = "action_attention_stage4_block2_point_coordinates"
RECONSTRUCTION_ERROR_KEY = (
    "action_attention_stage4_block2_reconstruction_max_abs_error"
)
RECONSTRUCTION_COSINE_KEY = (
    "action_attention_stage4_block2_reconstruction_cosine"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - radius, center + radius]


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    if rgb.size and rgb.max() > 1.5:
        rgb /= 255.0
    if rgb.size and rgb.min() < 0.0:
        low, high = np.percentile(rgb, [1, 99], axis=0)
        rgb = (rgb - low) / np.maximum(high - low, 1e-8)
    return np.clip(rgb, 0.0, 1.0)


def set_axes(ax, low: np.ndarray, high: np.ndarray) -> None:
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=34, azim=-138)
    ax.set_proj_type("ortho")


def load_frame(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as capture:
        missing = {
            "input_coordinates",
            "input_rgb",
            "input_to_stage4",
            ATTENTION_KEY,
            ATTENTION_COORD_KEY,
            "scene_center",
            "predicted_position",
        } - set(capture.files)
        if missing:
            raise KeyError(f"{path} is missing attention fields: {sorted(missing)}")
        center = capture["scene_center"].astype(np.float32).reshape(-1, 3)[0]
        xyz = capture["input_coordinates"].astype(np.float32) + center
        stage4_weights = capture[ATTENTION_KEY].astype(np.float32)
        dense_attention = stage4_weights[
            capture["input_to_stage4"].astype(np.int64)
        ]
        frame = {
            "xyz": xyz,
            "rgb": normalize_rgb(capture["input_rgb"]),
            "attention": dense_attention,
            "stage4_attention": stage4_weights,
            "predicted": capture["predicted_position"].astype(np.float32) + center,
            "reconstruction_error": np.asarray(
                capture[RECONSTRUCTION_ERROR_KEY], dtype=np.float32
            ),
            "reconstruction_cosine": np.asarray(
                capture[RECONSTRUCTION_COSINE_KEY], dtype=np.float32
            ),
        }
        if "corruption_full_coordinates_world" in capture:
            frame.update({
                "corruption_full_xyz": capture[
                    "corruption_full_coordinates_world"
                ].astype(np.float32),
                "corruption_full_rgb": normalize_rgb(
                    capture["corruption_full_rgb"]
                ),
                "corruption_removed_xyz": capture[
                    "corruption_removed_coordinates_world"
                ].astype(np.float32),
                "corruption_missing_rate": np.asarray(
                    capture["corruption_missing_rate"], dtype=np.float32
                ),
                "corruption_episode_id": np.asarray(
                    capture["corruption_episode_id"], dtype=np.int64
                ),
                "corruption_field_centers": capture[
                    "corruption_field_centers_world"
                ].astype(np.float32),
                "corruption_field_radii": capture[
                    "corruption_field_radii"
                ].astype(np.float32),
            })
            for key, dtype in (
                ("corruption_requested_missing_rate", np.float32),
                ("corruption_seed", np.int64),
                ("corruption_num_holes", np.int64),
            ):
                if key in capture:
                    frame[key] = np.asarray(capture[key], dtype=dtype)
        return frame


def render_episode(
    capture_paths: list[Path],
    result: dict,
    label: str,
    output_dir: Path,
    episode_index: int,
    attention_limits: tuple[float, float],
    xyz_limits: tuple[np.ndarray, np.ndarray],
    fps: int,
    task: str,
    variation: int,
) -> dict:
    outcome = "success" if result["success"] else "failure"
    stem = f"{task}+{variation}_episode_{episode_index:06d}_{outcome}_attention"
    video_dir = output_dir / "attention_videos"
    still_dir = output_dir / "attention_stills" / f"episode_{episode_index:06d}_{outcome}"
    video_dir.mkdir(parents=True, exist_ok=True)
    still_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{stem}.mp4"

    fig = plt.figure(figsize=(13.6, 6.8))
    fig.subplots_adjust(left=0.01, right=0.93, bottom=0.06, top=0.84, wspace=0.03)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3000,
        metadata={"title": stem, "artist": "PointACT RLBench evaluation"},
        extra_args=["-pix_fmt", "yuv420p"],
    )
    still_indices = {0, len(capture_paths) // 2, len(capture_paths) - 1}
    reconstruction_errors = []
    reconstruction_cosines = []

    with writer.saving(fig, str(video_path), dpi=110):
        for step, capture_path in enumerate(capture_paths):
            frame = load_frame(capture_path)
            reconstruction_errors.append(float(frame["reconstruction_error"]))
            reconstruction_cosines.append(float(frame["reconstruction_cosine"]))
            fig.clear()
            ax_rgb = fig.add_subplot(1, 2, 1, projection="3d")
            ax_attention = fig.add_subplot(1, 2, 2, projection="3d")
            point_size = 3.0 if len(frame["xyz"]) > 1000 else 8.0
            ax_rgb.scatter(
                frame["xyz"][:, 0], frame["xyz"][:, 1], frame["xyz"][:, 2],
                c=frame["rgb"], s=point_size, linewidths=0,
            )
            if "corruption_removed_xyz" in frame:
                removed = frame["corruption_removed_xyz"]
                ax_rgb.scatter(
                    removed[:, 0], removed[:, 1], removed[:, 2],
                    c="#00e5ff", s=point_size * 1.3, linewidths=0, alpha=0.8,
                    label="removed by training-matched 25% corruption",
                )
                ax_rgb.legend(loc="upper left", fontsize=7)
            attention_plot = ax_attention.scatter(
                frame["xyz"][:, 0], frame["xyz"][:, 1], frame["xyz"][:, 2],
                c=frame["attention"], cmap="magma",
                vmin=attention_limits[0], vmax=attention_limits[1],
                s=point_size, linewidths=0,
            )
            for ax in (ax_rgb, ax_attention):
                ax.scatter(
                    *frame["predicted"], marker="*", s=150, c="cyan",
                    edgecolors="black", linewidths=0.8,
                )
                set_axes(ax, *xyz_limits)
            if "corruption_missing_rate" in frame:
                ax_rgb.set_title(
                    "Retained model input (RGB) + removed points (cyan)\n"
                    f"actual missing={float(frame['corruption_missing_rate']):.2%}"
                )
            else:
                ax_rgb.set_title("Input point cloud (RGB) + predicted action")
            ax_attention.set_title("Stage-4 block-2 action→point attention")
            colorbar = fig.colorbar(attention_plot, ax=ax_attention, shrink=0.67, pad=0.03)
            colorbar.set_label("attention weight (shared scale for this checkpoint)")
            fig.suptitle(
                f"{label} | {task} variation {variation} | episode {episode_index:02d} "
                f"{outcome.upper()} | policy step {step + 1}/{len(capture_paths)}\n"
                "Direct action-query → point-key softmax attention; cyan star is the predicted position",
                fontsize=12,
            )
            writer.grab_frame()
            if step in still_indices:
                fig.savefig(still_dir / f"step_{step:03d}.png", dpi=140)
    plt.close(fig)
    return {
        **result,
        "attention_capture_start": int(capture_paths[0].stem.split("_")[-1]),
        "attention_capture_end": int(capture_paths[-1].stem.split("_")[-1]),
        "attention_video": str(video_path),
        "attention_stills": str(still_dir),
        "reconstruction_max_abs_error": max(reconstruction_errors),
        "reconstruction_min_cosine": min(reconstruction_cosines),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--task", default="stack_wine")
    parser.add_argument("--variation", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = args.run_dir / "episode_results.jsonl"
    capture_dir = args.run_dir / "attention_captures"
    results = read_jsonl(results_path)
    captures = sorted(capture_dir.glob("capture_*.npz"))
    if len(results) != args.expected_episodes:
        raise ValueError(
            f"Expected {args.expected_episodes} episode results, found {len(results)}"
        )
    if any(
        item["task"] != args.task or item["variation"] != args.variation
        for item in results
    ):
        raise ValueError(
            f"Expected only {args.task} variation {args.variation} results"
        )
    expected_captures = sum(int(item["policy_steps"]) for item in results)
    if len(captures) != expected_captures:
        raise ValueError(
            f"Expected one attention capture per policy step ({expected_captures}), "
            f"found {len(captures)}"
        )

    attention_samples = []
    corruption_rates = []
    corruption_fields: dict[int, set[bytes]] = {}
    corruption_seeds: set[int] = set()
    corruption_num_holes: set[int] = set()
    requested_missing_rates: set[float] = set()
    xyz_low = np.full(3, np.inf, dtype=np.float32)
    xyz_high = np.full(3, -np.inf, dtype=np.float32)
    for path in captures:
        frame = load_frame(path)
        attention_samples.append(frame["stage4_attention"])
        bounds_xyz = frame.get("corruption_full_xyz", frame["xyz"])
        xyz_low = np.minimum(xyz_low, bounds_xyz.min(axis=0))
        xyz_high = np.maximum(xyz_high, bounds_xyz.max(axis=0))
        if "corruption_missing_rate" in frame:
            corruption_rates.append(float(frame["corruption_missing_rate"]))
            episode_id = int(frame["corruption_episode_id"])
            field_key = (
                frame["corruption_field_centers"].tobytes()
                + frame["corruption_field_radii"].tobytes()
            )
            corruption_fields.setdefault(episode_id, set()).add(field_key)
            if "corruption_seed" in frame:
                corruption_seeds.add(int(frame["corruption_seed"]))
            if "corruption_num_holes" in frame:
                corruption_num_holes.add(int(frame["corruption_num_holes"]))
            if "corruption_requested_missing_rate" in frame:
                requested_missing_rates.add(
                    float(frame["corruption_requested_missing_rate"])
                )
    all_attention = np.concatenate(attention_samples)
    attention_low, attention_high = np.percentile(all_attention, [1.0, 99.0])
    if attention_high <= attention_low:
        attention_high = attention_low + 1e-8

    rendered_episodes = []
    cursor = 0
    for episode_index, result in enumerate(results):
        steps = int(result["policy_steps"])
        episode_captures = captures[cursor : cursor + steps]
        cursor += steps
        rendered_episodes.append(
            render_episode(
                episode_captures,
                result,
                args.checkpoint_label,
                args.run_dir,
                episode_index,
                (float(attention_low), float(attention_high)),
                (xyz_low, xyz_high),
                args.fps,
                args.task,
                args.variation,
            )
        )

    successes = sum(bool(item["success"]) for item in results)
    summary = {
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "task": args.task,
        "variation": args.variation,
        "episodes": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "success_rate": successes / len(results),
        "success_rate_wilson_95ci": wilson_interval(successes, len(results)),
        "attention_definition": (
            "Stage-4 block-2 direct action-query to point-key softmax attention, "
            "averaged over heads and non-state action queries."
        ),
        "attention_scale_percentiles": {
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "vmin": float(attention_low),
            "vmax": float(attention_high),
            "shared_within_checkpoint": True,
        },
        "attention_captures": len(captures),
        "episodes_detail": rendered_episodes,
    }
    if corruption_rates:
        if len(corruption_seeds) != 1:
            raise ValueError(f"Expected one corruption seed, got {corruption_seeds}")
        if len(corruption_num_holes) != 1:
            raise ValueError(
                f"Expected one corruption hole count, got {corruption_num_holes}"
            )
        if len(requested_missing_rates) != 1:
            raise ValueError(
                "Expected one requested corruption rate, got "
                f"{requested_missing_rates}"
            )
        summary["point_cloud_corruption"] = {
            "type": (
                "training-matched episode-consistent structured missingness "
                "with evaluation-only random seed"
            ),
            "requested_missing_rate": requested_missing_rates.pop(),
            "actual_missing_rate_min": min(corruption_rates),
            "actual_missing_rate_max": max(corruption_rates),
            "actual_missing_rate_mean": float(np.mean(corruption_rates)),
            "num_holes_per_episode": corruption_num_holes.pop(),
            "corruption_seed": corruption_seeds.pop(),
            "training_corruption_seed": 20260917,
            "seed_independent_from_training": True,
            "episodes_with_one_stable_field": sum(
                len(fields) == 1 for fields in corruption_fields.values()
            ),
            "episodes_checked": len(corruption_fields),
            "operation_order": (
                "workspace crop -> 1 cm voxel downsample -> remove exact 25% "
                "using episode-shared ellipsoid field -> max-4096 subsample -> center"
            ),
        }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in (
        "checkpoint_label", "episodes", "successes", "failures", "success_rate",
        "attention_captures",
    )}, indent=2))


if __name__ == "__main__":
    main()
