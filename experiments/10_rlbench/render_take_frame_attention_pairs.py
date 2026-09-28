"""Render paired initial attention maps for representative take-frame episodes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


CHECKPOINTS = ("old_complete", "new_incomplete25")


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def frame_outline(geometry: dict) -> np.ndarray:
    pose = np.asarray(geometry["pose_world"])
    bbox = np.asarray(geometry["bbox_local"])
    corners = np.asarray(
        [[x, y, z] for x in bbox[:2] for y in bbox[2:4] for z in bbox[4:6]]
    )
    world = Rotation.from_quat(pose[3:7]).apply(corners) + pose[:3]
    projected = world[:, 1:3]
    hull = ConvexHull(projected)
    return projected[np.r_[hull.vertices, hull.vertices[0]]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 4])
    args = parser.parse_args()

    captures = {}
    geometry = {}
    outcomes = {}
    for checkpoint in CHECKPOINTS:
        base = args.root / checkpoint
        captures[checkpoint] = {
            int(row["episode_id"]): base / "attention" / row["file"]
            for row in jsonl(base / "attention" / "manifest.jsonl")
            if int(row["request_in_episode"]) == 0
        }
        geometry[checkpoint] = {
            int(row["episode"]): row["before"]["frame"]
            for row in jsonl(base / "eval" / "take_frame_geometry_steps.jsonl")
            if int(row["step"]) == 0
        }
        outcomes[checkpoint] = {
            int(row["episode"]): row["failure_category"]
            for row in jsonl(base / "eval" / "take_frame_geometry_episodes.jsonl")
        }

    for episode in args.episodes:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
        for column, checkpoint in enumerate(CHECKPOINTS):
            with np.load(captures[checkpoint][episode]) as data:
                center = np.asarray(data["scene_center"]).reshape(-1, 3)[0]
                stage = max(
                    int(match.group(1))
                    for key in data.files
                    if (match := re.fullmatch(
                        r"action_attention_stage(\d+)_block\d+_point_coordinates", key
                    ))
                )
                blocks = [
                    int(match.group(1))
                    for key in data.files
                    if (match := re.fullmatch(
                        rf"action_attention_stage{stage}_block(\d+)_point_coordinates", key
                    ))
                ]
                prefix = f"action_attention_stage{stage}_block{max(blocks)}"
                points = np.asarray(data[f"{prefix}_point_coordinates"]) + center
                dense = np.asarray(data["input_coordinates"]) + center
                arrays = {
                    "action": np.asarray(data[f"{prefix}_point_weights"]),
                    "state": np.asarray(data[f"{prefix}_state_point_weights"]),
                }
                outline = frame_outline(geometry[checkpoint][episode])
                for row, query in enumerate(("action", "state")):
                    ax = axes[row, column]
                    weights = arrays[query]
                    scaled = weights / max(float(weights.max()), 1e-12)
                    ax.scatter(dense[:, 1], dense[:, 2], s=1, c="0.83", alpha=0.25)
                    image = ax.scatter(
                        points[:, 1], points[:, 2], s=18 + 150 * scaled,
                        c=scaled, cmap="turbo", vmin=0, vmax=1,
                        edgecolors="black", linewidths=0.25,
                    )
                    ax.plot(outline[:, 0], outline[:, 1], color="magenta", linewidth=2)
                    ax.set_title(
                        f"{checkpoint} · {query}\n{outcomes[checkpoint][episode]}"
                    )
                    ax.set_xlabel("world y (m)")
                    ax.set_ylabel("world z (m)")
                    ax.set_aspect("equal", adjustable="box")
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="weight / panel max")
        fig.suptitle(
            f"Episode {episode}, request 0, final PTV3 stage; magenta = frame OBB",
            fontsize=13,
        )
        fig.subplots_adjust(left=0.08, right=0.89, bottom=0.07, top=0.9, wspace=0.18, hspace=0.22)
        fig.savefig(args.root / f"attention_pair_episode_{episode:02d}.png", dpi=190)
        plt.close(fig)


if __name__ == "__main__":
    main()
