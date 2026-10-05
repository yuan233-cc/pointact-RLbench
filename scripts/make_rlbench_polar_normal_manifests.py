"""Create episode-disjoint train/validation manifests for RLBench v2 normals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--val-every", type=int, default=10, help="One in N episodes per task in validation")
    args = parser.parse_args()
    if args.val_every < 2:
        parser.error("--val-every must be at least 2")
    episode_file = args.dataset_root / "meta/episodes.jsonl"
    episodes = [json.loads(line) for line in episode_file.read_text().splitlines() if line.strip()]
    task_counts: dict[str, int] = {}
    manifests: dict[str, list[dict]] = {"train": [], "val": []}
    for episode in episodes:
        index = int(episode["episode_index"])
        task = str(episode["tasks"][0]).split("<br>")[0]
        ordinal = task_counts.get(task, 0)
        task_counts[task] = ordinal + 1
        split = "val" if ordinal % args.val_every == 0 else "train"
        for frame in range(int(episode["length"])):
            manifests[split].append({
                "episode_index": index,
                "frame_index": frame,
                "group": f"rlbench/{task}/episode_{index:06d}",
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, entries in manifests.items():
        target = args.output_dir / f"{split}.jsonl"
        target.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        print(f"{target}: {len(entries)} frames")
    print(f"{len(episodes)} episodes across {len(task_counts)} tasks")


if __name__ == "__main__":
    main()
