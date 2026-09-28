#!/usr/bin/env python3
"""Report progress and completed-episode QA for the full Spatial capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    summaries = sorted(args.dataset.glob("task_*/episode_*/summary.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
    frames = sum(1 for _ in args.dataset.glob("task_*/episode_*/frames/*.npz"))
    qa_frames = sum(len(record.get("frames", [])) for record in records)
    qa_passed = sum(
        bool(frame["polar"]["physics_qa"]["passed"])
        for record in records
        for frame in record.get("frames", [])
    )
    free_gib = shutil.disk_usage(args.dataset).free / 1024**3

    print(
        {
            "completed_episodes": len(records),
            "target_episodes": 500,
            "successful_episodes": sum(record.get("success") is True for record in records),
            "written_frames": frames,
            "completed_episode_frames": qa_frames,
            "polar_qa_passed_frames": qa_passed,
            "failed_marker": (args.dataset / "FAILED").exists(),
            "free_gib": round(free_gib, 2),
        }
    )


if __name__ == "__main__":
    main()
