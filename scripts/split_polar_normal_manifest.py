#!/usr/bin/env python3
"""Create deterministic, group-disjoint train/validation manifests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be strictly between zero and one")
    if args.train_output.exists() or args.val_output.exists():
        parser.error("refusing to overwrite an output manifest")

    content = json.loads(args.manifest.read_text())
    samples = content["samples"] if isinstance(content, dict) else content
    groups = sorted({str(sample.get("group") or "") for sample in samples})
    if "" in groups:
        parser.error("every sample must have a non-empty group")
    if len(groups) < 2:
        parser.error("at least two independent groups are required for a split")
    random.Random(args.seed).shuffle(groups)
    val_count = min(len(groups) - 1, max(1, round(len(groups) * args.val_fraction)))
    val_groups = set(groups[:val_count])
    train = [sample for sample in samples if str(sample["group"]) not in val_groups]
    val = [sample for sample in samples if str(sample["group"]) in val_groups]
    if not train or not val:
        raise RuntimeError("group split unexpectedly produced an empty partition")
    args.train_output.write_text(json.dumps({"samples": train}, indent=2) + "\n")
    args.val_output.write_text(json.dumps({"samples": val}, indent=2) + "\n")
    print(
        json.dumps(
            {
                "train_samples": len(train),
                "val_samples": len(val),
                "train_groups": len(set(groups) - val_groups),
                "val_groups": len(val_groups),
                "val_group_names": sorted(val_groups),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
