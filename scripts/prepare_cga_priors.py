#!/usr/bin/env python3
"""Pack physically computed candidate normals into CGA 11-channel priors.

This utility intentionally does not estimate candidates from depth or learn an
adapter.  Its inputs must already contain three SfP candidate normals from a
polarization physics pipeline (``est``/``candidate_normals``), intensity, and
specular confidence.  It validates and normalizes the candidates, then writes
portable NPZ records consumed by ``PolarNormalDataset``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointact.data.polar_normal_dataset import read_polar_normal_manifest  # noqa: E402


def load_record(path: Path) -> dict:
    if path.suffix.lower() in (".pt", ".pth"):
        return dict(torch.load(path, map_location="cpu", weights_only=False))
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def chw(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 2:
        value = value[None]
    elif value.ndim == 3 and value.shape[-1] <= 16 and value.shape[0] > 16:
        value = value.transpose(2, 0, 1)
    if value.ndim != 3:
        raise ValueError(f"{name} is not CHW/HWC: {value.shape}")
    return value


def pick(record: dict, *names: str):
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"Missing required physics result; expected one of {names}")


def build_prior(record: dict) -> np.ndarray:
    candidates = chw(pick(record, "est", "candidate_normals"), "candidate_normals")
    if candidates.shape[0] != 9:
        raise ValueError(f"Expected three candidate normals (9 channels), got {candidates.shape}")
    candidates = candidates.reshape(3, 3, *candidates.shape[-2:])
    norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    valid = np.isfinite(candidates).all(axis=1, keepdims=True) & (norms > 1e-6)
    candidates = np.where(valid, candidates / np.maximum(norms, 1e-6), 0.0)
    candidates = candidates.reshape(9, *candidates.shape[-2:])
    intensity = chw(pick(record, "Iun", "I_un", "intensity"), "intensity")[:1]
    confidence = chw(
        pick(record, "spec", "specular_confidence"), "specular_confidence"
    )[:1]
    if candidates.shape[-2:] != intensity.shape[-2:] or intensity.shape != confidence.shape:
        raise ValueError("Candidates, intensity, and specular confidence are not aligned")
    if not np.isfinite(intensity).all() or not np.isfinite(confidence).all():
        raise ValueError("Intensity and specular confidence must be finite")
    if confidence.min() < 0 or confidence.max() > 1:
        raise ValueError("Specular confidence must be scaled to [0,1]")
    return np.concatenate((candidates, intensity, confidence), axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    entries = read_polar_normal_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    output_entries = []
    for index, entry in enumerate(entries):
        source = Path(entry["path"])
        destination = args.output / f"{index:08d}_{source.stem}.npz"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {destination}")
        record = load_record(source)
        record["physical_prior"] = build_prior(record)
        arrays = {
            key: np.asarray(value)
            for key, value in record.items()
            if isinstance(value, (np.ndarray, torch.Tensor, int, float, bool))
        }
        np.savez_compressed(destination, **arrays)
        output_entries.append(
            {"path": destination.name, "group": entry["group"], "id": entry.get("id", source.stem)}
        )
    (args.output / "manifest.json").write_text(
        json.dumps({"samples": output_entries}, indent=2) + "\n"
    )
    print(f"Wrote {len(output_entries)} physical-prior samples to {args.output}")


if __name__ == "__main__":
    main()
