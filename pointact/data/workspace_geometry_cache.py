"""Immutable, uncompressed column cache for calibrated geometry + frozen teachers."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .observed_workspace_mask import operation_workspace

CACHE_VERSION = "observed_neighbors_v1_frozen_teacher_fp32_v1"


class PrefetchedEpochBatches:
    """Keep workers prefetching across epochs, retaining each partial last batch."""
    def __init__(self, frames, batch_size, epochs, seed=42):
        if min(frames, batch_size, epochs) < 1:
            raise ValueError("Positive frame, batch and epoch counts required")
        self.frames, self.batch_size, self.epochs, self.seed = frames, batch_size, epochs, seed
        self.steps_per_epoch = math.ceil(frames / batch_size)

    def __len__(self):
        return self.steps_per_epoch * self.epochs

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            indices = torch.randperm(self.frames, generator=generator).tolist()
            for start in range(0, self.frames, self.batch_size):
                yield indices[start:start+self.batch_size]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocessing_signature():
    paths = [Path(__file__).with_name(name) for name in
             ("workspace_geometry_dataset.py", "observed_workspace_mask.py", "rlbench_polar_normal_lmdb.py")]
    model_root = Path(__file__).parents[1] / "model" / "vla_pointact"
    paths += [model_root / "workspace_geometry.py",
              model_root / "action_head_3d" / "polarapp_tasknet_encoder.py",
              model_root / "action_head_3d" / "cga_dino_normal.py"]
    return {p.name: file_sha256(p) for p in paths}


class GeometryColumnWriter:
    """Write each frame once. Manifest publication marks the cache complete."""
    def __init__(self, root, keys, backbone, teacher_checkpoint, point_capacity=16384):
        self.root = Path(root)
        self.root.mkdir(exist_ok=False, parents=True)
        self.keys = [k.decode() if isinstance(k, bytes) else str(k) for k in keys]
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Duplicate cache frame IDs")
        self.backbone = backbone
        self.checkpoint_sha = file_sha256(teacher_checkpoint)
        self.capacity = point_capacity
        self.columns, self.rows = {}, 0

    def append(self, sample):
        n = len(sample["points"])
        if n > self.capacity:
            raise ValueError(f"Cache point capacity {self.capacity} < {n}; never truncate observations")
        arrays = {k: v.detach().cpu().numpy() for k, v in sample.items()}
        arrays["point_count"] = np.asarray(n, np.int32)
        for name, value in arrays.items():
            if name not in self.columns:
                shape = ((self.capacity,) + value.shape[1:] if name in
                         ("points", "point_pixel_indices") else value.shape)
                self.columns[name] = np.lib.format.open_memmap(
                    self.root / f"{name}.npy", mode="w+", dtype=value.dtype,
                    shape=(len(self.keys), *shape))
            dest = self.columns[name][self.rows]
            if name in ("points", "point_pixel_indices"):
                dest[:n] = value
            else:
                self.columns[name][self.rows] = value
        self.rows += 1

    def finish(self):
        if self.rows != len(self.keys):
            raise ValueError("Incomplete cache cannot be published")
        for col in self.columns.values():
            col.flush()
        metadata = dict(version=CACHE_VERSION, backbone=self.backbone,
            teacher_sha256=self.checkpoint_sha, preprocessing=preprocessing_signature(),
            workspace_bounds=operation_workspace().tolist(), keys=self.keys,
            columns={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in self.columns.items()},
            augmentation="none", precision="float32_outputs_from_bf16_teacher")
        partial = self.root / "manifest.partial.json"
        partial.write_text(json.dumps(metadata, indent=2))
        partial.replace(self.root / "manifest.json")


class CachedWorkspaceGeometryDataset(Dataset):
    def __init__(self, root, backbone, teacher_checkpoint, split="train", max_points=4096,
                 preload=True, verify=True):
        self.root = Path(root)
        self.meta = json.loads((self.root / "manifest.json").read_text())
        if self.meta["version"] != CACHE_VERSION or self.meta["backbone"] != backbone:
            raise ValueError("Cache format/backbone mismatch")
        if not np.array_equal(np.asarray(self.meta["workspace_bounds"], np.float32), operation_workspace()):
            raise ValueError("Cache operation workspace mismatch")
        if verify and (self.meta["teacher_sha256"] != file_sha256(teacher_checkpoint)
                       or self.meta["preprocessing"] != preprocessing_signature()):
            raise ValueError("Stale cache: checkpoint or preprocessing changed")
        self.indices = np.asarray([i for i, k in enumerate(self.meta["keys"])
            if (int(k.split("-")[0]) % 10 == 9) == (split == "val")])
        self.keys = [self.meta["keys"][i].encode() for i in self.indices]
        self.split, self.max_points = split, max_points
        self.columns = {}
        for name, spec in self.meta["columns"].items():
            column = np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if list(column.shape) != spec["shape"] or str(column.dtype) != spec["dtype"]:
                raise ValueError(f"Malformed cache column {name}")
            # Load only this split; contiguous RAM tensors are inherited read-only by workers.
            self.columns[name] = column[self.indices].copy() if preload else column
        self.preload = preload
        print(f"CACHE_READY split={split} frames={len(self)} preload={preload} "
              f"ram_gb={sum(v.nbytes for v in self.columns.values()) / 2**30:.3f}", flush=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = index if self.preload else int(self.indices[index])
        n = int(self.columns["point_count"][row])
        rng = np.random.default_rng(index) if self.split == "val" else np.random
        sampled = rng.choice(n, min(n, self.max_points), replace=False)
        out = {}
        for name, column in self.columns.items():
            if name == "point_count":
                continue
            value = column[row]
            if name in ("points", "point_pixel_indices"):
                value = value[sampled]
            out[name] = torch.from_numpy(np.array(value, copy=True))
        return out
