"""Utilities for capturing and rendering PointACT PTV3 point features.

This module is intentionally independent from the model implementation.  The
inference server registers a normal PyTorch forward hook and writes one NPZ per
policy request; the renderer can then be run offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def safe_slug(value: str, max_length: int = 80) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_")
    return (value or "unknown")[:max_length]


def split_by_offsets(array: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    starts = np.r_[0, offsets[:-1]].astype(np.int64)
    return [array[start:end] for start, end in zip(starts, offsets)]


@dataclass
class FeatureCapture:
    output_dir: Path
    capture_every: int = 1
    max_captures: int = 0
    save_input_points: bool = True

    def __post_init__(self) -> None:
        if self.capture_every <= 0:
            raise ValueError("capture_every must be positive")
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if next(self.output_dir.glob("capture_*.npz"), None) is not None:
            raise FileExistsError(
                f"Capture directory is not empty; refusing to overwrite: {self.output_dir}"
            )
        self.request_index = -1
        self.capture_index = 0
        self.metadata: dict = {}
        self.pending_arrays: dict[str, np.ndarray] | None = None

    @property
    def enabled_for_request(self) -> bool:
        if self.request_index < 0 or self.request_index % self.capture_every != 0:
            return False
        return self.max_captures <= 0 or self.capture_index < self.max_captures

    def begin_request(self, batch: dict) -> None:
        self.request_index += 1
        self.pending_arrays = None
        tasks = batch.get("task", ["unknown"])
        instruction = str(tasks[0]) if tasks else "unknown"
        self.metadata = {
            "request_index": self.request_index,
            "instruction": instruction,
        }

    def ptv3_pre_hook(self, _module, inputs) -> None:
        """Capture the unpooled point order before the PTV3 wrapper runs."""
        if not self.enabled_for_request:
            return
        input_points, input_counts = inputs[0], inputs[1]
        input_np = input_points.detach().float().cpu().numpy()
        self.pending_arrays = {
            "stage0_coordinates": input_np[:, :3],
            "input_coordinates": input_np[:, :3],
            "input_offsets": np.cumsum(input_counts.detach().cpu().numpy()),
        }
        if self.save_input_points and input_np.shape[1] >= 6:
            self.pending_arrays["input_rgb"] = input_np[:, 3:6]

    def make_pooling_hook(self, input_stage: int):
        """Return a hook recording the exact parent-to-child voxel assignment."""
        def pooling_hook(_module, _inputs, output) -> None:
            if self.pending_arrays is None:
                return
            output_stage = input_stage + 1
            inverse = output.pooling_inverse.detach().cpu().numpy().astype(np.int64)
            self.pending_arrays[
                f"pooling_inverse_stage{input_stage}_to_stage{output_stage}"
            ] = inverse
            self.pending_arrays[f"stage{output_stage}_coordinates"] = (
                output.coord.detach().float().cpu().numpy()
            )
            self.pending_arrays[f"stage{output_stage}_offsets"] = (
                output.offset.detach().cpu().numpy()
            )

        return pooling_hook

    def ptv3_hook(self, _module, inputs, output) -> None:
        if not self.enabled_for_request:
            return

        input_points, input_counts = inputs[0], inputs[1]
        features, coordinates, offsets, action_features = output[:4]
        input_offsets = np.cumsum(input_counts.detach().cpu().numpy())

        input_np = input_points.detach().float().cpu().numpy()
        if self.pending_arrays is None:
            self.pending_arrays = {}
        self.pending_arrays.update({
            "coordinates": coordinates.detach().float().cpu().numpy(),
            "features": features.detach().to(dtype=torch.float16).cpu().numpy(),
            "offsets": offsets.detach().cpu().numpy(),
            "input_offsets": input_offsets,
            "action_features": action_features.detach().to(dtype=torch.float16).cpu().numpy(),
        })
        if self.save_input_points:
            self.pending_arrays["input_coordinates"] = input_np[:, :3]
            if input_np.shape[1] >= 6:
                self.pending_arrays["input_rgb"] = input_np[:, 3:6]

    def action_head_hook(self, _module, _inputs, output) -> None:
        if self.pending_arrays is None:
            return

        # xt: (action_steps, xyz_axes, points, offset_bins). The action head
        # applies softmax jointly across points and bins for each xyz axis.
        # Summing the bin dimension produces the probability mass assigned to
        # each PTV3 point as a position anchor.
        position_logits = output[0].detach().float()
        first_step = position_logits[0]
        axis_probability = torch.softmax(first_step.reshape(3, -1), dim=-1).reshape_as(first_step)
        axis_point_probability = axis_probability.sum(dim=-1)
        self.pending_arrays["action_point_probability_xyz"] = (
            axis_point_probability.cpu().numpy()
        )
        self.pending_arrays["action_point_probability"] = (
            axis_point_probability.mean(dim=0).cpu().numpy()
        )
        predicted_actions = output[3]
        if predicted_actions is not None:
            self.pending_arrays["predicted_position"] = (
                predicted_actions[0, 0, :3].detach().float().cpu().numpy()
            )

        self._write_pending_capture()

    def _write_pending_capture(self) -> None:
        arrays = self.pending_arrays
        if arrays is None:
            return

        # Compose the exact cluster maps to preserve real pooling ancestry.
        input_to_stage = np.arange(int(arrays["input_offsets"][-1]), dtype=np.int64)
        output_stage = 1
        while True:
            key = f"pooling_inverse_stage{output_stage - 1}_to_stage{output_stage}"
            if key not in arrays:
                break
            input_to_stage = arrays[key][input_to_stage]
            arrays[f"input_to_stage{output_stage}"] = input_to_stage.copy()
            output_stage += 1
        if output_stage > 1:
            arrays["input_to_final"] = input_to_stage.copy()
            arrays["num_pooling_stages"] = np.asarray(output_stage - 1, dtype=np.int64)

        stem = f"capture_{self.capture_index:06d}"
        target = self.output_dir / f"{stem}.npz"
        np.savez_compressed(target, **arrays)

        record = {
            **self.metadata,
            "capture_index": self.capture_index,
            "file": target.name,
            "input_points": int(arrays["input_offsets"][-1]),
            "feature_points": int(arrays["coordinates"].shape[0]),
            "feature_dim": int(arrays["features"].shape[1]),
            "pooling_stages": int(arrays.get("num_pooling_stages", 0)),
        }
        with (self.output_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.capture_index += 1
        self.pending_arrays = None


def discover_captures(paths: Iterable[str | Path]) -> list[Path]:
    captures: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix == ".npz":
            captures.append(path)
        elif path.is_dir():
            captures.extend(path.rglob("capture_*.npz"))
    return sorted(set(captures))


def load_manifest_record(capture: Path) -> dict:
    manifest = capture.parent / "manifest.jsonl"
    if not manifest.exists():
        return {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("file") == capture.name:
                return record
    return {}


def fit_global_pca(captures: list[Path], max_samples: int, seed: int):
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    per_file = max(3, max_samples // max(1, len(captures)))
    samples = []
    for capture in captures:
        with np.load(capture) as data:
            feature = data["features"].astype(np.float32)
        if len(feature) > per_file:
            feature = feature[rng.choice(len(feature), per_file, replace=False)]
        samples.append(feature)
    matrix = np.concatenate(samples, axis=0)
    if len(matrix) > max_samples:
        matrix = matrix[rng.choice(len(matrix), max_samples, replace=False)]
    pca = PCA(n_components=3, svd_solver="randomized", random_state=seed)
    transformed = pca.fit_transform(matrix)

    # PCA signs are arbitrary. Make the largest-magnitude loading positive so
    # repeated runs have stable colors.
    dominant = np.abs(pca.components_).argmax(axis=1)
    signs = np.sign(pca.components_[np.arange(3), dominant])
    signs[signs == 0] = 1
    pca.components_ *= signs[:, None]
    transformed *= signs[None, :]
    low = np.percentile(transformed, 1, axis=0)
    high = np.percentile(transformed, 99, axis=0)
    return pca, low, high


def normalize_rgb(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    scale = np.maximum(high - low, 1e-8)
    return np.clip((values - low) / scale, 0.0, 1.0)


def normalize_input_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    if rgb.size and (rgb.max() > 1.5 or rgb.min() < -0.5):
        if rgb.min() >= 0:
            rgb = rgb / 255.0
        else:
            low, high = np.percentile(rgb, [1, 99], axis=0)
            rgb = normalize_rgb(rgb, low, high)
    return np.clip(rgb, 0.0, 1.0)


def nearest_reference_indices(query_xyz: np.ndarray, reference_xyz: np.ndarray) -> np.ndarray:
    """Map every dense input point to its nearest final PTV3 anchor."""
    if len(reference_xyz) == 0:
        raise ValueError("Cannot propagate features from an empty PTV3 point set")
    output = []
    for start in range(0, len(query_xyz), 2048):
        query = query_xyz[start : start + 2048]
        squared_distance = np.sum(
            (query[:, None, :] - reference_xyz[None, :, :]) ** 2,
            axis=-1,
        )
        output.append(np.argmin(squared_distance, axis=1))
    return np.concatenate(output)


def set_axes_equal(ax, xyz: np.ndarray) -> None:
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    radius = max(float(np.ptp(xyz, axis=0).max()) / 2, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    colors = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        np.savetxt(handle, np.c_[xyz, colors], fmt="%.7g %.7g %.7g %d %d %d")


def render_capture(capture: Path, output_dir: Path, pca, low, high, dpi: int) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(capture) as data:
        xyz = data["coordinates"].astype(np.float32)
        feature = data["features"].astype(np.float32)
        input_xyz = data.get("input_coordinates", xyz).astype(np.float32)
        input_rgb = normalize_input_rgb(data.get("input_rgb", np.full_like(input_xyz, 0.6)))
        action_probability = data.get(
            "action_point_probability", np.full(len(xyz), 1.0 / max(len(xyz), 1))
        ).astype(np.float32)
        predicted_position = data.get("predicted_position")
        if predicted_position is not None:
            predicted_position = predicted_position.astype(np.float32)

    projected = pca.transform(feature)
    pca_rgb = normalize_rgb(projected, low, high)
    nearest = nearest_reference_indices(input_xyz, xyz)
    dense_pca_rgb = pca_rgb[nearest]
    dense_action_probability = action_probability[nearest]
    relevance_low, relevance_high = np.percentile(action_probability, [1, 99])
    relevance_scale = max(float(relevance_high - relevance_low), 1e-12)
    dense_relevance_normalized = np.clip(
        (dense_action_probability - relevance_low) / relevance_scale, 0, 1
    )
    relevance_rgb = plt.get_cmap("magma")(dense_relevance_normalized)[:, :3]
    record = load_manifest_record(capture)

    fig = plt.figure(figsize=(14, 12))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.89, wspace=0.02, hspace=0.28)
    panels = [
        (input_xyz, input_rgb, "Input point cloud (RGB)", 35, 35),
        (
            input_xyz,
            dense_pca_rgb,
            "Final PTV3 feature PCA (propagated to input points)",
            35,
            35,
        ),
        (
            input_xyz,
            relevance_rgb,
            "Action-head position probability (propagated)",
            35,
            35,
        ),
        (xyz, pca_rgb, "Actual final PTV3 encoder points", 35, 35),
    ]
    for index, (coords, colors, title, elev, azim) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        point_size = 2 if len(coords) > 500 else 22
        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=colors, s=point_size, linewidths=0,
        )
        if index in (3, 4) and predicted_position is not None:
            ax.scatter(
                predicted_position[0], predicted_position[1], predicted_position[2],
                marker="*", s=180, c="white", edgecolors="black", linewidths=1.2,
                label="predicted action position",
            )
            ax.legend(loc="upper left", fontsize=8)
        if index == 3:
            probability_norm = matplotlib.colors.Normalize(
                vmin=float(relevance_low), vmax=float(relevance_low + relevance_scale)
            )
            colorbar_source = matplotlib.cm.ScalarMappable(
                norm=probability_norm, cmap="magma"
            )
            fig.colorbar(
                colorbar_source, ax=ax, shrink=0.55, pad=0.08,
                label="anchor probability mass",
            )
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        set_axes_equal(ax, coords)
        ax.set_title(title, pad=12)
    instruction = record.get("instruction", "")
    fig.suptitle(
        f"{capture.parent.name} / {capture.stem}  |  "
        f"{len(input_xyz)} input points → {len(xyz)} PTV3 points × {feature.shape[1]} dims\n"
        f"{instruction}",
        fontsize=11,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{capture.parent.name}_{capture.stem}.png"
    ply_path = output_dir / f"{capture.parent.name}_{capture.stem}.ply"
    sparse_ply_path = output_dir / f"{capture.parent.name}_{capture.stem}_sparse.ply"
    relevance_ply_path = output_dir / f"{capture.parent.name}_{capture.stem}_action_relevance.ply"
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    write_ply(ply_path, input_xyz, dense_pca_rgb)
    write_ply(sparse_ply_path, xyz, pca_rgb)
    write_ply(relevance_ply_path, input_xyz, relevance_rgb)
    return {
        "capture": str(capture),
        "png": str(png_path),
        "ply": str(ply_path),
        "sparse_ply": str(sparse_ply_path),
        "action_relevance_ply": str(relevance_ply_path),
        "display_points": len(input_xyz),
        "feature_points": len(xyz),
        "feature_dim": feature.shape[1],
        "feature_propagation": "nearest_final_ptv3_point",
        "action_probability_min": float(action_probability.min()),
        "action_probability_max": float(action_probability.max()),
        "instruction": instruction,
    }
