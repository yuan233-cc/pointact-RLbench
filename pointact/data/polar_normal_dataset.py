"""Unified native-CGA and robot dataset for supervised normal pretraining."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.utils.data import Dataset


def read_polar_normal_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON, JSONL, CSV, or one-path-per-line manifests.

    Every entry may contain ``path`` and ``group``.  ``group`` should identify
    the complete object, scene, or episode used to prevent split leakage.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        content = json.loads(path.read_text())
        entries = content["samples"] if isinstance(content, dict) else content
    elif suffix == ".jsonl":
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif suffix == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows and "path" in rows[0]:
            entries = rows
        else:
            with path.open(newline="") as handle:
                entries = [
                    {"path": row[0], "group": row[1] if len(row) > 1 else None}
                    for row in csv.reader(handle)
                    if row
                ]
    else:
        entries = [
            {"path": line.strip()} for line in path.read_text().splitlines() if line.strip()
        ]
    result = []
    for raw in entries:
        entry = {"path": raw} if isinstance(raw, str) else dict(raw)
        if not entry.get("path"):
            raise ValueError(f"Manifest entry has no path: {raw!r}")
        sample_path = Path(entry["path"])
        if not sample_path.is_absolute():
            sample_path = path.parent / sample_path
        entry["path"] = str(sample_path)
        entry["group"] = str(entry.get("group") or sample_path.parent.name)
        result.append(entry)
    if not result:
        raise ValueError(f"Empty normal-data manifest: {path}")
    return result


def assert_disjoint_groups(*datasets: "PolarNormalDataset") -> None:
    seen: dict[str, int] = {}
    for split_index, dataset in enumerate(datasets):
        for group in dataset.groups:
            previous = seen.setdefault(group, split_index)
            if previous != split_index:
                raise ValueError(
                    f"Group {group!r} occurs in dataset splits {previous} and {split_index}"
                )


def _load_record(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in (".pt", ".pth"):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(record, Mapping):
            raise TypeError(f"Expected a mapping in {path}")
        return dict(record)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    raise ValueError(f"Unsupported sample file {path}; expected .pt/.pth/.npz")


def _tensor(value: Any, name: str) -> Tensor:
    value = torch.as_tensor(value).float()
    if value.ndim == 2:
        value = value[None]
    elif value.ndim == 3 and value.shape[-1] <= 16 and value.shape[0] > 16:
        value = value.permute(2, 0, 1)
    if value.ndim != 3:
        raise ValueError(f"{name} must be CHW or HWC, got {tuple(value.shape)}")
    return value.contiguous()


def _first(record: Mapping[str, Any], names: Iterable[str], required: bool = True):
    for name in names:
        if name in record:
            return record[name]
    if required:
        raise KeyError(f"Missing all equivalent keys {tuple(names)}")
    return None


def _scalar_text(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected scalar text metadata, got shape {value.shape}")
        value = value.reshape(()).item()
    return str(value)


class PolarNormalDataset(Dataset):
    """Load paired RGB, polar observation, physical prior, and normal GT."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        input_mode: str = "native_cga",
        image_size: int | tuple[int, int] = 256,
        require_rgb: bool = True,
        require_calibration: bool = True,
        normal_gt_source: str | None = None,
        normal_transform: list[list[float]] | Tensor | None = None,
        normal_sign: float = 1.0,
        ray_dropout_prob: float = 0.0,
        limit: int | None = None,
    ):
        if input_mode not in ("native_cga", "robot"):
            raise ValueError("input_mode must be 'native_cga' or 'robot'")
        self.entries = read_polar_normal_manifest(manifest)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            self.entries = self.entries[:limit]
        self.input_mode = input_mode
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.require_rgb = require_rgb
        self.require_calibration = require_calibration
        if normal_gt_source is not None:
            normalized_source = normal_gt_source.lower().replace("_", " ")
            forbidden = ("damaged depth", "corrupted depth", "observed depth", "candidate normal")
            if any(term in normalized_source for term in forbidden):
                raise ValueError(
                    "normal_gt_source identifies an invalid supervision source; use independent "
                    "measured/public GT or mesh-rendered visible-surface normals"
                )
        self.normal_gt_source = normal_gt_source
        self.normal_transform = (
            torch.eye(3) if normal_transform is None else torch.as_tensor(normal_transform).float()
        )
        if self.normal_transform.shape != (3, 3):
            raise ValueError("normal_transform must be 3x3")
        self.normal_sign = float(normal_sign)
        self.ray_dropout_prob = float(ray_dropout_prob)
        if not 0.0 <= self.ray_dropout_prob <= 1.0:
            raise ValueError("ray_dropout_prob must be between 0 and 1")

    @property
    def groups(self) -> set[str]:
        return {entry["group"] for entry in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    def _branches(self, record: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        direct_observation = _first(
            record, ("polar_observation", "observation"), required=False
        )
        if direct_observation is not None:
            observation = _tensor(direct_observation, "polar_observation")
        elif self.input_mode == "native_cga":
            observation = torch.cat(
                tuple(
                    _tensor(_first(record, names), names[0])
                    for names in (
                        ("images",),
                        ("Iun", "I_un"),
                        ("cos1",),
                        ("cos2",),
                        ("DoP", "DoLP"),
                        ("image_coordinate", "viewing_direction"),
                    )
                ),
                dim=0,
            )
        else:
            observation = torch.cat(
                tuple(
                    _tensor(_first(record, names), names[0])
                    for names in (
                        ("Iun", "I_un"),
                        ("DoP", "DoLP"),
                        ("cos1",),
                        ("cos2",),
                        ("image_coordinate", "viewing_direction"),
                    )
                ),
                dim=0,
            )

        direct_prior = _first(record, ("physical_prior",), required=False)
        if direct_prior is not None:
            physical_prior = _tensor(direct_prior, "physical_prior")
        else:
            # Original CGA layout: three ambiguous normals (9), intensity (1),
            # and specular confidence (1).  This path consumes existing physical
            # estimates; it never derives them from the observation tensor.
            physical_prior = torch.cat(
                (
                    _tensor(_first(record, ("est", "candidate_normals")), "est"),
                    _tensor(_first(record, ("Iun", "I_un")), "Iun"),
                    _tensor(_first(record, ("spec", "specular_confidence")), "spec"),
                ),
                dim=0,
            )
        expected_observation = 11 if self.input_mode == "native_cga" else 7
        if observation.shape[0] != expected_observation:
            raise ValueError(
                f"{self.input_mode} observation must have {expected_observation} channels, "
                f"got {observation.shape[0]}"
            )
        if physical_prior.shape[0] != 11:
            raise ValueError(f"Physical prior must have 11 channels, got {physical_prior.shape[0]}")
        if observation.shape[-2:] != physical_prior.shape[-2:]:
            raise ValueError("Observation and physical prior are not spatially aligned")
        if not torch.isfinite(observation).all() or not torch.isfinite(physical_prior).all():
            raise ValueError("Observation and physical prior must contain only finite values")
        return observation, physical_prior

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        path = Path(entry["path"])
        record = _load_record(path)
        return self._sample_from_record(record, entry, path)

    def _sample_from_record(
        self, record: Mapping[str, Any], entry: Mapping[str, Any], path: Path
    ) -> dict[str, Any]:
        observation, physical_prior = self._branches(record)
        rgb_value = _first(record, ("rgb", "RGB", "color"), required=self.require_rgb)
        if rgb_value is None:
            rgb = torch.empty(0, *observation.shape[-2:])
        else:
            rgb = _tensor(rgb_value, "rgb")
            if rgb.shape[0] != 3:
                raise ValueError(f"RGB must have three channels, got {rgb.shape[0]}")
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            rgb = rgb.clamp(0.0, 1.0)
        normal = _tensor(
            _first(record, ("normal_gt", "label", "normal")), "normal_gt"
        )
        if normal.shape[0] != 3:
            raise ValueError(f"normal_gt must have three channels, got {normal.shape[0]}")
        mask = _tensor(
            _first(record, ("normal_valid_mask", "mask", "valid_mask")),
            "normal_valid_mask",
        )[:1]
        spatial = observation.shape[-2:]
        if any(tensor.numel() and tensor.shape[-2:] != spatial for tensor in (rgb, normal, mask)):
            raise ValueError(f"RGB/polar/prior/normal are not aligned in {path}")

        calibration = _first(record, ("K", "intrinsics", "camera_K"), required=False)
        has_coordinate = any(
            key in record for key in ("image_coordinate", "viewing_direction", "normal_coordinate_frame")
        )
        if self.require_calibration and calibration is None and not has_coordinate:
            raise ValueError(f"Sample has no camera calibration/coordinate information: {path}")

        original_height, original_width = spatial
        size = self.image_size
        observation = F.interpolate(observation[None], size=size, mode="bilinear", align_corners=False)[0]
        physical_prior = F.interpolate(physical_prior[None], size=size, mode="bilinear", align_corners=False)[0]
        observation[-3:] = F.normalize(observation[-3:], dim=0, eps=1e-6)
        if self.ray_dropout_prob == 1.0 or (
            self.ray_dropout_prob > 0.0 and torch.rand(()).item() < self.ray_dropout_prob
        ):
            observation[-3:] = 0.0
        candidate_normals = physical_prior[:9].reshape(3, 3, *size)
        physical_prior[:9] = F.normalize(candidate_normals, dim=1, eps=1e-6).reshape(9, *size)
        if rgb.numel():
            rgb = F.interpolate(rgb[None], size=size, mode="bilinear", align_corners=False)[0]
        normal = F.interpolate(normal[None], size=size, mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask[None], size=size, mode="nearest")[0] > 0.5
        normal = torch.einsum("ij,jhw->ihw", self.normal_transform, normal) * self.normal_sign
        finite = torch.isfinite(normal).all(dim=0, keepdim=True)
        nonzero = torch.linalg.vector_norm(torch.nan_to_num(normal), dim=0, keepdim=True) > 1e-6
        mask = mask & finite & nonzero
        normal = F.normalize(torch.nan_to_num(normal), dim=0, eps=1e-6)
        sample: dict[str, Any] = {
            "rgb": rgb,
            "polar_observation": observation,
            "physical_prior": physical_prior,
            "normal_gt": normal,
            "normal_valid_mask": mask,
            "sample_id": str(entry.get("id") or path.stem),
            "group_id": entry["group"],
            "dataset_id": _scalar_text(
                entry.get("dataset") or _first(record, ("dataset_id",), required=False),
                "unspecified",
            ),
            "normal_gt_source": self.normal_gt_source or str(entry.get("normal_gt_source", "unspecified")),
        }
        if calibration is not None:
            camera_k = torch.as_tensor(calibration).float().clone()
            if camera_k.shape != (3, 3):
                raise ValueError(f"Camera intrinsics must be 3x3, got {tuple(camera_k.shape)}")
            camera_k[0] *= size[1] / original_width
            camera_k[1] *= size[0] / original_height
            sample["camera_K"] = camera_k
        transform = _first(record, ("T_camera_from_world",), required=False)
        if transform is not None:
            sample["T_camera_from_world"] = torch.as_tensor(transform).float()
        return sample
