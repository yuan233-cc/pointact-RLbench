"""Optional dense polar and numeric material inputs for PointAct."""

import io
import json
from pathlib import Path

import numpy as np


MATERIAL_FEATURE_DIM = 32  # 16 values followed by their availability masks
_MATERIAL_TYPES = ("pplastic", "dielectric", "roughconductor")


def _spectrum(value):
    if isinstance(value, dict):
        return np.interp((450, 550, 650), value["wavelengths"], value["values"]).astype(np.float32)
    if value is None:
        return np.zeros(3, dtype=np.float32)
    return np.full(3, float(value), dtype=np.float32)


def material_vector(material: dict) -> np.ndarray:
    """Encode the optical coefficients; missing coefficients have a separate mask."""
    values = np.zeros(16, dtype=np.float32)
    known = np.zeros(16, dtype=np.float32)
    material_type = material.get("type")
    if material_type in _MATERIAL_TYPES:
        values[_MATERIAL_TYPES.index(material_type)] = 1
        known[:3] = 1
    for start, key in ((3, "eta"), (6, "k")):
        if key in material:
            values[start:start + 3] = _spectrum(material[key])
            known[start:start + 3] = 1
    for index, key in ((9, "alpha"), (10, "int_ior"), (11, "ext_ior"),
                       (15, "specular_reflectance")):
        if key in material and isinstance(material[key], (int, float)):
            values[index] = float(material[key])
            known[index] = 1
    if "diffuse_reflectance" in material:
        diffuse = material["diffuse_reflectance"]
        values[12:15] = _spectrum(diffuse) if not isinstance(diffuse, list) else diffuse[:3]
        known[12:15] = 1
    return np.concatenate((values, known))


def load_material_candidates(path: str | Path, names: list[str] | None = None) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    presets = data.get("presets", data)
    selected = names if names is not None else sorted(presets)
    if not selected:
        raise ValueError("At least one candidate material is required")
    return np.stack([material_vector(presets[name]) for name in selected]).astype(np.float32)


def dense_polar_from_npz(source) -> np.ndarray:
    """Return [DoLP, cos(2AoLP), sin(2AoLP), angle-valid] at full resolution."""
    with np.load(source) as frame:
        dolp = np.asarray(frame["DoLP"], dtype=np.float32)
        if "cos2AoLP" in frame:
            cos = np.asarray(frame["cos2AoLP"], dtype=np.float32)
            sin = np.asarray(frame["sin2AoLP"], dtype=np.float32)
            valid = np.asarray(frame["valid_mask"], dtype=bool)
        else:
            angle = np.asarray(frame["AoLP"], dtype=np.float32)
            valid = np.asarray(frame["AoLP_valid_mask"], dtype=bool) & np.isfinite(angle)
            cos = np.where(valid, np.cos(2 * angle), 0)
            sin = np.where(valid, np.sin(2 * angle), 0)
        if dolp.shape != valid.shape or cos.shape != valid.shape or sin.shape != valid.shape:
            raise ValueError("Dense polar arrays must have the same height and width")
        finite = lambda array: np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        return np.stack((finite(dolp), finite(cos), finite(sin),
                         valid.astype(np.float32))).astype(np.float32)


def dense_polar_from_bytes(payload: bytes) -> np.ndarray:
    return dense_polar_from_npz(io.BytesIO(payload))
