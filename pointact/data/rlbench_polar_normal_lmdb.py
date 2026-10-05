"""Scene-level RLBench v2 LMDB adapter for CGA normal pretraining.

The v2 rerender stores corrected DoLP/AoLP, but not corrected four-analyzer
intensities. Its I_un sidecar is an RGB-luminance proxy. Priors generated here
are therefore *approximate* physical priors, never ground-truth normals.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import av
import lmdb
import numpy as np
from scipy.ndimage import minimum_filter

from pointact.data.polar_normal_dataset import PolarNormalDataset
from pointact.data.polar_normal_sources import ambiguous_normals

_SHARED_ENVS: dict[str, lmdb.Environment] = {}


def generate_rlbench_cga_input(
    polar: dict[str, np.ndarray], sfp: dict[str, np.ndarray], *,
    refractive_index: float = 1.5, input_mode: str = "robot",
) -> tuple[np.ndarray, np.ndarray]:
    """Build a robot (7 CHW) or native-CGA (11 CHW) input without GT leakage.

    Both layouts use the canonical pretraining frame
    ``(+right,+down,+forward)``. Native-CGA additionally approximates four
    analyzer intensities from DoLP/AoLP and RGB-luminance I_un. These are not
    measured analyzer images.
    """
    if input_mode not in ("robot", "native_cga"):
        raise ValueError(f"Unsupported RLBench input mode: {input_mode}")
    dolp = np.asarray(polar["DoLP"], dtype=np.float32)
    cos2 = np.asarray(polar["cos2AoLP"], dtype=np.float32)
    sin2 = np.asarray(polar["sin2AoLP"], dtype=np.float32)
    intensity = np.asarray(sfp["I_un"], dtype=np.float32) / 255.0
    mask = np.asarray(polar["valid_mask"], dtype=bool)
    angle_mask = np.asarray(polar["AoLP_valid_mask"], dtype=bool)
    if not (dolp.shape == cos2.shape == sin2.shape == intensity.shape == mask.shape == angle_mask.shape):
        raise ValueError("RLBench polar/intensity sidecars have different image shapes")
    k = np.asarray(sfp["K"], dtype=np.float32)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid RLBench front-camera intrinsics")
    dolp = np.where(mask & np.isfinite(dolp), np.clip(dolp, 0, 1), 0)
    cos2 = np.where(angle_mask & np.isfinite(cos2), cos2, 0)
    sin2 = np.where(angle_mask & np.isfinite(sin2), sin2, 0)
    norm = np.sqrt(cos2**2 + sin2**2)
    cos2 = np.where(norm > 1e-8, cos2 / np.maximum(norm, 1e-8), 0)
    sin2 = np.where(norm > 1e-8, sin2 / np.maximum(norm, 1e-8), 0)
    intensity = np.where(np.isfinite(intensity), np.clip(intensity, 0, 1), 0)
    aolp = 0.5 * np.arctan2(sin2, cos2)
    candidates = ambiguous_normals(dolp, aolp, refractive_index=refractive_index)
    candidates = np.asarray(candidates, dtype=np.float32).copy()
    candidates *= (mask & angle_mask)[None]

    # Ideal I0/I45/I90/I135 reconstructed from corrected Stokes ratios and
    # proxy I_un. Their max-min is max(|S1|, |S2|), not a measured contrast.
    contrast = intensity * dolp * np.maximum(np.abs(cos2), np.abs(sin2))
    spec = minimum_filter(contrast, size=3, mode="nearest")
    spec = np.where(mask & angle_mask, spec, 0).astype(np.float32)
    yy, xx = np.mgrid[:dolp.shape[0], :dolp.shape[1]].astype(np.float32)
    yy += 0.5
    xx += 0.5
    ray_x = (xx - k[0, 2]) / k[0, 0]
    rays = np.stack((ray_x, (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)))
    rays /= np.maximum(np.linalg.norm(rays, axis=0, keepdims=True), 1e-8)
    if input_mode == "native_cga":
        q = intensity * dolp * cos2
        u = intensity * dolp * sin2
        pseudo_analyzers = np.stack((
            0.5 * (intensity + q), 0.5 * (intensity + u),
            0.5 * (intensity - q), 0.5 * (intensity - u),
        ))
        observation = np.concatenate((
            pseudo_analyzers, intensity[None], cos2[None], sin2[None], dolp[None], rays,
        ))
    else:
        observation = np.concatenate((intensity[None], dolp[None], cos2[None], sin2[None], rays))
    prior = np.concatenate((candidates, intensity[None], spec[None]))
    expected_channels = 11 if input_mode == "native_cga" else 7
    if observation.shape[0] != expected_channels or prior.shape[0] != 11:
        raise AssertionError("Unexpected CGA input channel count")
    return observation.astype(np.float32), prior.astype(np.float32)


def _read_npz(txn: lmdb.Transaction, key: bytes, label: str) -> dict[str, np.ndarray]:
    value = txn.get(key)
    if value is None:
        raise KeyError(f"Missing {label} LMDB key {key.decode()}")
    with np.load(io.BytesIO(value), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


class RLBenchPolarNormalLmdbDataset(PolarNormalDataset):
    """Join three v2 LMDB sidecars and episode RGB video by episode-frame key."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        dataset_root: str | Path,
        image_size: int | tuple[int, int] = 256,
        require_rgb: bool = True,
        refractive_index: float = 1.5,
        input_mode: str = "robot",
        ray_dropout_prob: float = 0.0,
        limit: int | None = None,
    ):
        self.root = Path(dataset_root).resolve()
        self.manifest = Path(manifest)
        self.refractive_index = float(refractive_index)
        if input_mode not in ("robot", "native_cga"):
            raise ValueError(f"Unsupported RLBench input mode: {input_mode}")
        self._envs: dict[str, lmdb.Environment] = {}
        self._video_cache: dict[int, np.ndarray] = {}
        entries = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if limit is not None:
            entries = entries[:limit]
        if not entries:
            raise ValueError(f"Empty RLBench manifest: {manifest}")
        for entry in entries:
            if not all(k in entry for k in ("episode_index", "frame_index", "group")):
                raise ValueError(f"Incomplete RLBench manifest entry: {entry}")
        self.entries = entries
        self.input_mode = input_mode
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.require_rgb = require_rgb
        self.require_calibration = True
        self.normal_gt_source = "archived Coppelia depth + same-object mask finite-difference normals"
        import torch

        self.normal_transform = torch.eye(3)
        self.normal_sign = 1.0
        self.ray_dropout_prob = float(ray_dropout_prob)
        if not 0.0 <= self.ray_dropout_prob <= 1.0:
            raise ValueError("ray_dropout_prob must be between 0 and 1")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_envs"] = {}
        state["_video_cache"] = {}
        return state

    def _env(self, name: str) -> lmdb.Environment:
        if name not in self._envs:
            path = self.root / name
            if not path.is_dir():
                raise FileNotFoundError(path)
            address = str(path)
            if address not in _SHARED_ENVS:
                _SHARED_ENVS[address] = lmdb.open(
                    address, readonly=True, lock=False, readahead=False, max_readers=256
                )
            self._envs[name] = _SHARED_ENVS[address]
        return self._envs[name]

    def _rgb(self, episode: int, frame: int) -> np.ndarray:
        if episode not in self._video_cache:
            path = self.root / f"videos/chunk-{episode // 1000:03d}/observation.images.front_image/episode_{episode:06d}.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)
            with av.open(str(path)) as container:
                frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            self._video_cache = {episode: np.stack(frames)}  # keep one episode per worker
        frames = self._video_cache[episode]
        if not 0 <= frame < len(frames):
            raise IndexError(f"Video episode {episode} has {len(frames)} frames, requested {frame}")
        return frames[frame]

    def __getitem__(self, index: int):
        entry = self.entries[index]
        episode, frame = int(entry["episode_index"]), int(entry["frame_index"])
        key = f"{episode}-{frame}".encode()
        with self._env("polar_frontview_dense").begin() as txn:
            polar = _read_npz(txn, key, "polar")
        with self._env("sfp_frontview_rgb_luminance_proxy").begin() as txn:
            sfp = _read_npz(txn, key, "sfp")
        with self._env("normal_frontview_dense").begin() as txn:
            normal = _read_npz(txn, key, "normal")
        if not np.allclose(sfp["K"], normal["K"], atol=1e-4):
            raise ValueError(f"Calibration mismatch at {key.decode()}")
        observation, prior = generate_rlbench_cga_input(
            polar, sfp, refractive_index=self.refractive_index, input_mode=self.input_mode
        )
        normal_gt = np.asarray(normal["normal_gt"], dtype=np.float32).copy()
        stored_frame = normal.get("normal_coordinate_frame")
        if stored_frame is None:
            # V2 sidecars written before the canonical-frame migration stored
            # +x-left normals without schema metadata.
            stored_frame = "sfp_wild"
        else:
            stored_frame = str(np.asarray(stored_frame).item())
        if stored_frame in ("sfp_wild", "+left,+down,+forward"):
            normal_gt[..., 0] *= -1.0
        elif stored_frame not in ("canonical", "opencv", "+right,+down,+forward"):
            raise ValueError(f"Unsupported normal coordinate frame: {stored_frame!r}")
        record = {
            "polar_observation": observation,
            "physical_prior": prior,
            "normal_gt": normal_gt,
            "normal_valid_mask": normal["normal_valid_mask"],
            "K": normal["K"],
            "T_camera_from_world": normal["T_camera_from_world"],
        }
        if self.require_rgb:
            record["rgb"] = self._rgb(episode, frame)
        sample_entry = {"group": entry["group"], "id": key.decode(), "dataset": "rlbench_polar_v2"}
        return self._sample_from_record(record, sample_entry, self.root / key.decode())
