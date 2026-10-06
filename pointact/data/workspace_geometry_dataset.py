"""Frame-key joined incomplete geometry; no complete depth or GT normals read."""
from __future__ import annotations
import io
from pathlib import Path
import av
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from .rlbench_polar_normal_lmdb import generate_rlbench_cga_input

msgpack_numpy.patch()
WORKSPACE = np.array([[-.5, 1.5], [-1., 1.], [.7505, 2.]], dtype=np.float32)


class WorkspaceGeometryDataset(Dataset):
    def __init__(self, root, backbone, split="train", max_points=4096, cga_records=None):
        self.root = Path(root)
        self.backbone, self.max_points = backbone, max_points
        self.cga_records = Path(cga_records) if cga_records else None
        self.envs, self.video_cache = {}, {}
        with lmdb.open(str(self.root / "points_frontview_polar_incomplete9"), readonly=True, lock=False) as env:
            keys = [key for key in env.begin().cursor().iternext(keys=True, values=False)
                    if key[:1].isdigit() and b"-" in key]
        self.keys = sorted([key for key in keys if (int(key.split(b"-")[0]) % 10 == 9) == (split == "val")],
                           key=lambda key: tuple(map(int, key.split(b"-"))))
        if not self.keys:
            raise ValueError("Empty episode-disjoint split")

    def __len__(self):
        return len(self.keys)

    def read(self, name, key, npz=False):
        if name not in self.envs:
            self.envs[name] = lmdb.open(str(self.root / name), readonly=True, lock=False,
                                       readahead=False, max_readers=512)
        payload = self.envs[name].begin().get(key)
        if payload is None:
            raise KeyError(f"Missing {name}/{key.decode()}")
        if npz:
            with np.load(io.BytesIO(payload), allow_pickle=False) as record:
                return {name: record[name] for name in record.files}
        return np.asarray(msgpack.unpackb(payload))

    def rgb(self, key):
        episode, frame = map(int, key.split(b"-"))
        if episode not in self.video_cache:
            path = self.root / f"videos/chunk-{episode // 1000:03d}/observation.images.front_image/episode_{episode:06d}.mp4"
            with av.open(str(path)) as container:
                self.video_cache = {episode: [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]}
        return self.video_cache[episode][frame].transpose(2, 0, 1).copy().astype(np.float32) / 255

    def __getitem__(self, index):
        key = self.keys[index]
        points = self.read("points_frontview_polar_incomplete9", key).astype(np.float32).copy()
        pixels = self.read("point_pixel_indices", key).astype(np.int64).reshape(-1)
        sidecar = "tasknet_frontview_native_stokes" if self.backbone == "tasknet" else "sfp_frontview_rgb_luminance_proxy"
        record = self.read(sidecar, key, True)
        k = np.asarray(record["K"], np.float32)
        transform = np.asarray(record["T_camera_from_world"], np.float32)
        if self.backbone == "tasknet":
            intensity = record["S0"].astype(np.float32)
            dolp, cos2, sin2 = (record[x].astype(np.float32) for x in ("DoLP", "cos2AoLP", "sin2AoLP"))
            pixel_valid = record["valid_mask"].astype(bool)
        else:
            polar = self.read("polar_frontview_dense", key, True)
            intensity = record["I_un"].astype(np.float32) / 255
            dolp, cos2, sin2 = (polar[x].astype(np.float32) for x in ("DoLP", "cos2AoLP", "sin2AoLP"))
            pixel_valid = polar["valid_mask"].astype(bool)
        height, width = intensity.shape
        if (height, width) != (256, 256) or len(points) != len(pixels) or points.shape[1] != 9:
            raise ValueError(f"Invalid calibrated frame {key!r}")
        if not np.isfinite(points).all() or not np.isfinite(k).all() or not np.isfinite(transform).all():
            raise ValueError("Nonfinite geometry/calibration")
        valid = ((points[:, :3] > WORKSPACE[:, 0]) & (points[:, :3] < WORKSPACE[:, 1])).all(1)
        points, pixels = points[valid], pixels[valid]
        if not len(points) or np.any((pixels < 0) | (pixels >= height * width)):
            raise ValueError("No valid workspace points or incorrect pixel IDs")
        camera = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        uv = (camera @ k.T)[:, :2] / camera[:, 2:3]
        expected = np.floor(uv + 1e-4).astype(np.int64)
        mismatch = (expected[:, 1] * width + expected[:, 0]) != pixels
        if mismatch.mean() > .01:
            raise ValueError(f"Calibration/pixel mismatch in {key.decode()}: {mismatch.mean():.3%}")
        # Targets come only from incomplete sensor points before subsampling.
        depth = np.full(height * width, np.inf, np.float32)
        np.minimum.at(depth, pixels[camera[:, 2] > 0], camera[camera[:, 2] > 0, 2])
        observed_valid = np.isfinite(depth)
        depth[~observed_valid] = 0
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        rays = np.stack(((xx + .5 - k[0, 2]) / k[0, 0], (yy + .5 - k[1, 2]) / k[1, 1], np.ones_like(xx)))
        unit_rays = rays / np.maximum(np.linalg.norm(rays, axis=0, keepdims=True), 1e-8)
        inverse = np.linalg.inv(transform)
        direction = rays.transpose(1, 2, 0) @ inverse[:3, :3].T
        origin = inverse[:3, 3]
        parallel = np.abs(direction) < 1e-9
        safe_direction = np.where(parallel, 1, direction)
        t0, t1 = (WORKSPACE[:, 0] - origin) / safe_direction, (WORKSPACE[:, 1] - origin) / safe_direction
        near = np.where(parallel, -np.inf, np.minimum(t0, t1)).max(-1)
        far = np.where(parallel, np.inf, np.maximum(t0, t1)).min(-1)
        outside_parallel = (parallel & ((origin < WORKSPACE[:, 0]) | (origin > WORKSPACE[:, 1]))).any(-1)
        workspace = (far >= np.maximum(near, 0)) & ~outside_parallel
        # Crop once, retaining optical observations over depth holes.
        envelope = np.zeros((height, width), bool)
        rows, cols = pixels // width, pixels % width
        envelope[rows.min():rows.max()+1, cols.min():cols.max()+1] = True
        workspace &= envelope
        sampled = np.random.choice(len(points), min(len(points), self.max_points), replace=False)
        points, pixels = points[sampled], pixels[sampled]
        points[:, 3:6] = 2 * points[:, 3:6] - 1
        points[:, 6] = 2 * points[:, 6] - 1
        center = points[:, :3].mean(0)
        points[:, :3] -= center
        camera_from_model = transform.copy()
        camera_from_model[:3, 3] += transform[:3, :3] @ center
        out = dict(points=points, point_pixel_indices=pixels,
            polar_images=np.concatenate((intensity[None], dolp[None], cos2[None], sin2[None], unit_rays))[None],
            polar_K=k[None], T_camera_from_model=camera_from_model[None], view_valid=np.ones(1, bool),
            pixel_valid=pixel_valid[None], polar_workspace_mask=workspace[None],
            observed_depth=depth.reshape(1, 1, height, width), observed_depth_valid=observed_valid.reshape(1, 1, height, width),
            point_pixel_image_hw=np.array([[height, width]], dtype=np.int64))
        if self.backbone == "cga_dinov3":
            if self.cga_records:
                episode, frame = map(int, key.split(b"-"))
                path = self.cga_records / f"episode_{episode:06d}_frame_{frame:06d}.npz"
                # Never access normal_gt or clean depth, even if the archive contains them.
                with np.load(path, allow_pickle=False) as cached:
                    if str(cached["source_id"]) != key.decode() or not np.allclose(cached["K"], k, atol=1e-5) or not np.allclose(cached["T_camera_from_world"], transform, atol=1e-5):
                        raise ValueError(f"Cached CGA frame/calibration mismatch: {path}")
                    observation, prior = cached["polar_observation"].copy(), cached["physical_prior"].copy()
                    rgb = cached["rgb"].transpose(2, 0, 1).copy().astype(np.float32) / 255
            else:
                observation, prior = generate_rlbench_cga_input(polar, record, input_mode="native_cga")
                # This checkpoint used integer-centered rays in archived records.
                archived_rays = np.stack(((xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)))
                archived_rays /= np.maximum(np.linalg.norm(archived_rays, axis=0, keepdims=True), 1e-8)
                observation[-3:] = archived_rays
                rgb = self.rgb(key)
            out.update(cga_observation=observation, cga_prior=prior, rgb=rgb)
        return {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in out.items()}


def collate_geometry(samples):
    output = {name: (torch.cat([s[name] for s in samples]) if name in ("points", "point_pixel_indices")
                     else torch.stack([s[name] for s in samples])) for name in samples[0]}
    output["npoints_in_batch"] = torch.tensor([len(s["points"]) for s in samples])
    return output
