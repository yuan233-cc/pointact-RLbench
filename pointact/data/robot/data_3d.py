import io
import json
import os
import random
from collections.abc import Callable
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
from lerobot.constants import ACTION, OBS_STATE
from pointact.constants import OBS_POINTS

from pointact.data.robot.base import LeRobotDatasetMixin
from pointact.data.robot.registry import register_robot_dataset
from pointact.data.polar_material import (
    dense_polar_from_bytes,
    dense_polar_from_npz,
    load_material_candidates,
    polar_vlm_image,
)
from pointact.data.transforms.pointcloud import (
    augment_point_cloud_color,
    normalize_polar_like_rgb,
    random_rotate_point_around_z,
    random_rotate_quat_around_z,
    random_rotate_delta_quat_around_z,
)

msgpack_numpy.patch()


@register_robot_dataset("LeRobotPointCloudDataset")
class LeRobotPointCloudDataset(LeRobotDatasetMixin):
    """LeRobot dataset variant backed by precomputed point clouds in LMDB.

    Point cloud LMDB entries are xyzrgb with RGB in [0, 1]. ``xyz_polar``
    replaces RGB with DoLP, cos(2 AoLP), and sin(2 AoLP), while
    ``xyzrgb_polar`` appends those three polarization features after RGB.
    State/action tensors are treated as world-frame values before optional rotation augmentation and point-cloud centering.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        # custom features
        select_video_keys: list[str] | None = None,
        select_state_keys: list[str] | None = None,
        select_action_keys: list[str] | None = None,
        train_subtask: str | None = None, # cumulate, mixture:0.5
        is_delta_action: bool = False,
        is_action_eef: bool = True,
        weight: float | None = None,
        image_size: int | None = None,
        converted_rot_type: str | None = None, # quat (default),euler, rot6d
        state_action_norm_file: str | None = None,
        # point related
        video_key_ids_for_vlm: list[int] | None = None,
        points_workspace: dict | None = None,
        max_npoints: int = 4096,
        augment_pc_rot: int = 0,
        augment_point_color: bool = True,
        point_cloud_dirname: str | None = None,
        point_feature_mode: str = "xyzrgb",
        polar_feature_normalization: str = "raw",
        polar_dense_dirname: str | None = None,
        polar_dense_frames_dir: str | None = None,
        sfp_input_dirname: str | None = None,
        depth_point_pixel_dirname: str | None = None,
        use_point_image_support: bool = False,
        vlm_image_mode: str = "rgb",
        point_pixel_dirname: str | None = None,
        material_profiles_file: str | None = None,
        material_candidate_names: list[str] | None = None,
        target_reconstruction_dirname: str | None = None,
        **kwargs,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            select_action_keys=select_action_keys,
            select_state_keys=select_state_keys,
            select_video_keys=select_video_keys,
            video_key_ids_for_vlm=video_key_ids_for_vlm,
            train_subtask=train_subtask,
            is_delta_action=is_delta_action,
            is_action_eef=is_action_eef,
            weight=weight,
            image_size=image_size,
            converted_rot_type=converted_rot_type,
            state_action_norm_file=state_action_norm_file,
        )

        self.points_workspace = points_workspace
        self.max_npoints = max_npoints
        self.augment_pc_rot = augment_pc_rot
        self.augment_point_color = augment_point_color
        if point_feature_mode not in ("xyzrgb", "xyz_polar", "xyzrgb_polar"):
            raise ValueError(f"Unsupported point_feature_mode={point_feature_mode!r}")
        self.point_feature_mode = point_feature_mode
        if polar_feature_normalization not in ("raw", "rgb"):
            raise ValueError(
                "Unsupported polar_feature_normalization="
                f"{polar_feature_normalization!r}; expected 'raw' or 'rgb'"
            )
        self.polar_feature_normalization = polar_feature_normalization
        if vlm_image_mode not in ("rgb", "polar"):
            raise ValueError(f"Unsupported vlm_image_mode={vlm_image_mode!r}; expected 'rgb' or 'polar'")
        self.vlm_image_mode = vlm_image_mode
        has_dense_polar = bool(polar_dense_dirname) != bool(polar_dense_frames_dir)
        if polar_dense_dirname and polar_dense_frames_dir:
            raise ValueError("Configure only one dense polar source")
        self.use_point_image_support = use_point_image_support
        material_requested = bool(material_profiles_file or material_candidate_names
                                  or (point_pixel_dirname and not use_point_image_support))
        if material_requested and not (point_pixel_dirname and material_profiles_file and has_dense_polar):
            raise ValueError("Material conditioning requires point pixels, material profiles, and one dense polar source")
        if vlm_image_mode == "polar" and not has_dense_polar:
            raise ValueError("Polar VLM image mode requires one dense polar source")
        if sfp_input_dirname and not has_dense_polar:
            raise ValueError("SfP-Wild inputs require one dense polar source")
        if depth_point_pixel_dirname and not sfp_input_dirname:
            raise ValueError("Sparse depth rasterization requires SfP-Wild camera metadata")
        if use_point_image_support and not (sfp_input_dirname and (point_pixel_dirname or depth_point_pixel_dirname)):
            raise ValueError("Image support requires dense inputs and a current-observation point pixel sidecar")
        if has_dense_polar and vlm_image_mode != "polar" and not material_requested and not sfp_input_dirname:
            raise ValueError(
                "A dense polar source requires polar VLM image mode, material conditioning, "
                "or SfP-Wild inputs"
            )
        self.use_polar_material_conditioning = material_requested
        self.polar_dense_dir = self._resolve_sidecar_path(polar_dense_dirname) if polar_dense_dirname else None
        self.polar_dense_frames_dir = self._resolve_sidecar_path(polar_dense_frames_dir) if polar_dense_frames_dir else None
        self.point_pixel_dir = self._resolve_sidecar_path(point_pixel_dirname) if point_pixel_dirname else None
        self.sfp_input_dir = self._resolve_sidecar_path(sfp_input_dirname) if sfp_input_dirname else None
        self.depth_point_pixel_dir = (
            self._resolve_sidecar_path(depth_point_pixel_dirname)
            if depth_point_pixel_dirname else None
        )
        self.material_candidates = (torch.from_numpy(load_material_candidates(
            self._resolve_sidecar_path(material_profiles_file), material_candidate_names
        )) if material_requested else None)
        self.target_reconstruction_dir = (
            self._resolve_sidecar_path(target_reconstruction_dirname)
            if target_reconstruction_dirname else None
        )
        if self.target_reconstruction_dir is not None:
            manifest_path = self.target_reconstruction_dir / "manifest.json"
            if manifest_path.exists() and not json.loads(manifest_path.read_text())["complete"]:
                raise ValueError(f"Target reconstruction sidecar is incomplete: {manifest_path}")
        self._sidecar_envs = {}
        self._sidecar_pid = None

        assert point_cloud_dirname is not None
        self.point_cloud_dir = os.path.join(self.root, point_cloud_dirname)
        self._point_cloud_lmdb_env = None
        self._point_cloud_lmdb_txn = None
        self._point_cloud_lmdb_pid = None    

    def __del__(self):
        for env in getattr(self, "_sidecar_envs", {}).values():
            env.close()
        if getattr(self, "_point_cloud_lmdb_txn", None) is not None:
            self._point_cloud_lmdb_txn.abort()
            self._point_cloud_lmdb_txn = None
        if getattr(self, "_point_cloud_lmdb_env", None) is not None:
            self._point_cloud_lmdb_env.close()
            self._point_cloud_lmdb_env = None
        self._point_cloud_lmdb_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_point_cloud_lmdb_env"] = None
        state["_point_cloud_lmdb_txn"] = None
        state["_point_cloud_lmdb_pid"] = None
        state["_sidecar_envs"] = {}
        state["_sidecar_pid"] = None
        return state

    def _resolve_sidecar_path(self, path):
        if path is None:
            return None
        candidate = Path(path)
        if candidate.is_absolute() or candidate.exists():
            return candidate
        return Path(self.root) / candidate

    def _read_sidecar(self, path: Path, ep_idx: int, frame_idx: int):
        if not path.is_dir():
            raise FileNotFoundError(path)
        file = path / f"{frame_idx:06d}.npy"
        if file.exists():
            return np.load(file)
        file = path / f"{frame_idx:06d}.npz"
        if file.exists():
            return file
        if self._sidecar_pid != os.getpid():
            self._sidecar_envs = {}
            self._sidecar_pid = os.getpid()
        key = str(path)
        if key not in self._sidecar_envs:
            self._sidecar_envs[key] = lmdb.open(key, readonly=True, lock=False, readahead=False)
        with self._sidecar_envs[key].begin(buffers=True) as txn:
            value = txn.get(f"{ep_idx}-{frame_idx}".encode("ascii"))
            if value is None:
                raise KeyError(f"Missing sidecar {ep_idx}-{frame_idx} in {path}")
            return bytes(value)

    def _load_point_pixels(self, ep_idx, frame_idx, path=None):
        value = self._read_sidecar(path or self.point_pixel_dir, ep_idx, frame_idx)
        return np.asarray(msgpack.unpackb(value) if isinstance(value, bytes) else value, dtype=np.int32)

    @staticmethod
    def _rasterize_sparse_depth(
        point_cloud: np.ndarray,
        point_pixels: np.ndarray,
        camera_from_world: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project aligned observed points into a z-buffered metric-depth map."""
        pixels = np.asarray(point_pixels, dtype=np.int64).reshape(-1)
        if len(pixels) != len(point_cloud):
            raise ValueError("Depth point pixels and point cloud have different row counts")
        xyz = np.asarray(point_cloud[:, :3], dtype=np.float32)
        transform = np.asarray(camera_from_world, dtype=np.float32)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("T_camera_from_world must be a finite 4x4 matrix")
        camera_xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        z = camera_xyz[:, 2]
        valid = (
            (pixels >= 0) & (pixels < height * width)
            & np.isfinite(z) & (z > 0)
        )
        flat_depth = np.full(height * width, np.inf, dtype=np.float32)
        np.minimum.at(flat_depth, pixels[valid], z[valid])
        flat_valid = np.isfinite(flat_depth)
        flat_depth[~flat_valid] = 0.0
        depth = torch.from_numpy(flat_depth.reshape(1, height, width))
        mask = torch.from_numpy(flat_valid.reshape(1, height, width))
        return depth, mask

    def _load_target_reconstruction(self, ep_idx, frame_idx, input_point_count):
        value = self._read_sidecar(self.target_reconstruction_dir, ep_idx, frame_idx)
        record = msgpack.unpackb(value) if isinstance(value, bytes) else value
        points = np.asarray(record["points"], dtype=np.float32).reshape(-1, 3).copy()
        point_mask = np.asarray(record["input_mask"], dtype=np.float32).reshape(-1)
        if (len(point_mask) != input_point_count or not np.isfinite(points).all()
                or not np.isin(point_mask, [0, 1]).all()):
            raise ValueError(f"Invalid target labels for {ep_idx}-{frame_idx}")
        return points, point_mask

    def _load_dense_polar(self, ep_idx, frame_idx):
        path = self.polar_dense_dir or self.polar_dense_frames_dir
        value = self._read_sidecar(path, ep_idx, frame_idx)
        if isinstance(value, np.ndarray):
            if value.ndim != 3 or value.shape[0] != 4:
                raise ValueError(f"Dense polar array must be [4,H,W], got {value.shape}")
            return value.astype(np.float32)
        return dense_polar_from_bytes(value) if isinstance(value, bytes) else dense_polar_from_npz(value)

    @staticmethod
    def _viewing_directions(K: np.ndarray, height: int, width: int) -> np.ndarray:
        """Build the SfP-Wild (+left, +down, +forward) unit ray image."""
        K = np.asarray(K, dtype=np.float32)
        if (K.shape != (3, 3) or not np.isfinite(K).all()
                or K[0, 0] <= 0 or K[1, 1] <= 0):
            raise ValueError("SfP intrinsics must be a finite 3x3 matrix with positive focal lengths")
        rows, cols = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        rays = np.stack((
            (K[0, 2] - cols) / K[0, 0],
            (rows - K[1, 2]) / K[1, 1],
            np.ones((height, width), dtype=np.float32),
        ))
        rays /= np.linalg.norm(rays, axis=0, keepdims=True).clip(1e-8)
        return rays.astype(np.float32)

    def _workspace_ray_mask(
        self, K: np.ndarray, camera_from_world: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Pixels whose OpenCV camera rays intersect the configured world AABB.

        This mask deliberately does not depend on depth.  Polar pixels behind a
        missing/corrupt depth sample therefore remain available as a complementary
        signal.  It is computed once while loading a sample and reused at every
        PTv3 stage.
        """
        if getattr(self, "points_workspace", None) is None:
            return np.ones((height, width), dtype=bool)
        bounds = np.asarray(
            [
                self.points_workspace["X_BBOX"],
                self.points_workspace["Y_BBOX"],
                self.points_workspace["Z_BBOX"],
            ],
            dtype=np.float64,
        )
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("points_workspace must contain finite increasing X/Y/Z_BBOX bounds")

        rows, cols = np.meshgrid(
            np.arange(height, dtype=np.float64),
            np.arange(width, dtype=np.float64),
            indexing="ij",
        )
        directions_camera = np.stack(
            (
                (cols - K[0, 2]) / K[0, 0],
                (rows - K[1, 2]) / K[1, 1],
                np.ones_like(rows),
            ),
            axis=-1,
        )
        world_from_camera = np.linalg.inv(camera_from_world.astype(np.float64))
        origin = world_from_camera[:3, 3]
        directions = directions_camera @ world_from_camera[:3, :3].T

        # Slab ray/AABB test with an explicit parallel-axis case.
        parallel = np.abs(directions) < 1e-12
        parallel_outside = parallel & (
            (origin[None, None, :] < bounds[:, 0])
            | (origin[None, None, :] > bounds[:, 1])
        )
        safe_directions = np.where(parallel, 1.0, directions)
        t0 = (bounds[:, 0] - origin) / safe_directions
        t1 = (bounds[:, 1] - origin) / safe_directions
        near = np.where(parallel, -np.inf, np.minimum(t0, t1)).max(axis=-1)
        far = np.where(parallel, np.inf, np.maximum(t0, t1)).min(axis=-1)
        mask = (~parallel_outside.any(axis=-1)) & (far >= np.maximum(near, 0.0))
        if not mask.any():
            raise ValueError("The configured 3D points_workspace does not intersect this camera image")
        return mask

    def _load_sfp_inputs(self, ep_idx: int, frame_idx: int, dense_polar: np.ndarray) -> dict:
        value = self._read_sidecar(self.sfp_input_dir, ep_idx, frame_idx)
        source = io.BytesIO(value) if isinstance(value, bytes) else value
        with np.load(source) as record:
            i_un = np.asarray(record["I_un"])
            K = np.asarray(record["K"], dtype=np.float32)
            camera_from_world = np.asarray(record["T_camera_from_world"], dtype=np.float32)
            prior = np.asarray(record["physical_prior"], dtype=np.float32) if "physical_prior" in record else None
            if prior is None and all(key in record for key in ("est", "spec")):
                prior = np.concatenate(
                    (
                        np.asarray(record["est"], dtype=np.float32),
                        np.asarray(record["I_un"], dtype=np.float32)[None],
                        np.asarray(record["spec"], dtype=np.float32).reshape(1, *i_un.shape),
                    ),
                    axis=0,
                )
            rgb = None
            for rgb_key in ("rgb", "RGB", "color"):
                if rgb_key in record:
                    rgb = np.asarray(record[rgb_key]).copy()
                    break
        if i_un.ndim != 2:
            raise ValueError(f"SfP I_un must be HxW, got {i_un.shape}")
        i_un = i_un.astype(np.float32) / 255.0 if i_un.dtype == np.uint8 else i_un.astype(np.float32)
        if dense_polar.shape != (4, *i_un.shape):
            raise ValueError(
                f"SfP intensity and dense polar shapes differ: {i_un.shape} vs {dense_polar.shape}"
            )
        if camera_from_world.shape != (4, 4) or not np.isfinite(camera_from_world).all():
            raise ValueError("T_camera_from_world must be a finite 4x4 matrix")
        if not np.isfinite(i_un).all():
            raise ValueError("SfP I_un contains non-finite values")
        rays = self._viewing_directions(K, *i_un.shape)
        polar_images = np.concatenate((i_un[None], dense_polar[:3], rays), axis=0)
        workspace_mask = (
            np.ones(i_un.shape, dtype=bool)
            if self.use_point_image_support
            else self._workspace_ray_mask(K, camera_from_world, *i_un.shape)
        )
        result = {
            "polar_images": torch.from_numpy(polar_images[None].copy()),
            "polar_K": torch.from_numpy(K[None].copy()),
            "T_camera_from_world": torch.from_numpy(camera_from_world[None].copy()),
            "view_valid": torch.ones(1, dtype=torch.bool),
            "pixel_valid": torch.from_numpy((dense_polar[3:4] > 0.5).copy()),
            "polar_workspace_mask": torch.from_numpy(workspace_mask[None].copy()),
        }
        if self.use_point_image_support:
            result["point_pixel_image_hw"] = torch.tensor([i_un.shape], dtype=torch.long)
        if prior is not None:
            if prior.shape != (11, *i_un.shape):
                raise ValueError(f"CGA physical_prior must be [11,H,W], got {prior.shape}")
            result["polar_physical_prior"] = torch.from_numpy(prior[None].copy())
        if rgb is not None:
            if rgb.ndim == 3 and rgb.shape[-1] == 3:
                rgb = rgb.transpose(2, 0, 1)
            if rgb.shape != (3, *i_un.shape):
                raise ValueError(f"CGA+DINO RGB must be [3,H,W], got {rgb.shape}")
            rgb = rgb.astype(np.float32)
            if rgb.max() > 1:
                rgb /= 255.0
            result["polar_rgb"] = torch.from_numpy(rgb[None].copy())
        return result

    def set_feature_keys(
        self, video_keys=None, state_keys=None, action_keys=None,
        video_key_ids_for_vlm=None, **kwargs
    ):
        # the point cloud can be constructed using all video keys, while the VLM only utilizes a subset of the video keys
        self.select_video_keys = self.meta.video_keys if video_keys is None else video_keys
        if video_key_ids_for_vlm is None:
            self.select_video_keys_for_vlm = self.select_video_keys
        else:
            self.select_video_keys_for_vlm = [self.select_video_keys[i] for i in video_key_ids_for_vlm]

        self.select_state_keys = (
            [key for key in self.meta.features if key.startswith(OBS_STATE)]
            if state_keys is None
            else state_keys
        )
        self.select_action_keys = (
            [key for key in self.meta.features if key.startswith(ACTION)]
            if action_keys is None
            else action_keys
        )

        self.select_feature_keys = self.select_video_keys_for_vlm + self.select_state_keys + self.select_action_keys
        self.select_action_is_pad_keys = [f"{key}_is_pad" for key in self.select_action_keys]

    def __getitem__(self, idx, delta_indices: dict = None) -> dict:
        delta_indices = delta_indices or self.delta_indices

        if self.weight is not None:
            idx = random.randint(0, self.num_frames - 1)

        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        frame_idx = item["frame_index"].item()

        item, query_indices = self.query_action_chunk(item, idx, ep_idx, delta_indices)
        item = self.add_video_frames(item, ep_idx, query_indices)
        dense_polar = None
        if self.sfp_input_dir is not None:
            dense_polar = self._load_dense_polar(ep_idx, frame_idx)
            item.update(self._load_sfp_inputs(ep_idx, frame_idx, dense_polar))
        if self.use_polar_material_conditioning:
            rgb_key = self.select_video_keys[0]
            if rgb_key not in item:
                item.update(self._query_videos({rgb_key: [item["timestamp"].item()]}, ep_idx))
            rgb = item[rgb_key]
            if rgb.ndim == 4:
                rgb = rgb[0]
            rgb = torch.as_tensor(rgb).float()
            if rgb.ndim == 3 and rgb.shape[-1] == 3:
                rgb = rgb.permute(2, 0, 1)
            if rgb.max() > 1:
                rgb = rgb / 255.0
            item["material_rgb"] = rgb.contiguous()
            dense_polar = self._load_dense_polar(ep_idx, frame_idx)
            item["polar_dense"] = torch.from_numpy(dense_polar)
            if item["material_rgb"].shape != (3, *item["polar_dense"].shape[-2:]):
                raise ValueError("Dense RGB/polar image sizes differ; pixel coordinates would be misaligned")
            item["material_candidates"] = self.material_candidates
        if self.vlm_image_mode == "polar":
            if len(self.select_video_keys_for_vlm) != 1:
                raise ValueError("Polar VLM image mode requires exactly one selected VLM image key")
            if dense_polar is None:
                dense_polar = self._load_dense_polar(ep_idx, frame_idx)
            item[self.select_video_keys_for_vlm[0]] = polar_vlm_image(dense_polar)
        else:
            self.apply_image_transforms(item, self.select_video_keys_for_vlm)

        point_cloud = self.load_point_cloud(ep_idx, frame_idx)
        depth_point_pixels = None
        if self.depth_point_pixel_dir is not None:
            depth_point_pixels = self._load_point_pixels(
                ep_idx, frame_idx, self.depth_point_pixel_dir
            ).reshape(-1)
            height, width = item["polar_images"].shape[-2:]
            sparse_depth, sparse_depth_valid = self._rasterize_sparse_depth(
                point_cloud,
                depth_point_pixels,
                item["T_camera_from_world"][0].numpy(),
                height,
                width,
            )
            item["observed_depth"] = sparse_depth.unsqueeze(0)
            item["observed_depth_valid"] = sparse_depth_valid.unsqueeze(0)
        if self.target_reconstruction_dir is not None:
            target_points, target_mask = self._load_target_reconstruction(
                ep_idx, frame_idx, len(point_cloud)
            )
            item["target_points"] = torch.from_numpy(target_points)
            point_cloud = np.column_stack((point_cloud, target_mask)).astype(np.float32)
        if self.use_polar_material_conditioning or self.use_point_image_support:
            pixel_dir = self.point_pixel_dir or self.depth_point_pixel_dir
            point_pixels = (
                depth_point_pixels
                if depth_point_pixels is not None and pixel_dir == self.depth_point_pixel_dir
                else self._load_point_pixels(ep_idx, frame_idx, pixel_dir).reshape(-1)
            )
            if len(point_pixels) != len(point_cloud):
                raise ValueError("Point pixels and point cloud have different row counts")
            if self.use_point_image_support:
                if depth_point_pixels is not None and not np.array_equal(point_pixels, depth_point_pixels):
                    raise ValueError("BBox support and sparse-depth targets must use identical current pixel IDs")
                height, width = item["point_pixel_image_hw"][0].tolist()
                if np.any((point_pixels < 0) | (point_pixels >= height * width)):
                    raise ValueError("Current-observation pixel indices are outside the polar image")
                if np.any(point_pixels > 2 ** 24):
                    raise ValueError("Pixel IDs exceed exact float32 precision for row-preserving augmentation")
            point_cloud = np.column_stack((point_cloud, point_pixels)).astype(np.float32)
        point_cloud = self.filter_point_cloud_by_workspace(point_cloud)
        if self.use_point_image_support and len(point_cloud) == 0:
            raise ValueError("Image-support fusion requires nonempty points after workspace filtering")
        if self.use_point_image_support and "polar_workspace_mask" in item:
            # An RLBench camera may itself lie inside the broad 3D AABB, making
            # ray/AABB intersection cover the whole image.  The envelope of all
            # pre-subsampling points that survived the workspace filter gives a
            # useful 2D crop while retaining every missing-depth/polar pixel in it.
            height, width = item["polar_workspace_mask"].shape[-2:]
            pixels = point_cloud[:, -1].astype(np.int64)
            rows, cols = pixels // width, pixels % width
            workspace_mask = torch.zeros_like(item["polar_workspace_mask"])
            workspace_mask[
                0,
                int(rows.min()):int(rows.max()) + 1,
                int(cols.min()):int(cols.max()) + 1,
            ] = True
            item["polar_workspace_mask"] = workspace_mask
        point_cloud = self.augment_point_cloud(point_cloud, item)
        point_cloud = self.center_point_cloud(point_cloud, item)
        if self.use_polar_material_conditioning or self.use_point_image_support:
            item["point_pixel_indices"] = torch.from_numpy(point_cloud[:, -1].astype(np.int64))
            point_cloud = np.ascontiguousarray(point_cloud[:, :-1])
            if len(item["point_pixel_indices"]) != len(point_cloud):
                raise ValueError("Pixel correspondence was lost during point row processing")
        if self.target_reconstruction_dir is not None:
            item["target_input_mask"] = torch.from_numpy(point_cloud[:, -1].copy())
            point_cloud = np.ascontiguousarray(point_cloud[:, :-1])
        item[OBS_POINTS] = torch.from_numpy(point_cloud)

        self.convert_eef_rotation(item)
        self.normalize_state_action(item)
        self.select_task_text(item, ep_idx, idx)

        return self.post_process(item)

    def load_point_cloud(self, ep_idx: int, frame_idx: int):
        txn = self.get_point_cloud_lmdb_txn()
        point_key = f"{ep_idx}-{frame_idx}"
        point_cloud = txn.get(point_key.encode("ascii"))
        if point_cloud is None:
            raise KeyError(f"Point cloud '{point_key}' not found in {self.point_cloud_dir}")
        point_cloud = np.asarray(msgpack.unpackb(point_cloud), dtype=np.float32).copy()
        if self.point_feature_mode == "xyzrgb":
            if point_cloud.ndim != 2 or point_cloud.shape[1] not in (6, 9):
                raise ValueError(
                    f"XYZRGB point cloud '{point_key}' must have shape Nx6, or Nx9 "
                    f"when reading an XYZRGB+polar archive, got {point_cloud.shape}"
                )
            # Let six-channel ablations use the exact same geometry, RGB and
            # point sampling as a polar archive without duplicating its LMDB.
            # The three optical channels are deliberately removed here.
            point_cloud = np.ascontiguousarray(point_cloud[:, :6])
            if not np.isfinite(point_cloud).all():
                raise ValueError(f"XYZRGB point cloud '{point_key}' has non-finite values")
        else:
            if point_cloud.ndim != 2 or point_cloud.shape[1] != 9:
                raise ValueError(
                    f"Polar point cloud '{point_key}' must have shape Nx9 "
                    f"[xyz, rgb, DoLP, cos2AoLP, sin2AoLP], got {point_cloud.shape}"
                )
            if not np.isfinite(point_cloud).all():
                raise ValueError(f"Polar point cloud '{point_key}' has non-finite values")
            if np.any((point_cloud[:, 6] < 0) | (point_cloud[:, 6] > 1)):
                raise ValueError(f"Polar point cloud '{point_key}' has DoLP outside [0, 1]")
            if np.any(np.abs(point_cloud[:, 7:9]) > 1.001):
                raise ValueError(f"Polar point cloud '{point_key}' has invalid AoLP channels")
            if self.point_feature_mode == "xyz_polar":
                # Reuse the exact same points from the filled9 archive while
                # replacing per-point RGB with its aligned polarization tuple.
                point_cloud = np.ascontiguousarray(
                    np.column_stack((point_cloud[:, :3], point_cloud[:, 6:9])),
                    dtype=np.float32,
                )
        return point_cloud

    def filter_point_cloud_by_workspace(self, point_cloud: np.ndarray):
        if self.points_workspace is None:
            return point_cloud

        workspace = self.points_workspace
        point_mask = (
            (point_cloud[:, 0] > workspace["X_BBOX"][0])
            & (point_cloud[:, 0] < workspace["X_BBOX"][1])
            & (point_cloud[:, 1] > workspace["Y_BBOX"][0])
            & (point_cloud[:, 1] < workspace["Y_BBOX"][1])
            & (point_cloud[:, 2] > workspace["Z_BBOX"][0])
            & (point_cloud[:, 2] < workspace["Z_BBOX"][1])
        )
        return point_cloud[point_mask]

    def get_point_cloud_lmdb_txn(self):
        current_pid = os.getpid()
        if self._point_cloud_lmdb_pid != current_pid:
            self._point_cloud_lmdb_env = None
            self._point_cloud_lmdb_txn = None
            self._point_cloud_lmdb_pid = current_pid

        if self._point_cloud_lmdb_env is None:
            self._point_cloud_lmdb_env = lmdb.open(
                self.point_cloud_dir,
                readonly=True,
                lock=False,
                readahead=False,
                max_spare_txns=1,
            )
            self._point_cloud_lmdb_txn = self._point_cloud_lmdb_env.begin(buffers=True)

        return self._point_cloud_lmdb_txn

    def augment_point_cloud(self, point_cloud: np.ndarray, item: dict):
        model_from_world = (
            np.eye(4, dtype=np.float32) if "polar_images" in item else None
        )
        max_npoints = min(int(len(point_cloud) * np.random.uniform(0.8, 1.0)), self.max_npoints)
        if self.use_point_image_support:
            if self.max_npoints <= 0:
                raise ValueError("Image-support fusion requires max_npoints > 0")
            max_npoints = max(1, max_npoints)
        if len(point_cloud) > max_npoints:
            ridxs = np.random.choice(len(point_cloud), max_npoints, replace=False)
            point_cloud = point_cloud[ridxs]

        if self.point_feature_mode in ("xyzrgb", "xyzrgb_polar"):
            point_cloud_color = point_cloud[:, 3:6]
            if self.augment_point_color:
                # Color augmentation is modality-specific. Preserve NumPy's
                # sampling RNG state so enabling RGB augmentation cannot alter
                # the point subsets selected for later examples in an ablation.
                rng_state = np.random.get_state()
                try:
                    point_cloud_color = augment_point_cloud_color(
                        point_cloud_color,
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        jitter_std=0.02,
                    )
                finally:
                    np.random.set_state(rng_state)
            point_cloud[:, 3:6] = point_cloud_color * 2 - 1

        if self.polar_feature_normalization == "rgb":
            polar_slice = slice(3, 6) if self.point_feature_mode == "xyz_polar" else slice(6, 9)
            point_cloud[:, polar_slice] = normalize_polar_like_rgb(
                point_cloud[:, polar_slice]
            )

        if self.augment_pc_rot != 0:
            angle = np.random.uniform(-1, 1) * np.deg2rad(self.augment_pc_rot)
            cosine, sine = np.cos(angle), np.sin(angle)
            if model_from_world is not None:
                model_from_world[:3, :3] = np.asarray(
                    [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]],
                    dtype=np.float32,
                )
            point_cloud[:, :3] = random_rotate_point_around_z(point_cloud[:, :3], angle=angle)
            if "target_points" in item:
                item["target_points"] = random_rotate_point_around_z(
                    item["target_points"], angle=angle
                )
            if self.is_action_eef:
                item[OBS_STATE][:3] = random_rotate_point_around_z(
                    item[OBS_STATE][:3].unsqueeze(0), angle=angle
                )[0]
                item[OBS_STATE][3:7] = random_rotate_quat_around_z(item[OBS_STATE][3:7], angle)
                item[ACTION][:, :3] = random_rotate_point_around_z(item[ACTION][:, :3], angle=angle)
                if self.is_delta_action:
                    item[ACTION][:, 3:7] = random_rotate_delta_quat_around_z(item[ACTION][:, 3:7], angle)
                else:
                    item[ACTION][:, 3:7] = random_rotate_quat_around_z(item[ACTION][:, 3:7], angle)

        if model_from_world is not None:
            item["T_model_from_world"] = torch.from_numpy(model_from_world)
        return point_cloud

    def center_point_cloud(self, point_cloud: np.ndarray, item: dict):
        point_center = point_cloud[:, :3].mean(0)
        point_cloud[:, :3] = point_cloud[:, :3] - point_center
        point_center = torch.from_numpy(point_center)
        if self.is_action_eef:
            item[OBS_STATE][:3] = item[OBS_STATE][:3] - point_center
            if not self.is_delta_action:
                item[ACTION][:, :3] = item[ACTION][:, :3] - point_center[None, :]
        item[f"{OBS_POINTS}.center"] = point_center
        if "polar_images" in item:
            if "T_camera_from_world" not in item:
                raise ValueError(
                    "Polar-token training samples require T_camera_from_world so random "
                    "rotation/centering can be composed exactly"
                )
            if "T_model_from_world" not in item:
                item["T_model_from_world"] = torch.eye(4, dtype=point_center.dtype)
            item["T_model_from_world"] = item["T_model_from_world"].clone()
            item["T_model_from_world"][:3, 3] = -point_center
            camera_from_world = torch.as_tensor(item["T_camera_from_world"], dtype=torch.float32)
            item["T_camera_from_model"] = camera_from_world @ torch.linalg.inv(
                item["T_model_from_world"]
            ).unsqueeze(0)
        if "target_points" in item:
            item["target_points"] = item["target_points"] - point_center
        return point_cloud

    def post_process(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ordered_keys = self.select_feature_keys + [OBS_POINTS, "task", f"{OBS_POINTS}.center"] + self.select_action_is_pad_keys
        if self.use_polar_material_conditioning:
            ordered_keys += ["material_rgb", "polar_dense", "material_candidates", "point_pixel_indices"]
        elif self.use_point_image_support:
            ordered_keys += ["point_pixel_indices"]
        if self.use_point_image_support:
            ordered_keys += ["point_pixel_image_hw"]
        if self.target_reconstruction_dir is not None:
            ordered_keys += ["target_points", "target_input_mask"]
        if "polar_images" in item:
            ordered_keys += [
                "polar_images", "polar_rgb", "polar_physical_prior", "polar_K",
                "T_camera_from_model", "T_model_from_world",
                "view_valid", "pixel_valid", "polar_pixel_transform",
                "polar_workspace_mask",
            ]
        if "observed_depth" in item:
            ordered_keys += ["observed_depth", "observed_depth_valid"]
        item = {key: item[key] for key in ordered_keys if key in item}
        return item
