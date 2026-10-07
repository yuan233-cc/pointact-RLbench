"""Observation-only operation-space masks. No scene/GT access or ray-box fill."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
from pointact.robot_envs.rlbench_utils.eval_utils import get_rlbench_robot_workspace


def operation_workspace():
    workspace = get_rlbench_robot_workspace()
    return np.asarray([workspace[k] for k in ("X_BBOX", "Y_BBOX", "Z_BBOX")], np.float32)


def observed_workspace_mask(depth, K, camera_from_world, radius=12, min_support=4):
    """Keep inside measurements and locally surrounded missing pixels.

    Unknown pixels are NaN. A hole requires >=min_support nearby measurements,
    support on both image axes, and NO outside measurement within the radius.
    Workspace membership uses measured positions only, never predicted depth.
    """
    depth = np.asarray(depth, np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    yy, xx = np.indices(depth.shape)
    rays = np.stack((xx, yy, np.ones_like(xx)), -1) @ np.linalg.inv(K).T
    world_from_camera = np.linalg.inv(camera_from_world)
    camera = rays * np.where(valid, depth, 0)[..., None]
    world = camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    bounds = operation_workspace()
    inside = valid & ((world > bounds[:, 0]) & (world < bounds[:, 1])).all(-1)
    outside = valid & ~inside
    holes = np.zeros_like(valid)
    support_yx = np.argwhere(inside)
    candidates = np.argwhere(~valid & ~maximum_filter(outside, size=2 * radius + 1, mode="constant"))
    if len(support_yx) >= min_support and len(candidates):
        distances, indices = cKDTree(support_yx).query(candidates, k=8, distance_upper_bound=radius)
        supported = np.isfinite(distances)
        safe_indices = np.minimum(indices, len(support_yx) - 1)
        delta = support_yx[safe_indices] - candidates[:, None, :]
        surround = ((supported & (delta[..., 0] < 0)).any(1)
                    & (supported & (delta[..., 0] > 0)).any(1)
                    & (supported & (delta[..., 1] < 0)).any(1)
                    & (supported & (delta[..., 1] > 0)).any(1))
        chosen = candidates[surround & (supported.sum(1) >= min_support)]
        holes[chosen[:, 0], chosen[:, 1]] = True
    return inside | holes, inside, holes
