"""Observation-only operation-space masks. No scene/GT access or ray-box fill."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter, binary_dilation, binary_fill_holes, label, find_objects
from pointact.robot_envs.rlbench_utils.eval_utils import get_rlbench_robot_workspace


def operation_workspace():
    workspace = get_rlbench_robot_workspace()
    return np.asarray([workspace[k] for k in ("X_BBOX", "Y_BBOX", "Z_BBOX")], np.float32)


def connected_workspace_holes(valid, inside, outside, bridge_radius=2,
                              min_boundary_support=8, max_area_fraction=0.35):
    """Admit enclosed missing regions from ORIGINAL boundary observations.

    A small support-band dilation bridges sampling gaps, not arbitrary image
    regions. Exterior/background components remain connected to the image
    border and are rejected. Never turn inferred holes into new 3D evidence.
    """
    if bridge_radius < 0 or min_boundary_support < 4 or not 0 < max_area_fraction <= 1:
        raise ValueError("Invalid connected-hole parameters")
    barrier = binary_dilation(inside, iterations=bridge_radius) if bridge_radius else inside.copy()
    enclosed = binary_fill_holes(barrier) & ~barrier & ~valid
    labels, _ = label(enclosed)  # Four-connected; diagonal contact does not merge surfaces.
    accepted = np.zeros_like(valid)
    margin = bridge_radius + 2
    for component, box in enumerate(find_objects(labels), start=1):
        if box is None:
            continue
        rows, cols = box
        if rows.start == 0 or cols.start == 0 or rows.stop == valid.shape[0] or cols.stop == valid.shape[1]:
            continue
        window = (slice(max(0, rows.start - margin), min(valid.shape[0], rows.stop + margin)),
                  slice(max(0, cols.start - margin), min(valid.shape[1], cols.stop + margin)))
        core = labels[window] == component
        if core.sum() > max_area_fraction * valid.size:
            continue
        region = binary_dilation(core, iterations=bridge_radius) if bridge_radius else core
        ring = binary_dilation(region, iterations=1) & ~core
        # An outside measurement on/near the enclosing boundary vetoes the
        # entire component. It cannot be covered up by a majority vote.
        if (outside[window] & (ring | region)).any():
            continue
        support = np.argwhere(inside[window] & ring)
        if len(support) < min_boundary_support:
            continue
        center = np.argwhere(core).mean(0)
        delta = support - center
        if not ((delta[:, 0] < 0).any() and (delta[:, 0] > 0).any()
                and (delta[:, 1] < 0).any() and (delta[:, 1] > 0).any()):
            continue
        accepted[window] |= region & ~valid[window]
    return accepted


def observed_workspace_mask(depth, K, camera_from_world, radius=12, min_support=4,
                            connected_holes=True, bridge_radius=2):
    """Keep inside measurements and observed-boundary-supported missing pixels.

    Unknown pixels are NaN. A hole requires >=min_support nearby measurements,
    support on both image axes, and NO outside measurement within the radius.
    Workspace membership uses measured positions only, never predicted depth.
    Larger enclosed holes use connected boundary support when enabled, rather
    than requiring observations within ``radius`` of every interior pixel.
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
        distances, indices = cKDTree(support_yx).query(candidates, k=32, distance_upper_bound=radius)
        supported = np.isfinite(distances)
        safe_indices = np.minimum(indices, len(support_yx) - 1)
        delta = support_yx[safe_indices] - candidates[:, None, :]
        surround = ((supported & (delta[..., 0] < 0)).any(1)
                    & (supported & (delta[..., 0] > 0)).any(1)
                    & (supported & (delta[..., 1] < 0)).any(1)
                    & (supported & (delta[..., 1] > 0)).any(1))
        chosen = candidates[surround & (supported.sum(1) >= min_support)]
        holes[chosen[:, 0], chosen[:, 1]] = True
    if connected_holes:
        holes |= connected_workspace_holes(valid, inside, outside, bridge_radius=bridge_radius)
    return inside | holes, inside, holes
