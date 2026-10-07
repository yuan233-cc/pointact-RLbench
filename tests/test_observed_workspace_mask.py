import numpy as np
from pointact.data.observed_workspace_mask import observed_workspace_mask, operation_workspace
from pointact.robot_envs.rlbench_utils.eval_utils import get_rlbench_robot_workspace


def test_workspace_comes_from_original_operation_space():
    original = get_rlbench_robot_workspace()
    np.testing.assert_allclose(operation_workspace(), [original[k] for k in ("X_BBOX", "Y_BBOX", "Z_BBOX")])


def test_holes_need_surrounding_inside_support_and_outside_veto():
    depth = np.full((19, 19), np.nan, np.float32)
    K = np.array([[100., 0, 9], [0, 100., 9], [0, 0, 1.]])
    T = np.eye(4)
    for row, col in ((7, 9), (11, 9), (9, 7), (9, 11)):
        depth[row, col] = 1.
    mask, inside, holes = observed_workspace_mask(depth, K, T, radius=3)
    assert inside.sum() == 4 and holes[9, 9] and mask[9, 9]
    assert not mask[0, 0]
    depth[8, 8] = 3.  # Measured point outside the operation-space Z range.
    mask, inside, holes = observed_workspace_mask(depth, K, T, radius=3)
    assert not holes[9, 9] and not mask[8, 8]
    depth[:] = np.nan
    assert not observed_workspace_mask(depth, K, T)[0].any()


def test_large_enclosed_hole_uses_boundary_not_center_radius():
    depth = np.ones((81, 81), np.float32)
    depth[20:61, 20:61] = np.nan
    K = np.array([[400., 0, 40.], [0, 400., 40.], [0, 0, 1.]])
    mask_old, _, holes_old = observed_workspace_mask(depth, K, np.eye(4), connected_holes=False)
    assert not holes_old[40, 40]
    mask, inside, holes = observed_workspace_mask(depth, K, np.eye(4))
    assert holes[20:61, 20:61].all()
    assert mask[40, 40] and not inside[40, 40]


def test_connected_hole_outside_boundary_and_exterior_are_rejected():
    depth = np.ones((81, 81), np.float32)
    depth[20:61, 20:61] = np.nan
    K = np.array([[400., 0, 40.], [0, 400., 40.], [0, 0, 1.]])
    depth[19, 40] = 3.
    assert not observed_workspace_mask(depth, K, np.eye(4))[2][40, 40]
    # Opening to the exterior is not a closed surface hole.
    depth[19, 40] = 1.
    depth[:21, 30:51] = np.nan
    assert not observed_workspace_mask(depth, K, np.eye(4))[2][40, 40]


def test_no_reconstructed_points_used_as_boundary_evidence():
    depth = np.full((81, 81), np.nan, np.float32)
    depth[30:50, 30:50] = 1.
    K = np.array([[400., 0, 40.], [0, 400., 40.], [0, 0, 1.]])
    mask, inside, holes = observed_workspace_mask(depth, K, np.eye(4))
    assert not holes[0, 0] and not holes[20, 20]
    assert np.array_equal(mask, inside)
