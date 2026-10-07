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
