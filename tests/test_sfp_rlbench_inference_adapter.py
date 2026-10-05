import numpy as np
import pytest

from pointact.robot_envs.rlbench_utils.sfp_adapter import (
    CAMERA_AXIS_CONVERSION,
    LUMINANCE_WEIGHTS,
    sfp_inputs_from_polar_frame,
)


def _frame():
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[20, 40, 60], [255, 255, 255], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    intrinsics = np.array(
        [[-100.0, 0.0, 1.0], [0.0, -120.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    to_world = np.eye(4, dtype=np.float32)
    to_world[:3, 3] = [0.1, -0.2, 1.0]
    return {
        "rgb": rgb,
        "DoLP": np.array([[0.2, 0.4, 0.6], [0.8, 1.2, 0.1]], dtype=np.float32),
        "AoLP": np.array([[0.0, np.pi / 4, np.pi / 2], [np.nan, 0.3, 0.6]], dtype=np.float32),
        "valid_mask": np.ones((2, 3), dtype=bool),
        "AoLP_valid_mask": np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
        "camera": {"intrinsics": intrinsics, "to_world": to_world},
    }


def test_live_polar_frame_matches_training_sfp_layout():
    frame = _frame()
    result = sfp_inputs_from_polar_frame(frame)

    assert result["polar_images"].shape == (1, 7, 2, 3)
    assert result["polar_K"].shape == (1, 3, 3)
    assert result["T_camera_from_world"].shape == (1, 4, 4)
    assert result["view_valid"].shape == (1,)
    assert result["pixel_valid"].shape == (1, 2, 3)

    expected_luma = np.clip(
        np.rint(frame["rgb"].astype(np.float32) @ LUMINANCE_WEIGHTS), 0, 255
    ) / 255.0
    np.testing.assert_allclose(result["polar_images"][0, 0], expected_luma)
    np.testing.assert_allclose(result["polar_images"][0, 1, 0], [0.2, 0.4, 0.6])
    assert result["polar_images"][0, 1, 1, 1] == 0.0  # invalid DoLP > 1
    np.testing.assert_allclose(result["polar_images"][0, 2, 0], [1.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(result["polar_images"][0, 3, 0], [0.0, 1.0, 0.0], atol=1e-6)
    assert result["polar_images"][0, 2, 1, 0] == 0.0  # non-finite AoLP
    assert result["polar_images"][0, 2, 1, 2] == 0.0  # AoLP mask

    expected_k = frame["camera"]["intrinsics"].copy()
    expected_k[0, 0] *= -1
    expected_k[1, 1] *= -1
    np.testing.assert_allclose(result["polar_K"][0], expected_k)
    expected_pose = CAMERA_AXIS_CONVERSION @ np.linalg.inv(frame["camera"]["to_world"])
    np.testing.assert_allclose(result["T_camera_from_world"][0], expected_pose)
    ray_norms = np.linalg.norm(result["polar_images"][0, 4:7], axis=0)
    np.testing.assert_allclose(ray_norms, 1.0, atol=1e-6)
    assert result["polar_images"][0, 4, 0, 0] < 0  # image-left is canonical -x
    assert result["polar_images"][0, 4, 0, 2] > 0  # image-right is canonical +x


def test_adapter_rejects_an_already_converted_camera():
    frame = _frame()
    frame["camera"]["intrinsics"][0, 0] *= -1
    frame["camera"]["intrinsics"][1, 1] *= -1
    with pytest.raises(ValueError, match="negative-focal"):
        sfp_inputs_from_polar_frame(frame)
