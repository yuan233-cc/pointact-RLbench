import io

import numpy as np
import torch

from pointact.data.robot.data_3d import LeRobotPointCloudDataset


def test_sfp_sidecar_assembles_official_seven_channels():
    height, width = 5, 7
    K = np.asarray([[20.0, 0.0, 3.0], [0.0, 21.0, 2.0], [0.0, 0.0, 1.0]],
                   dtype=np.float32)
    camera_from_world = np.eye(4, dtype=np.float32)
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        I_un=np.full((height, width), 128, dtype=np.uint8),
        K=K,
        T_camera_from_world=camera_from_world,
    )
    dense = np.zeros((4, height, width), dtype=np.float32)
    dense[0] = 0.25
    dense[1] = 0.6
    dense[2] = -0.8
    dense[3, 1:4, 2:6] = 1

    dataset = object.__new__(LeRobotPointCloudDataset)
    dataset.sfp_input_dir = "unused"
    dataset.use_point_image_support = False
    dataset.points_workspace = None
    dataset._read_sidecar = lambda *_args: payload.getvalue()
    item = dataset._load_sfp_inputs(0, 0, dense)

    assert item["polar_images"].shape == (1, 7, height, width)
    torch.testing.assert_close(
        item["polar_images"][0, :4],
        torch.from_numpy(np.concatenate((
            np.full((1, height, width), 128 / 255, dtype=np.float32), dense[:3]
        ))),
    )
    torch.testing.assert_close(item["polar_K"], torch.from_numpy(K[None]))
    torch.testing.assert_close(
        item["T_camera_from_world"], torch.from_numpy(camera_from_world[None])
    )
    assert item["view_valid"].tolist() == [True]
    assert item["pixel_valid"].shape == (1, height, width)
    torch.testing.assert_close(
        torch.linalg.vector_norm(item["polar_images"][0, 4:7], dim=0),
        torch.ones(height, width),
    )


def test_native_stokes_sidecar_overrides_legacy_dense_polar_channels():
    height, width = 3, 4
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        S0=np.full((height, width), 1.25, dtype=np.float16),
        DoLP=np.full((height, width), 0.3, dtype=np.float16),
        cos2AoLP=np.full((height, width), -0.6, dtype=np.float16),
        sin2AoLP=np.full((height, width), 0.8, dtype=np.float16),
        valid_mask=np.ones((height, width), dtype=np.uint8),
        K=np.asarray([[10.0, 0.0, 2.0], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        T_camera_from_world=np.eye(4, dtype=np.float32),
    )
    legacy_dense = np.zeros((4, height, width), dtype=np.float32)

    dataset = object.__new__(LeRobotPointCloudDataset)
    dataset.sfp_input_dir = "unused"
    dataset.use_point_image_support = True
    dataset.points_workspace = None
    dataset._read_sidecar = lambda *_args: payload.getvalue()
    item = dataset._load_sfp_inputs(0, 0, legacy_dense)

    expected = torch.tensor([1.25, 0.3, -0.6, 0.8])[:, None, None].expand(-1, height, width)
    torch.testing.assert_close(item["polar_images"][0, :4], expected, atol=3e-4, rtol=3e-4)
    assert item["pixel_valid"].all()


def test_workspace_mask_depends_on_camera_rays_not_depth_validity():
    dataset = object.__new__(LeRobotPointCloudDataset)
    dataset.points_workspace = {
        "X_BBOX": [-0.25, 0.25],
        "Y_BBOX": [-0.25, 0.25],
        "Z_BBOX": [1.0, 2.0],
    }
    K = np.asarray([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])
    mask = dataset._workspace_ray_mask(K, np.eye(4), 5, 5)
    assert mask.dtype == np.bool_
    assert mask[2, 2]
    assert not mask[0, 0]


def test_sfp_view_direction_center_and_axes():
    K = np.asarray([[10.0, 0.0, 2.5], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]])
    rays = LeRobotPointCloudDataset._viewing_directions(K, 3, 5)
    np.testing.assert_allclose(rays[:, 1, 2], [0, 0, 1], atol=1e-7)
    assert rays[0, 1, 0] < 0  # image-left
    assert rays[0, 1, 4] > 0  # image-right
    assert rays[1, 0, 2] < 0  # image-top
    assert rays[1, 2, 2] > 0  # image-bottom


def test_observed_points_are_z_buffered_into_sparse_metric_depth():
    points = np.asarray([
        [0.0, 0.0, 2.0, 1, 1, 1],
        [0.0, 0.0, 1.0, 1, 1, 1],  # same pixel; nearer point wins
        [0.0, 0.0, -1.0, 1, 1, 1],  # behind camera
        [1.0, 1.0, 3.0, 1, 1, 1],
    ], dtype=np.float32)
    pixels = np.asarray([4, 4, 5, 8], dtype=np.int32)
    depth, valid = LeRobotPointCloudDataset._rasterize_sparse_depth(
        points, pixels, np.eye(4, dtype=np.float32), height=3, width=3
    )
    assert depth.shape == valid.shape == (1, 3, 3)
    assert depth[0, 1, 1] == 1.0
    assert depth[0, 2, 2] == 3.0
    assert not valid[0, 1, 2]
