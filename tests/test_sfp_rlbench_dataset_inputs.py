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


def test_sfp_view_direction_center_and_axes():
    K = np.asarray([[10.0, 0.0, 2.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])
    rays = LeRobotPointCloudDataset._viewing_directions(K, 3, 5)
    np.testing.assert_allclose(rays[:, 1, 2], [0, 0, 1], atol=1e-7)
    assert rays[0, 1, 0] > 0  # image-left
    assert rays[0, 1, 4] < 0  # image-right
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
