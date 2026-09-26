import unittest
from unittest.mock import patch

import msgpack
import msgpack_numpy
import numpy as np

from pointact.data.robot.data_3d import LeRobotPointCloudDataset


msgpack_numpy.patch()


class _Transaction:
    def __init__(self, value):
        self.value = value

    def get(self, key):
        return self.value if key == b"0-0" else None


class TestXYZRGBFromPolarArchive(unittest.TestCase):
    def test_xyzrgb_mode_strips_polar_channels(self):
        source = np.arange(36, dtype=np.float32).reshape(4, 9)
        dataset = LeRobotPointCloudDataset.__new__(LeRobotPointCloudDataset)
        dataset.point_feature_mode = "xyzrgb"
        dataset.point_cloud_dir = "unused"
        dataset.get_point_cloud_lmdb_txn = lambda: _Transaction(msgpack.packb(source))

        result = dataset.load_point_cloud(0, 0)

        self.assertEqual(result.shape, (4, 6))
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result, source[:, :6])

    def test_xyz_polar_mode_replaces_rgb_with_aligned_polar_channels(self):
        source = np.array([
            [0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.25, -0.5, 0.75],
            [0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.80, 1.0, -1.0],
        ], dtype=np.float32)
        dataset = LeRobotPointCloudDataset.__new__(LeRobotPointCloudDataset)
        dataset.point_feature_mode = "xyz_polar"
        dataset.point_cloud_dir = "unused"
        dataset.get_point_cloud_lmdb_txn = lambda: _Transaction(msgpack.packb(source))

        result = dataset.load_point_cloud(0, 0)

        self.assertEqual(result.shape, (2, 6))
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result[:, :3], source[:, :3])
        np.testing.assert_array_equal(result[:, 3:6], source[:, 6:9])

    def test_xyz_polar_mode_uses_rgb_range_mapping_without_color_augmentation(self):
        source = np.array([
            [0.1, 0.2, 0.3, 0.25, -0.5, 0.75],
            [0.4, 0.5, 0.6, 0.80, 1.0, -1.0],
        ], dtype=np.float32)
        dataset = LeRobotPointCloudDataset.__new__(LeRobotPointCloudDataset)
        dataset.point_feature_mode = "xyz_polar"
        dataset.max_npoints = 4096
        dataset.augment_pc_rot = 0
        dataset.augment_point_color = False
        dataset.polar_feature_normalization = "rgb"

        with patch("numpy.random.uniform", return_value=1.0):
            result = dataset.augment_point_cloud(source.copy(), {})

        expected = source.copy()
        expected[:, 3] = expected[:, 3] * 2 - 1
        # Mapping cos/sin to [0, 1] and then to [-1, 1] is an identity.
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)

    def test_nine_channel_mode_matches_rgb_and_polar_normalization_without_augmentation(self):
        source = np.array([
            [0.1, 0.2, 0.3, 0.2, 0.5, 0.9, 0.25, -0.5, 0.75],
            [0.4, 0.5, 0.6, 0.8, 0.0, 0.4, 0.80, 1.0, -1.0],
        ], dtype=np.float32)
        dataset = LeRobotPointCloudDataset.__new__(LeRobotPointCloudDataset)
        dataset.point_feature_mode = "xyzrgb_polar"
        dataset.max_npoints = 4096
        dataset.augment_pc_rot = 0
        dataset.augment_point_color = False
        dataset.polar_feature_normalization = "rgb"

        with patch("numpy.random.uniform", return_value=1.0):
            result = dataset.augment_point_cloud(source.copy(), {})

        expected = source.copy()
        expected[:, 3:6] = expected[:, 3:6] * 2 - 1
        expected[:, 6] = expected[:, 6] * 2 - 1
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
