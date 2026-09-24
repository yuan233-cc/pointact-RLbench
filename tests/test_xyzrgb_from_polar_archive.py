import unittest

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


if __name__ == "__main__":
    unittest.main()
