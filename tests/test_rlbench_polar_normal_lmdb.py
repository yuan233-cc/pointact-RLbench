import numpy as np

from pointact.data.rlbench_polar_normal_lmdb import generate_rlbench_cga_input


def test_generator_shapes_coordinates_and_no_gt_dependency():
    height, width = 8, 10
    polar = {
        "DoLP": np.full((height, width), 0.3, np.float32),
        "cos2AoLP": np.ones((height, width), np.float32),
        "sin2AoLP": np.zeros((height, width), np.float32),
        "valid_mask": np.ones((height, width), bool),
        "AoLP_valid_mask": np.ones((height, width), bool),
    }
    sfp = {"I_un": np.full((height, width), 128, np.uint8),
           "K": np.array([[100, 0, 5], [0, 100, 4], [0, 0, 1]], np.float32)}
    observation, prior = generate_rlbench_cga_input(polar, sfp)
    assert observation.shape == (7, height, width)
    assert prior.shape == (11, height, width)
    assert np.isfinite(observation).all() and np.isfinite(prior).all()
    assert observation[4, 4, 4] < 0  # left of center is -x in canonical frame.
    assert observation[4, 4, 6] > 0
    assert (prior[[2, 5, 8]] < 0).all()  # candidates face camera.
    np.testing.assert_allclose(prior[10, 4, 4], 128 / 255 * 0.3, rtol=1e-5)
    polar["valid_mask"][2, 2] = False
    _, invalid_prior = generate_rlbench_cga_input(polar, sfp)
    np.testing.assert_array_equal(invalid_prior[:9, 2, 2], 0)


def test_native_cga_mode_reconstructs_proxy_analyzers_and_rightward_x():
    polar = {
        "DoLP": np.full((8, 10), 0.3, np.float32),
        "cos2AoLP": np.ones((8, 10), np.float32),
        "sin2AoLP": np.zeros((8, 10), np.float32),
        "valid_mask": np.ones((8, 10), bool),
        "AoLP_valid_mask": np.ones((8, 10), bool),
    }
    sfp = {"I_un": np.full((8, 10), 128, np.uint8),
           "K": np.array([[100, 0, 5], [0, 100, 4], [0, 0, 1]], np.float32)}
    robot, robot_prior = generate_rlbench_cga_input(polar, sfp)
    native, native_prior = generate_rlbench_cga_input(polar, sfp, input_mode="native_cga")
    assert native.shape == (11, 8, 10)
    np.testing.assert_allclose(native[4], 128 / 255, atol=1e-6)
    np.testing.assert_allclose(native[0] + native[2], native[4], atol=1e-6)
    np.testing.assert_allclose(native[1] + native[3], native[4], atol=1e-6)
    np.testing.assert_allclose(native[8:11], robot[4:7], atol=1e-6)
    np.testing.assert_allclose(native_prior, robot_prior, atol=1e-6)


def test_real_rlbench_frame():
    from pathlib import Path

    from pointact.data.rlbench_polar_normal_lmdb import RLBenchPolarNormalLmdbDataset

    root = Path("robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps_polar_rlbench9_v2")
    manifest = Path("robot_data/polar_normal_rlbench_v2_manifests/train.jsonl")
    if not root.exists() or not manifest.exists():
        return
    sample = RLBenchPolarNormalLmdbDataset(manifest, dataset_root=root, limit=1)[0]
    assert sample["rgb"].shape == (3, 256, 256)
    assert sample["polar_observation"].shape == (7, 256, 256)
    assert sample["physical_prior"].shape == (11, 256, 256)
    assert sample["normal_gt"].shape == (3, 256, 256)
    assert sample["normal_valid_mask"].any()

    native = RLBenchPolarNormalLmdbDataset(
        manifest, dataset_root=root, input_mode="native_cga", limit=1
    )[0]
    assert native["polar_observation"].shape == (11, 256, 256)
    np.testing.assert_allclose(native["normal_gt"], sample["normal_gt"], atol=1e-6)
    np.testing.assert_allclose(
        native["polar_observation"][8:11], sample["polar_observation"][4:7], atol=1e-6
    )
