import json

import numpy as np
import pytest

from pointact.data.polar_normal_dataset import PolarNormalDataset, assert_disjoint_groups


def make_record(path, height=40, width=48):
    record = {
        "images": np.random.rand(4, height, width).astype(np.float32),
        "est": np.random.randn(9, height, width).astype(np.float32),
        "spec": np.random.rand(1, height, width).astype(np.float32),
        "Iun": np.random.rand(height, width).astype(np.float32),
        "cos1": np.random.rand(1, height, width).astype(np.float32),
        "cos2": np.random.rand(1, height, width).astype(np.float32),
        "DoP": np.random.rand(1, height, width).astype(np.float32),
        "image_coordinate": np.random.randn(3, height, width).astype(np.float32),
        "rgb": np.random.randint(0, 256, (height, width, 3), dtype=np.uint8),
        "label": np.random.randn(3, height, width).astype(np.float32),
        "mask": np.ones((1, height, width), dtype=np.float32),
        "K": np.array([[40, 0, 24], [0, 40, 20], [0, 0, 1]], dtype=np.float32),
    }
    np.savez(path, **record)


def write_manifest(path, sample, group):
    path.write_text(json.dumps({"samples": [{"path": sample.name, "group": group}]}))


def test_native_cga_dataset_builds_true_branches_and_scales_intrinsics(tmp_path):
    sample_path = tmp_path / "sample.npz"
    manifest_path = tmp_path / "manifest.json"
    make_record(sample_path)
    write_manifest(manifest_path, sample_path, "object-a")
    sample = PolarNormalDataset(
        manifest_path,
        input_mode="native_cga",
        image_size=(80, 96),
        normal_gt_source="public_dataset_gt",
    )[0]
    assert sample["polar_observation"].shape == (11, 80, 96)
    assert sample["physical_prior"].shape == (11, 80, 96)
    assert sample["rgb"].shape == (3, 80, 96)
    assert sample["normal_gt"].shape == (3, 80, 96)
    np.testing.assert_allclose(sample["camera_K"][:2].numpy(), [[80, 0, 48], [0, 80, 40]])


def test_group_leakage_and_invalid_gt_source_are_rejected(tmp_path):
    sample_path = tmp_path / "sample.npz"
    make_record(sample_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_manifest(first, sample_path, "same-episode")
    write_manifest(second, sample_path, "same-episode")
    train = PolarNormalDataset(first, normal_gt_source="mesh_rendered_visible_surface")
    val = PolarNormalDataset(second, normal_gt_source="mesh_rendered_visible_surface")
    with pytest.raises(ValueError, match="occurs in dataset splits"):
        assert_disjoint_groups(train, val)
    with pytest.raises(ValueError, match="invalid supervision source"):
        PolarNormalDataset(first, normal_gt_source="damaged_depth_normals")


def test_ray_dropout_zeros_only_observation_rays(tmp_path):
    sample_path = tmp_path / "sample.npz"
    manifest_path = tmp_path / "manifest.json"
    make_record(sample_path)
    write_manifest(manifest_path, sample_path, "object-a")
    common = dict(input_mode="native_cga", image_size=32, normal_gt_source="public_dataset_gt")
    retained = PolarNormalDataset(manifest_path, ray_dropout_prob=0.0, **common)[0]
    dropped = PolarNormalDataset(manifest_path, ray_dropout_prob=1.0, **common)[0]
    assert retained["polar_observation"][-3:].abs().sum() > 0
    assert dropped["polar_observation"][-3:].count_nonzero() == 0
    np.testing.assert_array_equal(
        dropped["polar_observation"][:-3], retained["polar_observation"][:-3]
    )
    np.testing.assert_array_equal(dropped["physical_prior"], retained["physical_prior"])
    np.testing.assert_array_equal(dropped["normal_gt"], retained["normal_gt"])
    np.testing.assert_array_equal(dropped["normal_valid_mask"], retained["normal_valid_mask"])
    with pytest.raises(ValueError, match="ray_dropout_prob"):
        PolarNormalDataset(manifest_path, ray_dropout_prob=1.1, **common)
