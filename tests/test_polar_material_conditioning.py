import numpy as np
import torch

from pointact.data.polar_material import (
    MATERIAL_FEATURE_DIM,
    dense_polar_from_npz,
    material_vector,
    polar_vlm_image,
)
from pointact.model.vla_pointact.polar_material_conditioner import PolarMaterialConditioner
from pointact.model.vla_pointact.action_head_3d.ptv3_backbone import PointTransformerUnetWithAction


def test_material_coefficients_keep_missing_value_masks():
    dielectric = material_vector({"type": "dielectric", "int_ior": 1.52, "ext_ior": 1.0})
    assert dielectric.shape == (MATERIAL_FEATURE_DIM,)
    assert dielectric[1] == 1 and dielectric[16 + 1] == 1
    assert np.isclose(dielectric[10], 1.52) and dielectric[16 + 10] == 1
    assert dielectric[9] == 0 and dielectric[16 + 9] == 0


def test_dense_polar_retains_polar_only_hole_pixels(tmp_path):
    path = tmp_path / "frame.npz"
    np.savez(path, DoLP=np.ones((2, 2), np.float32),
             AoLP=np.zeros((2, 2), np.float32),
             AoLP_valid_mask=np.ones((2, 2), bool),
             valid_mask=np.zeros((2, 2), bool))
    dense = dense_polar_from_npz(path)
    assert dense.shape == (4, 2, 2)
    assert dense[3].sum() == 4  # no depth, yet the polar angles remain observable


def test_polar_vlm_image_has_fixed_channels_and_zero_invalid_angles():
    dense = torch.tensor([
        [[0.25, 1.25], [float("nan"), 0.75]],
        [[-1.0, 1.0], [0.5, -0.5]],
        [[1.0, -1.0], [0.0, 0.5]],
        [[1.0, 1.0], [0.0, 0.0]],
    ])
    image = polar_vlm_image(dense)
    assert image.shape == (3, 2, 2)
    torch.testing.assert_close(image[:, 0, 0], torch.tensor([0.25, 0.0, 1.0]))
    torch.testing.assert_close(image[:, 0, 1], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(image[:, 1, 0], torch.zeros(3))
    torch.testing.assert_close(image[:, 1, 1], torch.tensor([0.75, 0.0, 0.0]))


def test_conditioned_points_follow_pixel_indices_and_backpropagate():
    torch.manual_seed(0)
    conditioner = PolarMaterialConditioner(point_channels=8, hidden_channels=8)
    rgb = torch.rand(2, 3, 16, 16)
    polar = torch.rand(2, 4, 16, 16)
    materials = torch.rand(2, 3, MATERIAL_FEATURE_DIM)
    pixels = torch.tensor([0, 0, 255, 50, 50])
    counts = torch.tensor([3, 2])
    features, logits = conditioner(rgb, polar, materials, pixels, counts,
                                  torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool))
    assert features.shape == (5, 8)
    assert logits.shape == (2, 4, 4, 4)
    torch.testing.assert_close(features[0], features[1])
    torch.testing.assert_close(features[3], features[4])
    features.square().mean().backward()
    assert conditioner.image_encoder[0].weight.grad.abs().sum() > 0
    assert conditioner.material_encoder[0].weight.grad.abs().sum() > 0


def test_ptv3_batch_keeps_original_path_when_condition_is_absent():
    class MinimalWrapper:
        voxel_size = 0.01
        enc_channels = (8,)

    point = torch.zeros(3, 9)
    count = torch.tensor([3])
    context = torch.zeros(1, 2, 8)
    context_len = torch.tensor([2])
    action = torch.zeros(1, 1, 8)
    prepare = PointTransformerUnetWithAction.prepare_ptv3_batch
    plain = prepare(MinimalWrapper(), point, count, context, context_len, action)
    assert "point_condition" not in plain
    conditioned = prepare(MinimalWrapper(), point, count, context, context_len, action,
                          point_condition=torch.ones(3, 8))
    torch.testing.assert_close(conditioned["point_condition"], torch.ones(3, 8))
