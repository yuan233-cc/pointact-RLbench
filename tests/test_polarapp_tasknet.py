import tempfile

import pytest
import torch

from pointact.model.vla_pointact.action_head_3d.polar_router import (
    routed_feature_geometry,
)
from pointact.model.vla_pointact.action_head_3d.polar_depth_self_supervision import (
    PolarDepthSelfSupervision,
)
from pointact.model.vla_pointact.action_head_3d.polarapp_tasknet_encoder import (
    PolarAppTaskAwareEncoder,
    load_polarapp_tasknet_checkpoint,
)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_taskaware_pyramid_has_five_real_scales_on_odd_input():
    encoder = PolarAppTaskAwareEncoder(
        input_mode="sfp_proxy", num_blocks=(1, 1, 1), pyramid_channels=16
    ).eval()
    observation = torch.randn(2, 7, 33, 35)
    with torch.no_grad():
        task_features = encoder.forward_task_features(observation)
        levels = encoder.forward_features(observation)
    assert [tuple(value.shape) for value in task_features] == [
        (2, 96, 33, 35),
        (2, 96, 17, 18),
        (2, 192, 9, 9),
    ]
    assert [tuple(value.shape) for value in levels] == [
        (2, 16, 33, 35),
        (2, 16, 17, 18),
        (2, 16, 9, 9),
        (2, 16, 5, 5),
        (2, 16, 3, 3),
    ]


def test_sfp_proxy_conversion_reorders_angles_and_replaces_rays():
    encoder = PolarAppTaskAwareEncoder(num_blocks=(1, 1, 1), pyramid_channels=8)
    observation = torch.zeros(1, 7, 2, 4)
    observation[:, 0] = 0.25
    observation[:, 1] = 0.5
    observation[:, 2] = 0.75  # cos(2AoLP)
    observation[:, 3] = -0.25  # sin(2AoLP)
    observation[:, 4:] = 9.0  # SfP-Wild rays must not leak into TaskNet coords.
    converted = encoder.prepare_tasknet_input(observation)
    assert torch.equal(converted[:, 0], observation[:, 0])
    assert torch.equal(converted[:, 1], observation[:, 1])
    assert torch.equal(converted[:, 2], observation[:, 3])
    assert torch.equal(converted[:, 3], observation[:, 2])
    assert torch.equal(converted[:, 6], torch.ones_like(converted[:, 6]))
    assert converted[:, 4:7].abs().max() <= 1.0


def test_tasknet_freeze_leaves_new_pyramid_trainable():
    encoder = PolarAppTaskAwareEncoder(num_blocks=(1, 1, 1), pyramid_channels=8)
    encoder.set_tasknet_trainable(False)
    assert not any(
        parameter.requires_grad
        for module in encoder._tasknet_modules()
        for parameter in module.parameters()
    )
    assert all(
        parameter.requires_grad
        for name, parameter in encoder.named_parameters()
        if name.startswith("pyramid_")
    )


def test_unfrozen_tasknet_and_pyramid_receive_finite_gradients():
    encoder = PolarAppTaskAwareEncoder(
        input_mode="tasknet7", num_blocks=(1, 1, 1), pyramid_channels=8
    )
    levels = encoder.forward_features(torch.randn(1, 7, 8, 8))
    sum(level.mean() for level in levels).backward()
    tasknet_gradient = encoder.encoder1[0].attn.qkv.conv.weight.grad
    pyramid_gradient = encoder.pyramid_lateral1.weight.grad
    assert tasknet_gradient is not None and torch.isfinite(tasknet_gradient).all()
    assert pyramid_gradient is not None and torch.isfinite(pyramid_gradient).all()


def test_released_normal_head_is_unit_length_and_uses_shared_loss_frame():
    encoder = PolarAppTaskAwareEncoder(
        input_mode="tasknet7", num_blocks=(1, 1, 1), pyramid_channels=8
    ).eval()
    with torch.no_grad():
        features = encoder.forward_task_features(torch.randn(1, 7, 8, 10))
        native = encoder.decode_normals(features, output_frame="tasknet")
        shared = encoder.decode_normals(features, output_frame="sfp_wild")
    torch.testing.assert_close(
        torch.linalg.vector_norm(native, dim=1), torch.ones(1, 8, 10),
        atol=1e-5, rtol=1e-5,
    )
    torch.testing.assert_close(shared, -native)


def test_normal_teacher_can_stay_frozen_while_tasknet_features_train():
    encoder = PolarAppTaskAwareEncoder(num_blocks=(1, 1, 1), pyramid_channels=8)
    encoder.set_normal_head_trainable(False)
    assert encoder.encoder1[0].attn.qkv.conv.weight.requires_grad
    assert not any(
        parameter.requires_grad
        for module in encoder._normal_head_modules()
        for parameter in module.parameters()
    )


def test_checkpoint_loader_requires_tasknet_but_not_new_pyramid():
    source = PolarAppTaskAwareEncoder(num_blocks=(1, 1, 1), pyramid_channels=8)
    source_state = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith("pyramid_")
    }
    target = PolarAppTaskAwareEncoder(num_blocks=(1, 1, 1), pyramid_channels=8)
    with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint:
        torch.save({"model_state_dict": source_state}, checkpoint.name)
        report = load_polarapp_tasknet_checkpoint(target, checkpoint.name)
    assert report["loaded"] == len(source_state)
    assert all(name.startswith("pyramid_") for name in report["missing"])
    assert report["normal_head_loaded"]


def test_checkpoint_loader_can_require_released_normal_head():
    source = PolarAppTaskAwareEncoder(num_blocks=(1, 1, 1), pyramid_channels=8)
    source_state = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith(("pyramid_", "refinement.", "output."))
    }
    target = PolarAppTaskAwareEncoder(num_blocks=(1, 1, 1), pyramid_channels=8)
    with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint:
        torch.save({"model_state_dict": source_state}, checkpoint.name)
        with pytest.raises(ValueError, match="refinement|output"):
            load_polarapp_tasknet_checkpoint(
                target, checkpoint.name, require_normal_head=True
            )


def test_tasknet_depth_branch_backpropagates_without_moving_normal_teacher():
    torch.manual_seed(3)
    height = width = 16
    encoder = PolarAppTaskAwareEncoder(
        input_mode="tasknet7", num_blocks=(1, 1, 1), pyramid_channels=8
    )
    encoder.set_normal_head_trainable(False)
    observation = torch.randn(1, 7, height, width)
    task_features = encoder.forward_task_features(observation)
    image_levels = tuple(level[:, None] for level in encoder.build_pyramid(task_features))
    with torch.no_grad():
        targets = encoder.decode_normals(
            tuple(level.detach() for level in task_features),
            output_frame="sfp_wild",
        )[:, None]

    point_channels = (4, 6, 8, 10, 12)
    point_levels = tuple(
        torch.randn(1, 1, channels, *level.shape[-2:])
        for channels, level in zip(point_channels, image_levels)
    )
    point_valid = tuple(
        torch.ones_like(level[:, :, :1], dtype=torch.bool)
        for level in point_levels
    )
    polar = observation[:, None]
    intrinsics = torch.tensor(
        [[[[40.0, 0.0, 7.5], [0.0, 40.0, 7.5], [0.0, 0.0, 1.0]]]]
    )
    sparse = torch.zeros(1, 1, 1, height, width)
    sparse[:, :, :, 3::4, 3::4] = 1.0
    objective = PolarDepthSelfSupervision(
        (8, 8, 8, 8, 8), point_channels, min_depth=0.05, max_depth=3.0
    )
    result = objective(
        image_levels,
        point_levels,
        point_valid,
        polar,
        intrinsics,
        observed_depth=sparse,
        observed_depth_valid=sparse > 0,
        normal_targets=targets,
    )
    result["loss"].backward()
    assert encoder.encoder1[0].attn.qkv.conv.weight.grad is not None
    assert encoder.pyramid_lateral1.weight.grad is not None
    assert all(
        parameter.grad is None
        for module in encoder._normal_head_modules()
        for parameter in module.parameters()
    )


def test_router_uses_backbone_specific_feature_geometry():
    legacy = AttrDict()
    assert routed_feature_geometry(legacy, 2) == (4, 1.5)
    polarapp = AttrDict(
        polar_feature_strides=(1, 2, 4, 8, 16),
        polar_feature_offsets=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert routed_feature_geometry(polarapp, 2) == (4.0, 0.0)
