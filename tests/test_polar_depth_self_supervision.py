import torch

from pointact.model.vla_pointact.action_head_3d.polar_depth_self_supervision import (
    PolarDepthSelfSupervision,
    PolarPointDepthDecoder,
    depth_to_normals,
    mask_points_at_depth_targets,
    rasterize_fused_point_features,
)


POLAR_CHANNELS = (8, 16, 24, 32, 32)
POINT_CHANNELS = (6, 8, 12, 16, 20)


def test_weighted_workspace_has_two_losses_no_holdout_and_all_observed_targets():
    torch.manual_seed(7)
    maps, masks = _point_maps(requires_grad=True)
    module = PolarDepthSelfSupervision(POLAR_CHANNELS, POINT_CHANNELS,
        use_polar_features=False, supervision_mode="weighted_workspace", smoothness_weight=0.)
    depth = torch.ones(1, 1, 1, 32, 32)
    valid = torch.ones_like(depth, dtype=torch.bool)
    workspace = valid[:, :, 0].clone()
    workspace[..., :4, :] = False
    normal = torch.zeros(1, 1, 3, 32, 32)
    normal[:, :, 2] = -1
    k = torch.tensor([[[[100., 0, 16], [0, 100., 16], [0, 0, 1.]]]])
    q = torch.zeros_like(depth, requires_grad=True)
    with torch.no_grad():
        q[..., 16:, :] = 1
    args = (None, maps, masks, torch.zeros(1, 1, 7, 32, 32), k, depth, valid)
    kwargs = dict(normal_targets=normal, workspace_mask=workspace, observation_confidence=q)
    module.train()
    output = module(*args, **kwargs)
    assert "holdout_depth_loss" not in output and "anchor_depth_loss" not in output
    weights = output["point_fit_weights"]
    assert (weights[..., :4, :] == 0).all()
    assert torch.allclose(weights[..., 4:16, :], torch.full_like(weights[..., 4:16, :], .1))
    assert (weights[..., 16:, :] == 1).all()
    torch.testing.assert_close(output["loss"], output["normal_consistency_loss"] + output["point_fit_loss"])
    module.eval()
    evaluated = module(*args, **kwargs)
    torch.testing.assert_close(output["loss"], evaluated["loss"])
    output["predicted_depth"].retain_grad()
    output["loss"].backward()
    assert q.grad is None
    assert all(torch.isfinite(m.grad).all() for m in maps)
    # Large position errors must keep a non-redescending correcting gradient.
    residual = torch.tensor([2., 20.], requires_grad=True)
    torch.nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="sum").backward()
    torch.testing.assert_close(residual.grad, torch.ones_like(residual))


def test_supplied_holdout_is_preserved_and_invalid_depth_does_not_poison_loss():
    torch.manual_seed(1)
    levels = _features()
    point_levels, point_masks = _point_maps(requires_grad=True)
    module = PolarDepthSelfSupervision(POLAR_CHANNELS, POINT_CHANNELS, anchor_depth_weight=.05).train()
    depth = torch.ones(1, 1, 1, 32, 32)
    depth[..., 0, 0] = float("nan")
    valid = torch.isfinite(depth)
    hidden = torch.zeros_like(valid)
    hidden[..., 12:20, 12:20] = True
    workspace = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    workspace[..., 8:24, 8:24] = True
    normal = torch.zeros(1, 1, 3, 32, 32)
    normal[:, :, 2] = -1
    k = torch.tensor([[[[100., 0, 16], [0, 100., 16], [0, 0, 1.]]]])
    output = module(levels, point_levels, point_masks, torch.zeros(1, 1, 7, 32, 32), k,
        depth, valid, normal_targets=normal, depth_supervision_mask=hidden, workspace_mask=workspace)
    assert torch.isfinite(output["loss"])
    error = torch.log(output["predicted_depth"])
    expected = module._cauchy(error)[hidden].mean()
    assert torch.allclose(output["holdout_depth_loss"], expected)
    assert not output["anchor_weights"][..., :8, :].any()
    output["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())


def _features(batch=1, views=1, height=32, width=32):
    return tuple(
        torch.randn(batch, views, channel, height // (2 ** level), width // (2 ** level))
        for level, channel in enumerate(POLAR_CHANNELS)
    )


def _point_maps(batch=1, views=1, height=32, width=32, requires_grad=False):
    maps = tuple(
        torch.randn(
            batch, views, channel, height // (2 ** level), width // (2 ** level),
            requires_grad=requires_grad,
        )
        for level, channel in enumerate(POINT_CHANNELS)
    )
    masks = tuple(torch.ones_like(level[:, :, :1], dtype=torch.bool) for level in maps)
    return maps, masks


def test_depth_normals_face_the_camera_and_are_differentiable():
    rows, cols = torch.meshgrid(torch.arange(9), torch.arange(11), indexing="ij")
    depth = (1.0 + 0.01 * cols + 0.02 * rows).float()[None, None]
    depth.requires_grad_()
    intrinsics = torch.tensor([[[80.0, 0.0, 5.0], [0.0, 80.0, 4.0], [0.0, 0.0, 1.0]]])
    normals, valid = depth_to_normals(depth, intrinsics)
    assert valid[:, :, 1:-1, 1:-1].all()
    # The viewing vector points approximately along -z at the principal point.
    assert normals[0, 2, 4, 5] < 0
    normals[valid.expand_as(normals)].sum().backward()
    assert torch.isfinite(depth.grad).all()


def test_point_feature_rasterizer_preserves_gradient_path():
    levels = _features(height=16, width=16)
    stage_points = []
    for channels in POINT_CHANNELS:
        feat = torch.randn(2, channels, requires_grad=True)
        stage_points.append({
            "feat": feat,
            "coord": torch.tensor([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]]),
            "batch": torch.zeros(2, dtype=torch.long),
        })
    intrinsics = torch.tensor([[[[100.0, 0.0, 7.5], [0.0, 100.0, 7.5], [0.0, 0.0, 1.0]]]])
    maps, masks = rasterize_fused_point_features(
        tuple(stage_points), levels, intrinsics,
        torch.eye(4)[None, None], torch.tensor([[[16, 16]]]),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert len(maps) == len(masks) == 5
    assert all(mask.any() for mask in masks)
    sum(level.sum() for level in maps).backward()
    assert all(stage["feat"].grad is not None for stage in stage_points)


def test_point_feature_rasterizer_uses_encoder_feature_offsets():
    levels = _features(height=16, width=16)
    stage_points = tuple({
        "feat": torch.ones(1, channels),
        "coord": torch.tensor([[3.0, 3.0, 1.0]]),
        "batch": torch.zeros(1, dtype=torch.long),
    } for channels in POINT_CHANNELS)
    common = dict(
        stage_points=stage_points,
        feature_levels=levels,
        intrinsics=torch.eye(3)[None, None],
        transforms=torch.eye(4)[None, None],
        image_hw=torch.tensor([[[16, 16]]]),
        view_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    _, tasknet_masks = rasterize_fused_point_features(
        **common,
        feature_strides=(1, 2, 4, 8, 16),
        feature_offsets=(0, 0, 0, 0, 0),
    )
    _, legacy_masks = rasterize_fused_point_features(**common)

    # At stride 2, TaskNet's padding keeps the center at zero, whereas the
    # legacy 2x2 MaxPool grid has a 0.5-pixel center offset.
    assert tasknet_masks[1][0, 0, 0, 2, 2]
    assert legacy_masks[1][0, 0, 0, 1, 1]
    assert not tasknet_masks[1][0, 0, 0, 1, 1]
    assert not legacy_masks[1][0, 0, 0, 2, 2]


def test_depth_targets_are_removed_before_pointact():
    points = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
        [0.02, 0.0, 1.0, 2.0],
        [-0.02, 0.0, 1.0, 3.0],
    ])
    target = torch.zeros(1, 1, 1, 9, 9, dtype=torch.bool)
    target[0, 0, 0, 4, 4] = True
    intrinsics = torch.tensor([[[[50.0, 0.0, 4.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]]]])
    kept, counts, keep_mask = mask_points_at_depth_targets(
        points, torch.tensor([3]), target, intrinsics,
        torch.eye(4)[None, None], torch.ones(1, 1, dtype=torch.bool),
    )
    assert keep_mask.tolist() == [False, True, True]
    assert counts.tolist() == [2]
    torch.testing.assert_close(kept, points[1:])


def test_depth_target_removal_matches_v2_floor_pixel_convention():
    points = torch.tensor([
        [1.75, 1.75, 1.0, 1.0],
        [0.25, 0.25, 1.0, 2.0],
    ])
    target = torch.zeros(1, 1, 1, 4, 4, dtype=torch.bool)
    target[0, 0, 0, 1, 1] = True
    kept, counts, keep_mask = mask_points_at_depth_targets(
        points,
        torch.tensor([2]),
        target,
        torch.eye(3)[None, None],
        torch.eye(4)[None, None],
        torch.ones(1, 1, dtype=torch.bool),
    )

    # V2 stores (1.75, 1.75) in floor cell (1, 1). Nearest rounding would
    # incorrectly look at (2, 2) and leak this held-out source point.
    assert keep_mask.tolist() == [False, True]
    assert counts.tolist() == [1]
    torch.testing.assert_close(kept, points[1:])


def test_point_depth_decoder_fuses_both_modalities_and_backpropagates():
    levels = tuple(level[:, 0] for level in _features())
    point_levels, point_valid = _point_maps(requires_grad=True)
    point_levels = tuple(level[:, 0].detach().requires_grad_() for level in point_levels)
    point_valid = tuple(level[:, 0] for level in point_valid)
    decoder = PolarPointDepthDecoder(
        POLAR_CHANNELS, POINT_CHANNELS, min_depth=0.05, max_depth=3.0
    )
    prediction, decoded = decoder(levels, point_levels, point_valid)
    assert prediction.shape == (1, 1, 32, 32)
    assert decoded.shape == (1, 32, 32, 32)
    assert prediction.min() >= 0.05 and prediction.max() <= 3.0
    prediction.mean().backward()
    assert decoder.point_projections[0].weight.grad is not None
    assert point_levels[0].grad is not None
    assert decoder.fuse5.layers[0].weight.grad is not None


def test_full_self_supervision_loss_is_finite_and_has_decoder_gradients():
    torch.manual_seed(7)
    height = width = 32
    levels = _features()
    point_levels, point_valid = _point_maps(requires_grad=True)
    polar = torch.zeros(1, 1, 7, height, width)
    polar[:, :, 0] = 0.5
    polar[:, :, 1] = 0.2
    polar[:, :, 2] = 1.0
    polar[:, :, 6] = 1.0
    intrinsics = torch.tensor([[[[60.0, 0.0, 15.5], [0.0, 60.0, 15.5], [0.0, 0.0, 1.0]]]])
    sparse = torch.zeros(1, 1, 1, height, width)
    sparse[:, :, :, 4::8, 4::8] = 1.0
    valid = sparse > 0
    normal_targets = torch.zeros(1, 1, 3, height, width)
    normal_targets[:, :, 2] = -1.0
    normal_targets.requires_grad_()
    objective = PolarDepthSelfSupervision(
        POLAR_CHANNELS, POINT_CHANNELS, min_depth=0.05, max_depth=3.0
    )
    result = objective(
        levels,
        point_levels,
        point_valid,
        polar,
        intrinsics,
        sparse,
        valid,
        pixel_valid=torch.ones(1, 1, height, width, dtype=torch.bool),
        view_valid=torch.ones(1, 1, dtype=torch.bool),
        normal_targets=normal_targets,
    )
    assert result["predicted_depth"].shape == sparse.shape
    assert result["predicted_normals"].shape == (1, 1, 3, height, width)
    torch.testing.assert_close(result["normal_targets"], result["sfp_normals"])
    for name in (
        "loss", "normal_consistency_loss",
        "sparse_depth_loss", "smoothness_loss",
    ):
        assert torch.isfinite(result[name])
    result["loss"].backward()
    assert objective.decoder.depth_head.weight.grad is not None
    assert all(level.grad is not None for level in point_levels)
    assert normal_targets.grad is None


def test_point_only_decoder_has_no_dense_polar_input_path():
    point_levels, point_valid = _point_maps(requires_grad=True)
    point_levels = tuple(level[:, 0].detach().requires_grad_() for level in point_levels)
    point_valid = tuple(level[:, 0] for level in point_valid)
    decoder = PolarPointDepthDecoder(
        POLAR_CHANNELS,
        POINT_CHANNELS,
        min_depth=0.05,
        max_depth=3.0,
        use_polar_features=False,
    )
    prediction, decoded = decoder(None, point_levels, point_valid)
    assert prediction.shape == (1, 1, 32, 32)
    assert decoded.shape == (1, 32, 32, 32)
    # Coarsest block receives only projected points (64) and validity (1).
    assert decoder.fuse5.layers[0].in_channels == 65
    prediction.mean().backward()
    assert point_levels[0].grad is not None
