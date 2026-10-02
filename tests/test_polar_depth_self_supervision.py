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
    sfp_normals = torch.zeros(1, 1, 3, height, width)
    sfp_normals[:, :, 2] = -1.0
    sfp_normals.requires_grad_()
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
        sfp_normals=sfp_normals,
    )
    assert result["predicted_depth"].shape == sparse.shape
    assert result["predicted_normals"].shape == (1, 1, 3, height, width)
    for name in (
        "loss", "normal_consistency_loss",
        "sparse_depth_loss", "smoothness_loss",
    ):
        assert torch.isfinite(result[name])
    result["loss"].backward()
    assert objective.decoder.depth_head.weight.grad is not None
    assert all(level.grad is not None for level in point_levels)
    assert sfp_normals.grad is None
