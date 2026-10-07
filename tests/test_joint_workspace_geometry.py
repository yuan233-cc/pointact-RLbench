import torch
from unittest.mock import patch
from pointact.model.vla_pointact.joint_workspace_geometry import (
    native_cga_observation, prepare_workspace_points,
)
from pointact.model.vla_pointact.action_head_3d.polar_depth_self_supervision import PolarDepthSelfSupervision


def test_native_cga_layout_integer_rays_and_stokes():
    images = torch.zeros(1, 1, 7, 8, 8)
    images[:, :, 0] = .8
    images[:, :, 1] = .5
    images[:, :, 2] = 1.
    k = torch.tensor([[[[10., 0., 4.], [0., 10., 4.], [0., 0., 1.]]]])
    observation = native_cga_observation(images, k)
    assert observation.shape == (1, 11, 8, 8)
    torch.testing.assert_close(observation[0, :4, 4, 4], torch.tensor([.6, .4, .2, .4]))
    torch.testing.assert_close(observation[0, -3:, 4, 4], torch.tensor([0., 0., 1.]))
    assert observation[0, -3, 4, 5] > 0
    assert observation[0, -3, 4, 3] < 0


def test_point_filter_keeps_frame_and_original_sensor_targets():
    depth = torch.ones(1, 1, 1, 8, 8)
    points = torch.randn(3, 9)
    counts = torch.tensor([3])
    context = dict(polar_workspace_mask=torch.ones(1, 1, 8, 8, dtype=torch.bool),
        view_valid=torch.ones(1, 1, dtype=torch.bool), polar_K=torch.eye(3)[None, None],
        point_pixel_indices=torch.tensor([20, 21, 22]))
    normal = torch.zeros(1, 1, 3, 8, 8)
    normal[:, :, 2] = -1.
    confidence = torch.zeros_like(depth)
    confidence.flatten()[20] = .5
    confidence.flatten()[22] = 1.
    with patch("pointact.model.vla_pointact.joint_workspace_geometry.normal_anchor_confidence",
               return_value=confidence[:, 0]):
        selected, selected_counts, keep, updated = prepare_workspace_points(
            points, counts, context, depth, torch.ones_like(depth, dtype=torch.bool), normal)
    torch.testing.assert_close(selected, points[[0, 2]])
    assert selected_counts.tolist() == [2]
    assert updated["point_pixel_indices"].tolist() == [20, 22]
    assert context["point_pixel_indices"].tolist() == [20, 21, 22]
    assert torch.equal(depth, torch.ones_like(depth))
    assert not updated["observation_confidence"].requires_grad


def test_hole_normal_weight_and_finite_gradients():
    module = PolarDepthSelfSupervision((2,) * 5, (2,) * 5,
        use_polar_features=False, supervision_mode="weighted_workspace",
        pixel_center_offset=0., smoothness_weight=0., hole_normal_weight=3.)
    maps = tuple(torch.randn(1, 1, 2, s, s, requires_grad=True) for s in (16, 8, 4, 2, 1))
    masks = tuple(torch.ones(1, 1, 1, s, s, dtype=torch.bool) for s in (16, 8, 4, 2, 1))
    depth = torch.ones(1, 1, 1, 32, 32)
    depth[..., 12:20, 12:20] = float("nan")
    valid = torch.isfinite(depth)
    workspace = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    workspace[..., :4, :] = False
    normals = torch.zeros(1, 1, 3, 32, 32)
    normals[:, :, 2] = -1.
    k = torch.tensor([[[[100., 0., 16.], [0., 100., 16.], [0., 0., 1.]]]])
    out = module(None, maps, masks, torch.zeros(1, 1, 7, 32, 32), k, depth, valid,
        normal_targets=normals, workspace_mask=workspace,
        observation_confidence=torch.zeros_like(depth))
    weights = out["normal_supervision_weights"]
    assert (weights[..., 12:20, 12:20] == 3.).all()
    assert (weights[..., 6:10, 6:10] == 1.).all()
    assert (weights[..., :4, :] == 0.).all()
    assert (out["point_fit_weights"][..., 12:20, 12:20] == 0.).all()
    # Finite rejected observations remain weight 1, not hole weight 3.
    assert weights[..., 8, 8].item() == 1.
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(m.grad is not None and torch.isfinite(m.grad).all() for m in maps)


def test_action_joint_forward_backward_cuda():
    """Real small Concerto/action/decoder; only the frozen image teacher is mocked."""
    if not torch.cuda.is_available():
        return
    from torch import nn
    from pointact.model.vla_pointact.configuration_pointact import VLAEncDec3DModelConfig
    from pointact.model.vla_pointact.modeling_vla_pointact import VLAEncDec3DWithActionClassificationModel
    from pointact.model.vla_pointact.action_head_3d.ptv3_backbone import PointTransformerUnetWithAction
    from pointact.model.vla_pointact.action_head_3d.classification_head import PointWithActionCenteredClassificationMLPActionHead

    channels = (32, 64, 96, 128, 128)

    class Teacher(nn.Module):
        feature_strides = (1, 2, 4, 8, 16)
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.), requires_grad=False)
            self.calls = 0
        def forward(self, observation, prior, rgb):
            self.calls += 1
            assert observation.shape[1] == 11 and not torch.is_grad_enabled()
            normal = observation.new_zeros((len(observation), 3, 128, 128))
            normal[:, 2] = -1
            levels = tuple(observation[:, :1, ::2**i, ::2**i].expand(-1, c, -1, -1).contiguous()
                           for i, c in enumerate(channels))
            return dict(normal=normal, feature_levels=levels)

    cfg = VLAEncDec3DModelConfig(polar_enabled=True, polar_backbone="cga_dinov3_normal",
        cga_dino_normal_checkpoint="unused.pt", cga_dino_use_dino=False,
        cga_freeze=True, cga_dino_input_mode="native_cga", polar_fusion_mode="workspace",
        use_polar_depth_self_supervision=True, polar_depth_supervision_mode="weighted_workspace",
        use_robot_state=False, use_target_reconstruction=False, action_head_pos_center="zero",
        action_chunk_size=2, ctx_embed_size=16, ptv3_enc_channels=channels,
        ptv3_enc_depths=(1,) * 5, polar_workspace_attend_action=True)
    model = VLAEncDec3DWithActionClassificationModel.__new__(VLAEncDec3DWithActionClassificationModel)
    nn.Module.__init__(model)
    model.config = cfg
    model.cga_dino_encoder = Teacher().eval()
    model.ctx_proj = nn.Linear(16, 16)
    model.position_embedding = nn.Embedding(2, channels[0])
    model.ptv3_model = PointTransformerUnetWithAction(input_size=9, ctx_embed_size=16,
        enc_channels=channels, enc_depths=(1,) * 5, enc_num_head=(2, 4, 6, 8, 8),
        patch_size=8, enc_mode=True, apply_point_ca=False, polar_enabled=True,
        sfp_feature_channels=channels, polar_fusion_mode="workspace",
        polar_bbox_feature_levels=(0, 1, 2, 3, 4), polar_workspace_attend_action=True)
    model.polar_depth_self_supervision = PolarDepthSelfSupervision(channels, channels,
        use_polar_features=False, supervision_mode="weighted_workspace", pixel_center_offset=0.,
        hole_normal_weight=3., smoothness_weight=0.)
    model.completion_action_projection = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, channels[-1]))
    nn.init.zeros_(model.completion_action_projection[-1].weight)
    nn.init.zeros_(model.completion_action_projection[-1].bias)
    model.action_head = PointWithActionCenteredClassificationMLPActionHead(
        channels[-1], 2, dropout=0., euler_resolution=5, pos_bins=100, pos_bin_size=.01)
    model.cuda().train()
    images = torch.zeros(2, 1, 7, 128, 128, device="cuda")
    images[:, :, 0] = .8
    images[:, :, 1] = .2
    images[:, :, 2] = 1.
    depth = torch.ones(2, 1, 1, 128, 128, device="cuda")
    depth[..., 58:70, 58:70] = float("nan")
    k = torch.tensor([[[100., 0., 64.], [0., 100., 64.], [0., 0., 1.]]], device="cuda").repeat(2, 1, 1)[:, None]
    yy, xx = torch.meshgrid(torch.arange(20, 100, 10, device="cuda"),
                           torch.arange(20, 100, 10, device="cuda"), indexing="ij")
    pixels = (yy * 128 + xx).flatten().repeat(2)
    points = torch.rand(128, 9, device="cuda")
    points[:, :3] = torch.stack(((xx.flatten().repeat(2) - 64) / 100,
                                (yy.flatten().repeat(2) - 64) / 100, torch.ones(128, device="cuda")), 1)
    kwargs = dict(polar_images=images, polar_physical_prior=torch.zeros(2, 1, 11, 128, 128, device="cuda"),
        polar_K=k, T_camera_from_model=torch.eye(4, device="cuda")[None, None].repeat(2, 1, 1, 1),
        view_valid=torch.ones(2, 1, device="cuda", dtype=torch.bool),
        pixel_valid=torch.ones(2, 1, 128, 128, device="cuda", dtype=torch.bool),
        polar_workspace_mask=torch.ones(2, 1, 128, 128, device="cuda", dtype=torch.bool),
        point_pixel_indices=pixels, observed_depth=depth, observed_depth_valid=torch.isfinite(depth))
    actions = torch.zeros(2, 2, 7, device="cuda")
    actions[..., 6] = 1
    with patch("pointact.model.vla_pointact.modeling_vla_pointact.structured_depth_holdout",
               side_effect=AssertionError("Joint weighted mode must never hold out depth")):
        loss, parts, _, geometry = model.compute_action_loss(points, torch.tensor([64, 64], device="cuda"),
            torch.empty(2, 0, 16, device="cuda"), torch.zeros(2, device="cuda", dtype=torch.long),
            None, actions, torch.zeros(2, 2, device="cuda", dtype=torch.bool), **kwargs)
        assert torch.isfinite(loss) and torch.isfinite(geometry["loss"])
        (loss + geometry["loss"]).backward()
        assert model.position_embedding.weight.grad is not None
        assert model.position_embedding.weight.grad.abs().sum() > 0
        assert model.cga_dino_encoder.scale.grad is None
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_head.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.polar_depth_self_supervision.decoder.parameters())
        assert any("polar" in n and p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.ptv3_model.named_parameters())
        assert model.cga_dino_encoder.calls == 1
        assert (geometry["normal_supervision_weights"][..., 60:68, 60:68] == 3.).all()
        model.eval()
        with torch.no_grad():
            predicted = model.compute_action(points, torch.tensor([64, 64], device="cuda"),
                torch.empty(2, 0, 16, device="cuda"), torch.zeros(2, device="cuda", dtype=torch.long),
                None, **kwargs)
        assert predicted.shape == (2, 2, 7) and torch.isfinite(predicted).all()
