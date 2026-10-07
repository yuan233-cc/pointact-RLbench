import copy

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from pointact.model.vla_pointact.action_head_3d import cga_dino_normal
from pointact.model.vla_pointact.action_head_3d.cga_dino_normal import (
    CgaDinoNormalNet,
    load_cga_dino_normal_checkpoint,
    load_dinov3_convnext_base,
    masked_cosine_normal_loss,
)


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Conv2d(3, 128, 1),
                nn.Conv2d(128, 256, 1),
                nn.Conv2d(256, 512, 1),
                nn.Conv2d(512, 1024, 1),
            ]
        )
        self.last_request = None

    def get_intermediate_layers(self, x, n, reshape):
        self.last_request = (n, reshape, torch.is_grad_enabled())
        outputs = []
        for stage in self.stages:
            x = F.avg_pool2d(x, 4 if not outputs else 2)
            x = stage(x)
            outputs.append(x)
        return tuple(outputs)


def test_frozen_cga_chunking_preserves_normals_and_feature_bank():
    from pointact.model.vla_pointact.workspace_geometry import frozen_cga_features

    torch.manual_seed(42)
    model = CgaDinoNormalNet(FakeDino(), observation_channels=7,
                           physical_prior_channels=11, transformer_blocks=0).eval()
    observation = torch.randn(2, 7, 64, 64)
    prior = torch.randn(2, 11, 64, 64)
    rgb = torch.rand(2, 3, 64, 64)
    chunked = frozen_cga_features(model, observation, prior, rgb, chunk_size=1)
    whole = frozen_cga_features(model, observation, prior, rgb, chunk_size=2)
    torch.testing.assert_close(chunked[0], whole[0], atol=1e-5, rtol=1e-4)
    for actual, expected in zip(chunked[1], whole[1], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        assert not actual.requires_grad
    assert not chunked[0].requires_grad


def test_dinov3_loader_uses_vendored_constructor_and_external_weights(monkeypatch, tmp_path):
    source = FakeDino()
    weights = tmp_path / "dinov3_convnext_base.pth"
    torch.save(source.state_dict(), weights)
    monkeypatch.setattr(cga_dino_normal, "dinov3_convnext_base", FakeDino)

    loaded = load_dinov3_convnext_base(weights)

    assert isinstance(loaded, FakeDino)
    for expected, actual in zip(source.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_cga_dino_normal_shapes_freezing_and_gradients():
    dino = FakeDino()
    model = CgaDinoNormalNet(
        dino,
        observation_channels=7,
        physical_prior_channels=11,
        transformer_blocks=1,
    ).train()
    observation = torch.randn(1, 7, 64, 64)
    prior = torch.randn(1, 11, 64, 64)
    rgb = torch.rand(1, 3, 64, 64)
    result = model(observation, prior, rgb)
    assert result["normal"].shape == (1, 3, 64, 64)
    assert [tuple(level.shape) for level in result["feature_levels"]] == [
        (1, 64, 64, 64),
        (1, 128, 32, 32),
        (1, 256, 16, 16),
        (1, 512, 8, 8),
        (1, 512, 4, 4),
    ]
    torch.testing.assert_close(
        torch.linalg.vector_norm(result["normal"], dim=1), torch.ones(1, 64, 64)
    )
    assert dino.last_request == ([0, 1, 2, 3], True, False)
    assert not dino.training
    loss = masked_cosine_normal_loss(
        result["normal"], torch.randn_like(result["normal"]), torch.ones(1, 1, 64, 64)
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in dino.parameters())
    assert model.cga_encoder.inc.double_conv[0].weight.grad is not None
    assert model.dino_projections[0].weight.grad is not None
    assert model.fuse5.block[0].weight.grad is not None
    assert model.normal_head.weight.grad is not None


def test_forward_features_skips_decoder_and_requires_real_prior():
    model = CgaDinoNormalNet(
        FakeDino(), observation_channels=7, physical_prior_channels=11, transformer_blocks=0
    ).eval()
    decoder_called = []
    hook = model.up1.register_forward_hook(lambda *args: decoder_called.append(True))
    levels = model.forward_features(
        torch.randn(1, 7, 64, 64),
        torch.randn(1, 11, 64, 64),
        torch.rand(1, 3, 64, 64),
    )
    hook.remove()
    assert len(levels) == 5
    assert not decoder_called
    try:
        model.forward_features(
            torch.randn(1, 7, 64, 64),
            torch.randn(1, 7, 64, 64),
            torch.rand(1, 3, 64, 64),
        )
    except ValueError as error:
        assert "physical prior" in str(error)
    else:
        raise AssertionError("A seven-channel fake prior must be rejected")


def test_compact_checkpoint_round_trip_is_exact(tmp_path):
    source = CgaDinoNormalNet(
        FakeDino(), observation_channels=7, physical_prior_channels=11, transformer_blocks=0
    ).eval()
    target = copy.deepcopy(source)
    with torch.no_grad():
        target.normal_head.weight.zero_()
    checkpoint = tmp_path / "normal.pt"
    torch.save({"model": source.trainable_state_dict()}, checkpoint)
    report = load_cga_dino_normal_checkpoint(target, checkpoint)
    assert report["missing"] == ()
    observation = torch.randn(1, 7, 64, 64)
    prior = torch.randn(1, 11, 64, 64)
    rgb = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = source(observation, prior, rgb)["normal"]
        restored = target(observation, prior, rgb)["normal"]
    torch.testing.assert_close(restored, expected)
