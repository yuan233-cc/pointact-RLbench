import tempfile

import torch

from pointact.model.vla_pointact.action_head_3d.cga_transformer_encoder import (
    CgaTransformerFeatureEncoder,
    load_cga_transformer_checkpoint,
)


def test_cga_encoder_returns_router_compatible_pyramid():
    encoder = CgaTransformerFeatureEncoder(residual_num=1).eval()
    with torch.no_grad():
        levels = encoder.forward_features(torch.randn(2, 7, 64, 96))
    assert [tuple(level.shape) for level in levels] == [
        (2, 64, 64, 96),
        (2, 128, 32, 48),
        (2, 256, 16, 24),
        (2, 512, 8, 12),
        (2, 512, 4, 6),
    ]


def test_cga_checkpoint_loader_accepts_original_encoder_names():
    source = CgaTransformerFeatureEncoder(residual_num=1)
    target = CgaTransformerFeatureEncoder(residual_num=1)
    checkpoint = {
        "state_dict": {
            f"module.{name}": value.clone()
            for name, value in source.state_dict().items()
            if not name.startswith(("signal_adapter.", "physical_prior_adapter."))
        }
    }
    with tempfile.NamedTemporaryFile(suffix=".pth") as handle:
        torch.save(checkpoint, handle.name)
        report = load_cga_transformer_checkpoint(target, handle.name)
    assert report["loaded"] > 0
    assert torch.equal(target.inc.double_conv[0].weight, source.inc.double_conv[0].weight)
