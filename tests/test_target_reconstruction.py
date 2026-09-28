import torch

from pointact.model.ptv3.concerto.model_ca_action import GridUnpoolingWithAction
from pointact.model.ptv3.concerto.structure import Point
from pointact.model.vla_pointact.target_reconstruction import (
    VisibleTargetReconstructionHead,
    copy_point_tree,
)


class FakeSparse:
    def __init__(self, features):
        self.features = features

    def replace_feature(self, features):
        return FakeSparse(features)


def test_auxiliary_unpool_does_not_mutate_encoder_action_features():
    parent = Point(
        feat=torch.randn(5, 4, requires_grad=True),
        coord=torch.randn(5, 3),
        batch=torch.zeros(5, dtype=torch.long),
        action_feat=torch.randn(1, 2, 4, requires_grad=True),
    )
    parent.sparse_conv_feat = FakeSparse(parent.feat)
    encoded = Point(
        feat=torch.randn(2, 8, requires_grad=True),
        coord=torch.randn(2, 3),
        batch=torch.zeros(2, dtype=torch.long),
        action_feat=torch.randn(1, 2, 8, requires_grad=True),
        pooling_parent=parent,
        pooling_inverse=torch.tensor([0, 0, 1, 1, 1]),
    )
    original_point = parent.feat.clone()
    original_action = parent.action_feat.clone()
    unpool = GridUnpoolingWithAction(8, 4, 4)
    decoded = unpool(copy_point_tree(encoded))
    assert decoded.feat.shape == (5, 4)
    assert torch.equal(parent.feat, original_point)
    assert torch.equal(parent.action_feat, original_action)
    decoded.feat.sum().backward()
    assert parent.feat.grad is not None
    assert encoded.feat.grad is not None


def test_reconstruction_loss_uses_predicted_target_selection_and_gradients():
    head = VisibleTargetReconstructionHead(8, max_pred_points=3)
    features = torch.randn(7, 8, requires_grad=True)
    coords = torch.randn(7, 3)
    target = torch.randn(2, 5, 3)
    mask_loss, geometry_loss = head.loss(
        features, coords, torch.tensor([4, 7]),
        target, torch.tensor([5, 2]),
        torch.tensor([0, 1, 1, 0, 0, 1, 0], dtype=torch.float32),
    )
    assert mask_loss.isfinite()
    assert geometry_loss.isfinite()
    (mask_loss + geometry_loss).backward()
    assert features.grad is not None
    assert head.mask.weight.grad is not None
    assert head.delta.weight.grad is not None


def test_no_visible_target_skips_geometry_without_nan():
    head = VisibleTargetReconstructionHead(4)
    features = torch.randn(3, 4, requires_grad=True)
    mask_loss, geometry_loss = head.loss(
        features, torch.randn(3, 3), torch.tensor([3]),
        torch.zeros(1, 0, 3), torch.tensor([0]), torch.zeros(3),
    )
    assert mask_loss.isfinite()
    assert geometry_loss == 0
    (mask_loss + geometry_loss).backward()
    assert features.grad is not None


def test_geometry_never_fills_target_shortfall_with_background_points():
    head = VisibleTargetReconstructionHead(4, max_pred_points=3)
    with torch.no_grad():
        head.delta.weight.zero_()
        head.delta.bias.zero_()

    features = torch.randn(4, 4, requires_grad=True)
    coords = torch.tensor([
        [0.0, 0.0, 0.0],       # the only visible target input
        [10.0, 0.0, 0.0],      # unrelated background
        [0.0, 10.0, 0.0],      # unrelated background
        [0.0, 0.0, 10.0],      # unrelated background
    ])
    # The second row is padding and must also be ignored by target_counts.
    padded_target = torch.tensor([[[0.0, 0.0, 0.0], [999.0, 999.0, 999.0]]])
    mask_loss, geometry_loss = head.loss(
        features,
        coords,
        torch.tensor([4]),
        padded_target,
        torch.tensor([1]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )

    assert mask_loss.isfinite()
    assert geometry_loss == 0
    (mask_loss + geometry_loss).backward()
    assert features.grad is not None
    assert head.delta.weight.grad is not None


def test_geometry_skips_frame_when_no_target_input_is_visible():
    head = VisibleTargetReconstructionHead(4, max_pred_points=3)
    features = torch.randn(3, 4, requires_grad=True)
    mask_loss, geometry_loss = head.loss(
        features,
        torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]),
        torch.tensor([3]),
        torch.tensor([[[0.0, 0.0, 0.0]]]),
        torch.tensor([1]),
        torch.zeros(3),
    )

    assert mask_loss.isfinite()
    assert geometry_loss == 0
    (mask_loss + geometry_loss).backward()
    assert features.grad is not None
