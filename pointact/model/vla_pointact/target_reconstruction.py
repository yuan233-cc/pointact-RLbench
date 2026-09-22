"""Training-only visible target surface supervision for the action PTv3 encoder."""

import torch
from torch import nn
from torch.nn import functional as F

from pointact.model.ptv3.concerto.structure import Point


def copy_point_tree(point):
    """Copy mutable Point containers while preserving tensors and their gradients.

    GridUnpoolingWithAction pops pooling_parent and rewrites parent features.
    Sharing those containers with the action path would change its output.
    """
    copied = Point({key: value for key, value in point.items() if key != "pooling_parent"})
    if "pooling_parent" in point:
        copied["pooling_parent"] = copy_point_tree(point.pooling_parent)
    return copied


class VisibleTargetReconstructionHead(nn.Module):
    """Select target points and correct their XYZ on the existing input support."""

    def __init__(self, channels, max_pred_points=256):
        super().__init__()
        if max_pred_points < 1:
            raise ValueError("max_pred_points must be positive")
        self.max_pred_points = max_pred_points
        self.shared = nn.Sequential(nn.Linear(channels, channels), nn.GELU())
        self.mask = nn.Linear(channels, 1)
        self.delta = nn.Linear(channels, 3)

    def forward(self, features, coords):
        hidden = self.shared(features)
        logits = self.mask(hidden).squeeze(-1)
        corrected = coords + 0.1 * torch.tanh(self.delta(hidden))
        return logits, corrected

    def loss(self, features, coords, offsets, target_points, target_counts, target_input_mask):
        if target_input_mask.shape != (len(features),):
            raise ValueError("Target mask must have one entry per decoded input point")
        if target_points.ndim != 3 or target_points.shape[-1] != 3:
            raise ValueError("Target points must be a padded [B, M, 3] tensor")
        if len(target_counts) != len(offsets) or len(target_points) != len(offsets):
            raise ValueError("Target counts and point offsets must match the batch")
        if target_counts.max().item() > target_points.shape[1]:
            raise ValueError("Target count exceeds padded target point length")

        logits, corrected = self(features, coords)
        labels = target_input_mask.to(logits.dtype)
        positives = labels.sum()
        negatives = labels.numel() - positives
        positive_weight = (negatives / positives.clamp_min(1)).clamp(1, 20)
        mask_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=positive_weight
        )

        geometric_losses = []
        start = 0
        for sample, (end, count) in enumerate(zip(offsets.tolist(), target_counts.tolist())):
            if count:
                sample_logits = logits[start:end]
                sample_coords = corrected[start:end]
                selected = torch.topk(
                    sample_logits, min(self.max_pred_points, len(sample_logits))
                ).indices
                gt = target_points[sample, :count].to(sample_coords.dtype)
                distances = torch.cdist(sample_coords[selected], gt).square()
                geometric_losses.append(
                    distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()
                )
            start = end
        geometry_loss = torch.stack(geometric_losses).mean() if geometric_losses else logits.sum() * 0
        return mask_loss, geometry_loss
