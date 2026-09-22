"""Match dense RGB/polar regions to candidate materials and condition PTv3 points."""

import torch
from torch import nn
from torch.nn import functional as F

from pointact.data.polar_material import MATERIAL_FEATURE_DIM


class PolarMaterialConditioner(nn.Module):
    def __init__(self, point_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(7, hidden_channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1), nn.GELU(),
        )
        self.material_encoder = nn.Sequential(
            nn.Linear(MATERIAL_FEATURE_DIM, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.unknown_material = nn.Parameter(torch.zeros(1, 1, hidden_channels))
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels, 1), nn.GELU(),
            nn.Conv2d(hidden_channels, point_channels, 1),
        )
        self.scale = hidden_channels ** -0.5

    def forward(self, rgb, polar, material_candidates, point_pixel_indices,
                npoints_in_batch, material_candidate_mask=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or polar.shape != (rgb.shape[0], 4, *rgb.shape[-2:]):
            raise ValueError("Expected RGB [B,3,H,W] and polar [B,4,H,W] at the same resolution")
        if (material_candidates.ndim != 3 or material_candidates.shape[0] != rgb.shape[0]
                or material_candidates.shape[1] < 1
                or material_candidates.shape[2] != MATERIAL_FEATURE_DIM):
            raise ValueError(f"Expected material candidates [B,K,{MATERIAL_FEATURE_DIM}]")
        if npoints_in_batch.numel() != rgb.shape[0] or point_pixel_indices.numel() != int(npoints_in_batch.sum()):
            raise ValueError("Point pixel indices must be aligned with the concatenated points")
        height, width = rgb.shape[-2:]
        indices = point_pixel_indices.long()
        if torch.any(indices < 0) or torch.any(indices >= height * width):
            raise ValueError("Point pixel index outside dense RGB/polar image")

        image_features = self.image_encoder(torch.cat((rgb.float(), polar.float()), dim=1))
        material_features = self.material_encoder(material_candidates.float())
        material_features = torch.cat((material_features,
                                       self.unknown_material.expand(rgb.shape[0], -1, -1)), dim=1)
        logits = torch.einsum("bchw,bkc->bkhw", image_features, material_features) * self.scale
        if material_candidate_mask is not None:
            valid = torch.cat((material_candidate_mask.bool(),
                               torch.ones((rgb.shape[0], 1), dtype=torch.bool, device=rgb.device)), dim=1)
            logits = logits.masked_fill(~valid[:, :, None, None], torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=1)
        matched = torch.einsum("bkhw,bkc->bchw", weights, material_features)
        conditioned_map = self.fuse(torch.cat((image_features, matched), dim=1))

        offsets = torch.cat((npoints_in_batch.new_zeros(1), npoints_in_batch.cumsum(0)))
        per_point = []
        for batch_index in range(rgb.shape[0]):
            sample_indices = indices[offsets[batch_index]:offsets[batch_index + 1]]
            x = sample_indices.remainder(width).float()
            y = torch.div(sample_indices, width, rounding_mode="floor").float()
            grid = torch.stack(((x + 0.5) * (2 / width) - 1,
                                (y + 0.5) * (2 / height) - 1), dim=-1)
            sampled = F.grid_sample(conditioned_map[batch_index:batch_index + 1],
                                    grid.reshape(1, 1, -1, 2), align_corners=False)
            per_point.append(sampled[0, :, 0, :].T)
        return torch.cat(per_point, dim=0), logits
