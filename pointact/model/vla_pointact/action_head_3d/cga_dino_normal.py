"""CGA + frozen DINOv3 ConvNeXt dense surface-normal network.

The CGA observation and physical-prior branches are deliberately separate.
For native CGA data both contain 11 channels.  Robot compatibility mode uses
a seven-channel observation plus an independently precomputed physical prior;
the 1x1 stems only align channels for fusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from pointact.third_party.dinov3 import dinov3_convnext_base

from .cga_transformer_encoder import CgaTransformerFeatureEncoder, _DoubleConv

DINO_CHANNELS = (128, 256, 512, 1024)
FEATURE_CHANNELS = (64, 128, 256, 512, 512)


class _Up(nn.Module):
    def __init__(self, decoder_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # The released CGA Up block keeps its BatchNorm default.
        self.conv = _DoubleConv(decoder_channels + skip_channels, out_channels, norm="bn")

    def forward(self, decoder: Tensor, skip: Tensor) -> Tensor:
        decoder = F.interpolate(
            decoder, size=skip.shape[-2:], mode="bilinear", align_corners=True
        )
        return self.conv(torch.cat((skip, decoder), dim=1))


class _Fuse(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


def load_dinov3_convnext_base(weights_path: str | Path) -> nn.Module:
    """Load the vendored DINOv3 ConvNeXt-Base with external weights."""
    weights_path = Path(weights_path).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 ConvNeXt-Base weights not found: {weights_path}")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in state.items()
    ):
        raise TypeError("DINOv3 weights must be a plain tensor state dict")
    model = dinov3_convnext_base()
    model.load_state_dict(state, strict=True)
    return model


class CgaDinoNormalNet(nn.Module):
    """Predict normals and expose router-compatible fused feature levels."""

    feature_channels = FEATURE_CHANNELS
    feature_strides = (1, 2, 4, 8, 16)

    def __init__(
        self,
        dino: nn.Module | None,
        *,
        observation_channels: int = 11,
        physical_prior_channels: int = 11,
        fusion_channels: int = 11,
        transformer_blocks: int = 8,
        transformer_dropout: float = 0.0,
        use_dino: bool = True,
    ):
        super().__init__()
        if use_dino and dino is None:
            raise ValueError("use_dino=True requires a DINOv3 backbone")
        self.use_dino = use_dino
        self.cga_encoder = CgaTransformerFeatureEncoder(
            residual_num=transformer_blocks,
            dropout=transformer_dropout,
            observation_channels=observation_channels,
            physical_prior_channels=physical_prior_channels,
            fusion_channels=fusion_channels,
        )
        self.dino = dino
        if self.dino is not None:
            self.dino.requires_grad_(False)
            self.dino.eval()

        if use_dino:
            self.dino_projections = nn.ModuleList(
                nn.Conv2d(channels, 128, kernel_size=1) for channels in DINO_CHANNELS
            )
            self.fuse3 = _Fuse(256 + 128, 256)
            self.fuse4 = _Fuse(512 + 128, 512)
            self.fuse5 = _Fuse(512 + 128 + 128, 512)
        else:
            self.dino_projections = nn.ModuleList()
            self.fuse3 = nn.Identity()
            self.fuse4 = nn.Identity()
            self.fuse5 = nn.Identity()

        self.up1 = _Up(512, 512, 256)
        self.up2 = _Up(256, 256, 128)
        self.up3 = _Up(128, 128, 64)
        self.up4 = _Up(64, 64, 64)
        self.normal_head = nn.Conv2d(64, 3, kernel_size=1)
        self.register_buffer(
            "rgb_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    @classmethod
    def from_dinov3(
        cls,
        weights_path: str | Path,
        **kwargs,
    ) -> "CgaDinoNormalNet":
        return cls(load_dinov3_convnext_base(weights_path), **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.dino is not None:
            self.dino.eval()
        return self

    def _dino_features(self, rgb: Tensor) -> tuple[Tensor, ...]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"RGB must be [N,3,H,W], got {tuple(rgb.shape)}")
        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("DINOv3 RGB must be finite and scaled to [0,1]")
        rgb = rgb.to(dtype=self.rgb_mean.dtype)
        rgb_normalized = (rgb - self.rgb_mean) / self.rgb_std
        with torch.no_grad():
            levels = self.dino.get_intermediate_layers(
                rgb_normalized, n=[0, 1, 2, 3], reshape=True
            )
        if len(levels) != 4:
            raise RuntimeError(f"DINOv3 returned {len(levels)} levels instead of four")
        for index, (level, channels) in enumerate(zip(levels, DINO_CHANNELS, strict=True)):
            if level.ndim != 4 or level.shape[1] != channels:
                raise RuntimeError(
                    f"DINOv3 level {index} must be [N,{channels},H,W], got {tuple(level.shape)}"
                )
        return tuple(levels)

    def _encode(
        self,
        polar_observation: Tensor,
        physical_prior: Tensor,
        rgb: Tensor | None,
    ) -> tuple[Tensor, ...]:
        e1, e2, e3, e4, e5 = self.cga_encoder.forward_features(
            polar_observation, physical_prior
        )
        if not self.use_dino:
            return e1, e2, self.fuse3(e3), self.fuse4(e4), self.fuse5(e5)
        if rgb is None:
            raise ValueError("CGA+DINO feature extraction requires aligned RGB")
        if rgb.shape[0] != polar_observation.shape[0] or rgb.shape[-2:] != polar_observation.shape[-2:]:
            raise ValueError("RGB and polarization must be batch/spatially aligned")
        d1, d2, d3, d4 = self._dino_features(rgb)
        projected = tuple(
            projection(level.to(dtype=projection.weight.dtype))
            for projection, level in zip(self.dino_projections, (d1, d2, d3, d4), strict=True)
        )
        p1, p2, p3, p4 = projected
        expected_sizes = (e3.shape[-2:], e4.shape[-2:], e5.shape[-2:])
        if (p1.shape[-2:], p2.shape[-2:], p3.shape[-2:]) != expected_sizes:
            raise RuntimeError(
                "DINOv3 native stages do not align with CGA at 1/4, 1/8, 1/16: "
                f"got {[p1.shape[-2:], p2.shape[-2:], p3.shape[-2:]]}, expected {list(expected_sizes)}"
            )
        f3 = self.fuse3(torch.cat((e3, p1), dim=1))
        f4 = self.fuse4(torch.cat((e4, p2), dim=1))
        p4 = F.interpolate(p4, size=e5.shape[-2:], mode="bilinear", align_corners=False)
        f5 = self.fuse5(torch.cat((e5, p3, p4), dim=1))
        return e1, e2, f3, f4, f5

    def forward_features(
        self,
        polar_observation: Tensor,
        physical_prior: Tensor,
        rgb: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Return ``(E1, E2, F3, F4, F5)`` without running the decoder."""
        return self._encode(polar_observation, physical_prior, rgb)

    def forward(
        self,
        polar_observation: Tensor,
        physical_prior: Tensor,
        rgb: Tensor | None = None,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        e1, e2, f3, f4, f5 = self._encode(polar_observation, physical_prior, rgb)
        z4 = self.up1(f5, f4)
        z3 = self.up2(z4, f3)
        z2 = self.up3(z3, e2)
        z1 = self.up4(z2, e1)
        normal = F.normalize(self.normal_head(z1), dim=1, eps=1e-6)
        return {
            "normal": normal,
            "feature_levels": (e1, e2, f3, f4, f5),
            "dense_feature": z1,
        }

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Return a compact checkpoint without the frozen DINO parameters."""
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("dino.")
        }


def load_cga_dino_normal_checkpoint(
    model: CgaDinoNormalNet,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, object]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"CGA+DINO normal checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state: Mapping[str, Tensor]
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("model"), Mapping):
        state = checkpoint["model"]
    elif isinstance(checkpoint, Mapping):
        state = checkpoint
    else:
        raise TypeError("Normal checkpoint must be a state-dict mapping")
    cleaned = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            continue
        while name.startswith(("module.", "model.")):
            name = name.split(".", 1)[1]
        cleaned[name] = value
    incompatible = model.load_state_dict(cleaned, strict=False)
    missing_non_dino = tuple(
        name for name in incompatible.missing_keys if not name.startswith("dino.")
    )
    if strict and (missing_non_dino or incompatible.unexpected_keys):
        raise RuntimeError(
            "Incompatible CGA+DINO normal checkpoint: "
            f"missing={missing_non_dino}, unexpected={tuple(incompatible.unexpected_keys)}"
        )
    return {
        "path": str(path),
        "loaded": len(cleaned),
        "missing": missing_non_dino,
        "unexpected": tuple(incompatible.unexpected_keys),
    }


def masked_cosine_normal_loss(
    prediction: Tensor, target: Tensor, valid_mask: Tensor
) -> Tensor:
    """Masked mean ``1-cos`` after normalizing prediction and target."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("prediction and target must both be [N,3,H,W]")
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, None]
    if valid_mask.shape != prediction.shape[:1] + (1,) + prediction.shape[-2:]:
        raise ValueError("valid_mask must be [N,1,H,W] or [N,H,W]")
    prediction = F.normalize(prediction, dim=1, eps=1e-6)
    target_finite = torch.isfinite(target).all(dim=1, keepdim=True)
    target_norm = torch.linalg.vector_norm(target, dim=1, keepdim=True)
    valid = valid_mask.bool() & target_finite & (target_norm > 1e-6)
    target = F.normalize(torch.nan_to_num(target), dim=1, eps=1e-6)
    cosine = (prediction * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    weights = valid.to(cosine.dtype)
    return ((1.0 - cosine) * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def normal_metrics(
    prediction: Tensor, target: Tensor, valid_mask: Tensor
) -> dict[str, Tensor]:
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, None]
    prediction = F.normalize(prediction, dim=1, eps=1e-6)
    target_norm = torch.linalg.vector_norm(torch.nan_to_num(target), dim=1, keepdim=True)
    valid = valid_mask.bool() & torch.isfinite(target).all(dim=1, keepdim=True) & (target_norm > 1e-6)
    target = F.normalize(torch.nan_to_num(target), dim=1, eps=1e-6)
    degrees = torch.rad2deg(
        torch.acos((prediction * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0))
    )
    count = valid.sum()
    denominator = count.clamp_min(1).to(degrees.dtype)
    return {
        "angle_sum": (degrees * valid).sum(),
        "within_11_25": ((degrees < 11.25) & valid).sum(),
        "within_22_5": ((degrees < 22.5) & valid).sum(),
        "valid_count": count,
        "mae": (degrees * valid).sum() / denominator,
        "pct_11_25": ((degrees < 11.25) & valid).sum() / denominator,
        "pct_22_5": ((degrees < 22.5) & valid).sum() / denominator,
    }
