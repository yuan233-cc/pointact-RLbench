"""CGA-Transformer image features for PointACT polar token routing.

The encoder core is adapted from the MIT-licensed CGA-Transformer project:
https://github.com/singobl/CGA-Transformer

The published network consumes two distinct application-specific branches.
The observation and physical-prior tensors are therefore always separate
arguments.  In robot compatibility mode independent stems only align their
channel dimensions; a stem never synthesizes candidate normals from the
observation branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "in"):
        super().__init__()
        if norm == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm == "in":
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unsupported CGA normalization {norm!r}")
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "in"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_channels, out_channels, norm=norm),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class _SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.sa(torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1))


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ca(self.gap(x))


class _PixelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pa2 = nn.Conv2d(
            channels * 2,
            channels,
            7,
            padding=3,
            padding_mode="reflect",
            groups=channels,
            bias=True,
        )

    def forward(self, x: Tensor, first_attention: Tensor) -> Tensor:
        paired = torch.stack((x, first_attention.expand_as(x)), dim=2)
        paired = paired.flatten(1, 2)
        return torch.sigmoid(self.pa2(paired))


class _CgaFusion(nn.Module):
    """Side-effect-free form of CGA-Transformer's content-guided fusion."""

    def __init__(self, channels: int):
        super().__init__()
        self.sa = _SpatialAttention()
        self.ca = _ChannelAttention(channels)
        self.pa = _PixelAttention(channels)
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, signal: Tensor, physical_prior: Tensor) -> Tensor:
        initial = signal + physical_prior
        first_attention = self.sa(initial) + self.ca(initial)
        gate = self.pa(initial, first_attention)
        return self.conv(initial + gate * signal + (1.0 - gate) * physical_prior)


class _Mlp(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels * 4)
        self.fc2 = nn.Linear(channels * 4, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class _Attention(nn.Module):
    def __init__(self, channels: int, heads: int = 8, head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.query = nn.Linear(channels, inner)
        self.key = nn.Linear(channels, inner)
        self.value = nn.Linear(channels, inner)
        self.out = nn.Linear(inner, channels)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[:-1], self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        query = self._split_heads(self.query(x))
        key = self._split_heads(self.key(x))
        value = self._split_heads(self.value(x))
        attention = (query @ key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attention = self.attn_dropout(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).flatten(2)
        return self.proj_dropout(self.out(output))


class _TransformerBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.attention_norm = nn.LayerNorm(channels, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(channels, eps=1e-6)
        self.attn = _Attention(channels, dropout=dropout)
        self.ffn = _Mlp(channels, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class CgaTransformerFeatureEncoder(nn.Module):
    """Return five feature grids from genuine CGA observation/prior branches."""

    feature_channels = (64, 128, 256, 512, 512)
    cga_channels = 11

    def __init__(
        self,
        residual_num: int = 8,
        dropout: float = 0.0,
        observation_channels: int = 11,
        physical_prior_channels: int = 11,
        fusion_channels: int = 11,
    ):
        super().__init__()
        if residual_num < 0:
            raise ValueError("residual_num must be non-negative")
        if min(observation_channels, physical_prior_channels, fusion_channels) <= 0:
            raise ValueError("CGA branch channel counts must be positive")
        self.observation_channels = observation_channels
        self.physical_prior_channels = physical_prior_channels
        self.fusion_channels = fusion_channels
        self.observation_stem = self._make_stem(observation_channels, fusion_channels)
        self.physical_prior_stem = self._make_stem(
            physical_prior_channels, fusion_channels
        )
        self.fusion = _CgaFusion(fusion_channels)
        # Match the released CGA implementation: the first block uses its
        # BatchNorm default while down1..down4 receive norm='in'.
        self.inc = _DoubleConv(fusion_channels, 64, norm="bn")
        self.down1 = _Down(64, 128, norm="in")
        self.down2 = _Down(128, 256, norm="in")
        self.down3 = _Down(256, 512, norm="in")
        self.down4 = _Down(512, 512, norm="in")
        self.resblock_layers = nn.ModuleList(
            _TransformerBlock(512, dropout=dropout) for _ in range(residual_num)
        )

    @staticmethod
    def _make_stem(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward_features(
        self, observation: Tensor, physical_prior: Tensor
    ) -> tuple[Tensor, ...]:
        if observation.ndim != 4 or observation.shape[1] != self.observation_channels:
            raise ValueError(
                "CGA observation must be [N,"
                f"{self.observation_channels},H,W], got {tuple(observation.shape)}"
            )
        if physical_prior.ndim != 4 or physical_prior.shape[1] != self.physical_prior_channels:
            raise ValueError(
                "CGA physical prior must be [N,"
                f"{self.physical_prior_channels},H,W], got {tuple(physical_prior.shape)}"
            )
        if observation.shape[0] != physical_prior.shape[0] or observation.shape[-2:] != physical_prior.shape[-2:]:
            raise ValueError("CGA observation and physical prior must be batch/spatially aligned")
        if min(observation.shape[-2:]) < 32:
            raise ValueError("CGA polar input height and width must both be at least 32")
        signal = self.observation_stem(observation)
        prior = self.physical_prior_stem(physical_prior)
        fused = self.fusion(signal, prior)
        x1 = self.inc(fused)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        batch, channels, height, width = x5.shape
        tokens = x5.flatten(2).transpose(1, 2)
        for block in self.resblock_layers:
            tokens = block(tokens)
        x5 = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return x1, x2, x3, x4, x5

    def forward(
        self, observation: Tensor, physical_prior: Tensor
    ) -> tuple[Tensor, ...]:
        return self.forward_features(observation, physical_prior)


def _checkpoint_state_dict(checkpoint) -> Mapping[str, Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise TypeError("CGA checkpoint must contain a state_dict mapping")


def load_cga_transformer_checkpoint(
    model: CgaTransformerFeatureEncoder,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    """Load all shape-compatible CGA encoder weights from a released checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"CGA-Transformer checkpoint not found: {path}")
    source = _checkpoint_state_dict(torch.load(path, map_location="cpu"))
    target = model.state_dict()
    compatible = {}
    skipped = []
    for raw_name, value in source.items():
        name = raw_name
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name in target and isinstance(value, Tensor) and value.shape == target[name].shape:
            compatible[name] = value
        elif name in target:
            skipped.append(name)
    core_names = ("fusion.", "inc.", "down1.", "down2.", "down3.", "down4.")
    if not any(name.startswith(core_names) for name in compatible):
        raise ValueError("CGA checkpoint has no compatible fusion or encoder weights")
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "path": str(path),
        "loaded": len(compatible),
        "skipped_shape": tuple(skipped),
        "missing": tuple(missing),
        "unexpected": tuple(unexpected),
    }
