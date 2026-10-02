"""SfP-Wild feature extraction and optional normal decoding.

The architecture in this file is derived from the MIT-licensed SfP-Wild
implementation (Copyright (c) 2022 Chenyang LEI):
https://github.com/ChenyangLEI/sfp-wild

The released encoder, bottleneck transformer, U-Net decoder, and normal head
are instantiated with checkpoint-compatible names.  PointACT's Polar path only
calls :meth:`forward_features`, so adding the decoder does not add inference
work unless normal prediction is explicitly requested.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn


class _DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        norm: str = "in",
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        if norm == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm == "in":
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unsupported SfP-Wild normalization {norm!r}")
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            _DoubleConv(in_channels, out_channels, norm=norm),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class _Up(nn.Module):
    """Checkpoint-compatible SfP-Wild U-Net upsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        norm: str = "bn",
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _DoubleConv(
                in_channels,
                out_channels,
                mid_channels=in_channels // 2,
                norm=norm,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = _DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        diff_y = x2.shape[-2] - x1.shape[-2]
        diff_x = x2.shape[-1] - x1.shape[-1]
        x1 = torch.nn.functional.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        return self.conv(torch.cat((x2, x1), dim=1))


class _OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _Mlp(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(torch.nn.functional.gelu(self.fc1(x))))


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.query = nn.Linear(dim, inner)
        self.key = nn.Linear(dim, inner)
        self.value = nn.Linear(dim, inner)
        self.out = nn.Linear(inner, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _heads(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[:-1], self.heads, self.dim_head).permute(0, 2, 1, 3)

    def forward(self, x: Tensor) -> Tensor:
        query = self._heads(self.query(x))
        key = self._heads(self.key(x))
        value = self._heads(self.value(x))
        weights = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(self.dim_head), dim=-1)
        result = self.attn_dropout(weights) @ value
        result = result.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.proj_dropout(self.out(result))


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int = 512, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = _Mlp(hidden_size, dropout=dropout)
        self.attn = _Attention(hidden_size, dropout=dropout)
        self.drop_path = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop_path(self.attn(self.attention_norm(x))) + x
        return self.drop_path(self.ffn(self.ffn_norm(x))) + x


class SfpWildFeatureEncoder(nn.Module):
    """Official ``onlyiun_pol_vd`` network with reusable feature scales.

    Input is ``[I_un, DoLP, cos(2*AoLP), sin(2*AoLP), vx, vy, vz]``. SfP-Wild
    loads I_un and DoLP from its ``polar.npy`` without additional normalization,
    and AoLP is in radians. Callers must preserve those units.
    """

    feature_channels = (64, 128, 256, 512, 512)

    def __init__(
        self,
        in_channels: int = 7,
        residual_num: int = 8,
        norm: str = "in",
        dropout: float = 0.0,
        bilinear: bool = True,
        normal_channels: int = 3,
    ):
        super().__init__()
        if in_channels != 7:
            raise ValueError("The supported SfP-Wild onlyiun_pol_vd checkpoint requires 7 input channels")
        if not bilinear:
            raise ValueError("The released onlyiun_pol_vd checkpoint requires bilinear=True")
        # The released implementation does not forward ``--norm in`` to inc;
        # consequently its checkpoint has BatchNorm (including running stats)
        # in the first DoubleConv and InstanceNorm in down1..down4.
        self.inc = _DoubleConv(in_channels, 64, norm="bn")
        self.down1 = _Down(64, 128, norm=norm)
        self.down2 = _Down(128, 256, norm=norm)
        self.down3 = _Down(256, 512, norm=norm)
        self.down4 = _Down(512, 512, norm=norm)
        self.resblock_layers = nn.ModuleList(
            [_TransformerBlock(512, dropout=dropout) for _ in range(residual_num)]
        )
        # The released TransUnet does not pass its encoder normalization choice
        # into Up, so all four decoder blocks use Up's BatchNorm default.
        factor = 2
        self.up1 = _Up(1024, 512 // factor, bilinear=bilinear, norm="bn")
        self.up2 = _Up(512, 256 // factor, bilinear=bilinear, norm="bn")
        self.up3 = _Up(256, 128 // factor, bilinear=bilinear, norm="bn")
        self.up4 = _Up(128, 64, bilinear=bilinear, norm="bn")
        self.outc = _OutConv(64, normal_channels)

    def set_normal_decoder_trainable(self, trainable: bool) -> None:
        """Enable or freeze the decoder independently from the feature encoder."""
        for module in (self.up1, self.up2, self.up3, self.up4, self.outc):
            module.requires_grad_(trainable)
            if not trainable:
                module.eval()

    def set_normal_decoder_training(self, training: bool) -> None:
        """Set decoder mode without changing the shared feature encoder mode."""
        for module in (self.up1, self.up2, self.up3, self.up4, self.outc):
            module.train(training)

    def forward_features(self, polar_images: Tensor) -> tuple[Tensor, ...]:
        if polar_images.ndim != 4 or polar_images.shape[1] != 7:
            raise ValueError(f"polar_images must be [N,7,H,W], got {tuple(polar_images.shape)}")
        if min(polar_images.shape[-2:]) < 16:
            raise ValueError("SfP-Wild input height and width must both be at least 16")
        x1 = self.inc(polar_images)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        batch, channels, height, width = x5.shape
        tokens = x5.flatten(2).transpose(1, 2)
        for block in self.resblock_layers:
            # SfP-Wild's released full-model configuration uses skip_res=True;
            # each block already contains its own two residual connections.
            tokens = block(tokens)
        x5 = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return x1, x2, x3, x4, x5

    def forward(self, polar_images: Tensor) -> tuple[Tensor, ...]:
        return self.forward_features(polar_images)

    def decode_normals(
        self,
        feature_levels: tuple[Tensor, ...],
        normalize: bool = False,
    ) -> Tensor:
        """Decode ``x1..x5`` into camera-frame normal predictions.

        Raw logits are returned by default to match SfP-Wild's ``TransUnet``;
        its training code applies L2 normalization before computing the loss.
        """
        if len(feature_levels) != 5:
            raise ValueError("feature_levels must contain x1, x2, x3, x4, x5")
        x1, x2, x3, x4, x5 = feature_levels
        decoded = self.up1(x5, x4)
        decoded = self.up2(decoded, x3)
        decoded = self.up3(decoded, x2)
        decoded = self.up4(decoded, x1)
        normals = self.outc(decoded)
        if normalize:
            normals = torch.nn.functional.normalize(normals, p=2, dim=1)
        return normals

    def forward_with_normals(
        self,
        polar_images: Tensor,
        normalize_normals: bool = False,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        """Return the five feature levels and the decoded normal image."""
        feature_levels = self.forward_features(polar_images)
        normals = self.decode_normals(feature_levels, normalize=normalize_normals)
        return feature_levels, normals


def _checkpoint_state(checkpoint) -> Mapping[str, Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError("SfP checkpoint must contain a state_dict mapping")
    state = {}
    for key, value in checkpoint.items():
        if not isinstance(value, Tensor):
            continue
        while key.startswith("module.") or key.startswith("model."):
            key = key.split(".", 1)[1]
        state[key] = value
    return state


def load_sfp_wild_checkpoint(
    model: SfpWildFeatureEncoder,
    checkpoint_path: str | Path,
    require_decoder: bool = False,
) -> dict:
    """Load an official full checkpoint or a previously extracted encoder.

    Encoder tensors are always required. If any decoder tensor is present, the
    complete decoder must be present as well. ``require_decoder=True`` can be
    used by normal-supervised training to reject encoder-only checkpoints.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"SfP-Wild checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = _checkpoint_state(raw)
    expected = model.state_dict()
    decoder_prefixes = ("up1.", "up2.", "up3.", "up4.", "outc.")
    decoder_keys = {key for key in expected if key.startswith(decoder_prefixes)}
    encoder_keys = set(expected) - decoder_keys
    provided_keys = set(state) & set(expected)
    provided_decoder = provided_keys & decoder_keys
    missing_encoder = sorted(encoder_keys - provided_keys)
    missing_decoder = sorted(decoder_keys - provided_keys)
    mismatched = sorted(
        key for key in provided_keys if state[key].shape != expected[key].shape
    )
    partial_decoder = bool(provided_decoder) and bool(missing_decoder)
    if missing_encoder or mismatched or partial_decoder or (require_decoder and missing_decoder):
        raise RuntimeError(
            "SfP-Wild checkpoint is incompatible; "
            f"missing_encoder={missing_encoder}, missing_decoder={missing_decoder}, "
            f"shape_mismatch={mismatched}"
        )
    loadable = {key: state[key] for key in provided_keys}
    incompatible = model.load_state_dict(loadable, strict=False)
    ignored = sorted(set(state) - provided_keys)
    return {
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": ignored,
        "decoder_loaded": not missing_decoder,
    }


def viewing_directions_from_K(K: Tensor, height: int, width: int) -> Tensor:
    """Generate SfP-Wild viewing directions from real image intrinsics.

    The released ``vd_local.npy`` uses +x toward image-left, +y toward
    image-bottom and +z forward. This is the x-negated OpenCV camera ray.
    ``K`` must describe the actual (already undistorted) image geometry.
    """
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K must end in [3,3], got {tuple(K.shape)}")
    if not torch.isfinite(K).all() or (K[..., 0, 0] <= 0).any() or (K[..., 1, 1] <= 0).any():
        raise ValueError("K must be finite with positive focal lengths")
    rows = torch.arange(height, device=K.device, dtype=K.dtype)
    cols = torch.arange(width, device=K.device, dtype=K.dtype)
    vv, uu = torch.meshgrid(rows, cols, indexing="ij")
    lead = K.shape[:-2]
    uu = uu.expand(*lead, height, width)
    vv = vv.expand(*lead, height, width)
    x = (K[..., 0, 2, None, None] - uu) / K[..., 0, 0, None, None]
    y = (vv - K[..., 1, 2, None, None]) / K[..., 1, 1, None, None]
    rays = torch.stack((x, y, torch.ones_like(x)), dim=-3)
    return torch.nn.functional.normalize(rays, dim=-3)


def assemble_onlyiun_pol_vd(
    i_un: Tensor,
    dolp: Tensor,
    aolp_radians: Tensor,
    K: Tensor,
) -> Tensor:
    """Assemble the official seven channels without changing input units."""
    if i_un.shape != dolp.shape or i_un.shape != aolp_radians.shape:
        raise ValueError("I_un, DoLP and AoLP must have identical [B,V,1,H,W] shapes")
    if i_un.ndim != 5 or i_un.shape[2] != 1:
        raise ValueError("I_un, DoLP and AoLP must be [B,V,1,H,W]")
    if K.shape != (*i_un.shape[:2], 3, 3):
        raise ValueError("K must be [B,V,3,3] and match the image batch/views")
    rays = viewing_directions_from_K(K, i_un.shape[-2], i_un.shape[-1])
    phi = torch.cat((torch.cos(2 * aolp_radians), torch.sin(2 * aolp_radians)), dim=2)
    return torch.cat((i_un, dolp, phi, rays), dim=2)
