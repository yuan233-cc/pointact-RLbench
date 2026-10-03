"""PolarAPP SfP TaskNet features adapted for PointACT Polar routing.

The TaskNet blocks are derived from the MIT-licensed PolarAPP SfP release:
https://github.com/roydon-luo/PolarAPP

PointACT uses the three task-aware outputs produced by PolarAPP's decoder
(``TaF1``, ``TaF2``, and ``TaF3``), including the released normal prediction
head, then adds a lightweight FPN bridge that returns five genuine spatial
scales for the five PTv3 encoder stages.  The bridge is PointACT-specific and
is not part of the released PolarAPP model.
"""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class _CheckpointConv2d(nn.Module):
    """Conv wrapper whose names match PolarAPP's ``MetaConv2d.conv`` keys."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(value)


class _WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, value: Tensor) -> Tensor:
        mean = value.mean(-1, keepdim=True)
        variance = value.var(-1, keepdim=True, unbiased=False)
        return (value - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class _LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = _WithBiasLayerNorm(channels)

    def forward(self, value: Tensor) -> Tensor:
        batch, channels, height, width = value.shape
        tokens = value.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        tokens = self.body(tokens)
        return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class _FeedForward(nn.Module):
    def __init__(self, channels: int, expansion: float, bias: bool):
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = _CheckpointConv2d(channels, hidden * 2, 1, bias=bias)
        self.dwconv = _CheckpointConv2d(
            hidden * 2,
            hidden * 2,
            3,
            stride=1,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = _CheckpointConv2d(hidden, channels, 1, bias=bias)

    def forward(self, value: Tensor) -> Tensor:
        first, second = self.dwconv(self.project_in(value)).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class _Attention(nn.Module):
    def __init__(self, channels: int, heads: int, bias: bool):
        super().__init__()
        if channels % heads:
            raise ValueError("PolarAPP TaskNet channels must be divisible by heads")
        self.num_heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = _CheckpointConv2d(channels, channels * 3, 1, bias=bias)
        self.qkv_dwconv = _CheckpointConv2d(
            channels * 3,
            channels * 3,
            3,
            stride=1,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.project_out = _CheckpointConv2d(channels, channels, 1, bias=bias)

    def forward(self, value: Tensor) -> Tensor:
        batch, channels, height, width = value.shape
        qkv = self.qkv_dwconv(self.qkv(value))
        query, key, val = qkv.chunk(3, dim=1)
        head_channels = channels // self.num_heads

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                batch, self.num_heads, head_channels, height * width
            )

        query = F.normalize(split_heads(query), dim=-1)
        key = F.normalize(split_heads(key), dim=-1)
        val = split_heads(val)
        attention = (query @ key.transpose(-2, -1)) * self.temperature
        output = attention.softmax(dim=-1) @ val
        output = output.reshape(batch, channels, height, width)
        return self.project_out(output)


class _TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, expansion: float, bias: bool):
        super().__init__()
        self.norm1 = _LayerNorm2d(channels)
        self.attn = _Attention(channels, heads, bias)
        self.norm2 = _LayerNorm2d(channels)
        self.ffn = _FeedForward(channels, expansion, bias)

    def forward(self, value: Tensor) -> Tensor:
        value = value + self.attn(self.norm1(value))
        return value + self.ffn(self.norm2(value))


class _ImageFeatureExtractor(nn.Module):
    def __init__(self, input_channels: int, channels: int, bias: bool):
        super().__init__()
        self.proj = nn.Sequential(
            _CheckpointConv2d(
                input_channels, channels, 3, stride=1, padding=1, bias=bias
            ),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.proj(value)


class _DownsampleConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = _CheckpointConv2d(
            channels, channels * 2, 3, stride=2, padding=1, bias=False
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.body(value)


class _UpsampleInter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = _CheckpointConv2d(
            channels, channels // 2, 3, stride=1, padding=1, bias=False
        )

    def forward(self, value: Tensor, size: tuple[int, int]) -> Tensor:
        value = F.interpolate(value, size=size, mode="nearest")
        return self.body(value)


class _RefinementDown(nn.Module):
    """Released TaskNet normal refinement head with checkpoint-compatible names."""

    def __init__(self, channels: int):
        super().__init__()
        self.refinement = nn.Sequential(
            _CheckpointConv2d(channels, channels, 3, stride=1, padding=1),
            nn.ReLU(),
            _CheckpointConv2d(channels, channels, 3, stride=1, padding=1),
            nn.ReLU(),
            _CheckpointConv2d(channels, channels, 3, stride=1, padding=1),
            nn.ReLU(),
            _CheckpointConv2d(channels, channels // 2, 3, stride=1, padding=1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.refinement(value)


class PolarAppTaskAwareEncoder(nn.Module):
    """Return a five-level FPN built from PolarAPP's task-aware decoder features.

    ``input_mode='sfp_proxy'`` converts PointACT's existing SfP-Wild observation
    ``[I_un, DoLP, cos(2AoLP), sin(2AoLP), vx, vy, vz]`` into the PolarAPP
    TaskNet layout ``[S0, DoP, sin(2AoP), cos(2AoP), x, y, 1]``.  ``I_un`` is
    necessarily an S0 proxy because the current PointACT archive does not carry
    PolarAPP DemNet's four reconstructed analyzer images.

    ``input_mode='tasknet7'`` accepts an already prepared PolarAPP TaskNet tensor
    without conversion.
    """

    feature_strides = (1, 2, 4, 8, 16)
    feature_offsets = (0.0, 0.0, 0.0, 0.0, 0.0)

    def __init__(
        self,
        input_mode: str = "sfp_proxy",
        pyramid_channels: int = 128,
        dim: int = 48,
        num_blocks: tuple[int, int, int] = (4, 4, 4),
        heads: tuple[int, int, int] = (1, 2, 4),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()
        if input_mode not in ("sfp_proxy", "tasknet7"):
            raise ValueError("polarapp_input_mode must be 'sfp_proxy' or 'tasknet7'")
        if pyramid_channels <= 0:
            raise ValueError("polarapp_pyramid_channels must be positive")
        if len(num_blocks) != 3 or len(heads) != 3:
            raise ValueError("PolarAPP TaskNet requires three block/head stages")
        self.input_mode = input_mode
        self.pyramid_channels = pyramid_channels
        self.feature_channels = (pyramid_channels,) * 5

        # Keep the released TaskNet names so its checkpoint can be loaded without
        # modifying the source archive.
        self.img_feature_extractor = _ImageFeatureExtractor(7, dim, bias)
        self.encoder1 = nn.Sequential(
            *(
                _TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
                for _ in range(num_blocks[0])
            )
        )
        self.down1 = _DownsampleConv(dim)
        self.encoder2 = nn.Sequential(
            *(
                _TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
                for _ in range(num_blocks[1])
            )
        )
        self.down2 = _DownsampleConv(dim * 2)
        self.bottleneck = nn.Sequential(
            *(
                _TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
                for _ in range(num_blocks[2])
            )
        )
        self.up3 = _UpsampleInter(dim * 4)
        self.reduce_chan3 = _CheckpointConv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder2 = nn.Sequential(
            *(
                _TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
                for _ in range(num_blocks[1])
            )
        )
        self.up2 = _UpsampleInter(dim * 2)
        self.decoder1 = nn.Sequential(
            *(
                _TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias)
                for _ in range(num_blocks[0])
            )
        )

        # Released TaskNet normal head.  Keeping the original module names lets
        # ``TaskNet.pth`` load this branch directly.
        self.refinement = _RefinementDown(dim * 2)
        self.output = _CheckpointConv2d(
            dim, 3, 3, stride=1, padding=1, bias=bias
        )

        # PointACT-only 3 -> 5 task-aware feature pyramid.
        self.pyramid_lateral1 = nn.Conv2d(dim * 2, pyramid_channels, 1)
        self.pyramid_lateral2 = nn.Conv2d(dim * 2, pyramid_channels, 1)
        self.pyramid_lateral3 = nn.Conv2d(dim * 4, pyramid_channels, 1)
        self.pyramid_smooth1 = nn.Sequential(
            nn.Conv2d(pyramid_channels, pyramid_channels, 3, padding=1), nn.GELU()
        )
        self.pyramid_smooth2 = nn.Sequential(
            nn.Conv2d(pyramid_channels, pyramid_channels, 3, padding=1), nn.GELU()
        )
        self.pyramid_smooth3 = nn.Sequential(
            nn.Conv2d(pyramid_channels, pyramid_channels, 3, padding=1), nn.GELU()
        )
        self.pyramid_down4 = nn.Sequential(
            nn.Conv2d(pyramid_channels, pyramid_channels, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pyramid_down5 = nn.Sequential(
            nn.Conv2d(pyramid_channels, pyramid_channels, 3, stride=2, padding=1),
            nn.GELU(),
        )

    def _tasknet_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.img_feature_extractor,
            self.encoder1,
            self.down1,
            self.encoder2,
            self.down2,
            self.bottleneck,
            self.up3,
            self.reduce_chan3,
            self.decoder2,
            self.up2,
            self.decoder1,
            self.refinement,
            self.output,
        )

    def _normal_head_modules(self) -> tuple[nn.Module, ...]:
        return self.refinement, self.output

    def set_tasknet_trainable(self, trainable: bool) -> None:
        for module in self._tasknet_modules():
            module.requires_grad_(trainable)
            if not trainable:
                module.eval()

    def set_tasknet_training(self, training: bool) -> None:
        for module in self._tasknet_modules():
            module.train(training)

    def set_normal_head_trainable(self, trainable: bool) -> None:
        for module in self._normal_head_modules():
            module.requires_grad_(trainable)
            if not trainable:
                module.eval()

    def set_normal_head_training(self, training: bool) -> None:
        for module in self._normal_head_modules():
            module.train(training)

    @staticmethod
    def _normalized_coordinates(reference: Tensor) -> Tensor:
        batch, _, height, width = reference.shape
        denominator_x = max(0.5 * width, 1.0)
        denominator_y = max(0.5 * height, 1.0)
        x = (
            torch.arange(width, device=reference.device, dtype=reference.dtype)
            - 0.5 * width
        ) / denominator_x
        y = (
            torch.arange(height, device=reference.device, dtype=reference.dtype)
            - 0.5 * height
        ) / denominator_y
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy, torch.ones_like(xx)), dim=0)
        return coordinates.unsqueeze(0).expand(batch, -1, -1, -1)

    def prepare_tasknet_input(self, observation: Tensor) -> Tensor:
        if observation.ndim != 4 or observation.shape[1] != 7:
            raise ValueError(
                f"PolarAPP observation must be [N,7,H,W], got {tuple(observation.shape)}"
            )
        if self.input_mode == "tasknet7":
            return observation
        coordinates = self._normalized_coordinates(observation)
        return torch.cat(
            (
                observation[:, 0:1],  # I_un used as the documented S0 proxy.
                observation[:, 1:2],
                observation[:, 3:4],  # sin(2AoLP)
                observation[:, 2:3],  # cos(2AoLP)
                coordinates,
            ),
            dim=1,
        )

    def forward_task_features(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        value = self.prepare_tasknet_input(observation)
        if min(value.shape[-2:]) < 4:
            raise ValueError("PolarAPP TaskNet input height and width must both be at least 4")

        x1 = self.encoder1(self.img_feature_extractor(value))
        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        x3_up = self.up3(x3, size=x2.shape[-2:])
        x2_dec = self.decoder2(self.reduce_chan3(torch.cat((x3_up, x2), dim=1)))
        x2_up = self.up2(x2_dec, size=x1.shape[-2:])
        x1_dec = self.decoder1(torch.cat((x2_up, x1), dim=1))
        return x1_dec, x2_dec, x3

    def build_pyramid(
        self, task_features: tuple[Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, ...]:
        """Build PointACT's five scales without recomputing TaskNet."""
        task1, task2, task3 = task_features
        level3 = self.pyramid_smooth3(self.pyramid_lateral3(task3))
        level2 = self.pyramid_smooth2(
            self.pyramid_lateral2(task2)
            + F.interpolate(
                level3,
                size=task2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        level1 = self.pyramid_smooth1(
            self.pyramid_lateral1(task1)
            + F.interpolate(
                level2,
                size=task1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        level4 = self.pyramid_down4(level3)
        level5 = self.pyramid_down5(level4)
        return level1, level2, level3, level4, level5

    def decode_normals(
        self,
        task_features: tuple[Tensor, Tensor, Tensor],
        normalize: bool = True,
        output_frame: str = "sfp_wild",
    ) -> Tensor:
        """Decode TaskNet normals in either native or depth-loss coordinates.

        The released head uses the opposite direction for all three axes from
        the V2 SfP-Wild comparison frame ``(+left,+down,+forward)``.  This was
        verified against the rendered V2 normal sidecars, so the shared loss
        frame is obtained with ``(-nx,-ny,-nz)``.
        """
        normals = self.output(self.refinement(task_features[0]))
        if normalize:
            normals = F.normalize(normals, dim=1, eps=1e-6)
        if output_frame == "tasknet":
            return normals
        if output_frame == "sfp_wild":
            return -normals
        raise ValueError("output_frame must be 'tasknet' or 'sfp_wild'")

    def forward_features(self, observation: Tensor) -> tuple[Tensor, ...]:
        return self.build_pyramid(self.forward_task_features(observation))

    def forward(self, observation: Tensor) -> tuple[Tensor, ...]:
        return self.forward_features(observation)


def _checkpoint_state_dict(checkpoint) -> Mapping[str, Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise TypeError("PolarAPP TaskNet checkpoint must contain a state_dict mapping")


def load_polarapp_tasknet_checkpoint(
    model: PolarAppTaskAwareEncoder,
    checkpoint_path: str | Path,
    require_normal_head: bool = False,
) -> dict[str, object]:
    """Load released TaskNet weights while leaving the five-level FPN initialized."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"PolarAPP TaskNet checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    source = _checkpoint_state_dict(checkpoint)
    target = model.state_dict()
    compatible: dict[str, Tensor] = {}
    skipped_shape: list[str] = []
    unused: list[str] = []
    for raw_name, value in source.items():
        name = raw_name
        for prefix in ("module.", "model.", "task_model.", "TaskNet."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name in target and isinstance(value, Tensor):
            if value.shape == target[name].shape:
                compatible[name] = value
            else:
                skipped_shape.append(name)
        else:
            unused.append(name)

    required_prefixes = (
        "img_feature_extractor.",
        "encoder1.",
        "down1.",
        "encoder2.",
        "down2.",
        "bottleneck.",
        "up3.",
        "reduce_chan3.",
        "decoder2.",
        "up2.",
        "decoder1.",
    )
    if require_normal_head:
        required_prefixes += ("refinement.", "output.")
    missing_components = [
        prefix for prefix in required_prefixes
        if not any(name.startswith(prefix) for name in compatible)
    ]
    if missing_components:
        raise ValueError(
            "PolarAPP checkpoint is missing compatible TaskNet components: "
            + ", ".join(missing_components)
        )
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "path": str(path),
        "loaded": len(compatible),
        "skipped_shape": tuple(skipped_shape),
        "unused_source_count": len(unused),
        "missing": tuple(missing),
        "unexpected": tuple(unexpected),
        "pyramid_randomly_initialized": True,
        "normal_head_loaded": all(
            any(name.startswith(prefix) for name in compatible)
            for prefix in ("refinement.", "output.")
        ),
    }
