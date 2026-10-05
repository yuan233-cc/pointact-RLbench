#!/usr/bin/env python3
"""Convert Hugging Face DINOv3 ConvNeXt weights to the vendored PyTorch model."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

from safetensors.torch import load_file
import torch

from pointact.third_party.dinov3 import dinov3_convnext_base


EXPECTED_SHA256 = "ec90bd798b5fc5b8e30443796a6c24a7a73e28ad85c6c0ceda78b1d249a694cc"


def convert_name(name: str) -> str:
    if name.startswith("layer_norm."):
        return "norm." + name.removeprefix("layer_norm.")
    match = re.fullmatch(r"stages\.(\d+)\.downsample_layers\.(\d+)\.(weight|bias)", name)
    if match:
        stage, layer, parameter = match.groups()
        return f"downsample_layers.{stage}.{layer}.{parameter}"
    match = re.fullmatch(
        r"stages\.(\d+)\.layers\.(\d+)\."
        r"(depthwise_conv|layer_norm|pointwise_conv1|pointwise_conv2|gamma)"
        r"(?:\.(weight|bias))?",
        name,
    )
    if match:
        stage, layer, module, parameter = match.groups()
        module = {
            "depthwise_conv": "dwconv",
            "layer_norm": "norm",
            "pointwise_conv1": "pwconv1",
            "pointwise_conv2": "pwconv2",
            "gamma": "gamma",
        }[module]
        suffix = f".{parameter}" if parameter is not None else ""
        return f"stages.{stage}.{layer}.{module}{suffix}"
    raise ValueError(f"unrecognized Hugging Face tensor name: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    if destination.exists() or destination.with_suffix(destination.suffix + ".partial").exists():
        raise SystemExit("destination already exists; inspect before retrying")

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise SystemExit("source safetensors SHA-256 mismatch")

    original = load_file(str(source), device="cpu")
    converted = {}
    for name, tensor in original.items():
        mapped = convert_name(name)
        if mapped in converted:
            raise ValueError(f"duplicate mapped tensor: {mapped}")
        converted[mapped] = tensor
    # The vendored model registers the same final LayerNorm twice, as `norm`
    # and `norms.3`; PyTorch's state_dict consequently expects both names.
    converted["norms.3.weight"] = converted["norm.weight"]
    converted["norms.3.bias"] = converted["norm.bias"]
    model = dinov3_convnext_base()
    model.load_state_dict(converted, strict=True)
    del model, original

    partial = destination.with_suffix(destination.suffix + ".partial")
    torch.save(converted, partial)
    reloaded = torch.load(partial, map_location="cpu", weights_only=True)
    model = dinov3_convnext_base()
    model.load_state_dict(reloaded, strict=True)
    partial.rename(destination)
    print(f"converted {len(converted)} tensors to {destination}")
    print(f"output bytes={destination.stat().st_size}")


if __name__ == "__main__":
    main()
