# Vendored DINOv3 ConvNeXt

This directory contains the minimal official DINOv3 source required by the
CGA+DINOv3 normal encoder. It is vendored from
[`facebookresearch/dinov3`](https://github.com/facebookresearch/dinov3) commit
`6876159a11b4df116f30f667f8c9888617df0751` (2026-07-15).

`convnext.py` is copied unchanged. `__init__.py` only exposes the official
ConvNeXt-Base dimensions used by P-VLA. Model weights are intentionally not
included and must be supplied separately at runtime.

This source is governed by the DINOv3 License Agreement in `LICENSE.md`, not
the repository's Apache-2.0 license.
