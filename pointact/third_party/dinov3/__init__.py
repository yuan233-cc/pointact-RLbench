"""Minimal vendored DINOv3 ConvNeXt implementation used by CGA fusion.

The implementation is copied from facebookresearch/dinov3 at commit
6876159a11b4df116f30f667f8c9888617df0751. See ``LICENSE.md`` in this
directory for the governing DINOv3 License Agreement.
"""

from .convnext import ConvNeXt


def dinov3_convnext_base() -> ConvNeXt:
    """Construct the official DINOv3 ConvNeXt-Base architecture."""
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
