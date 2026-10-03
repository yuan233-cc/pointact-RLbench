# Third-party notices

## DINOv3

P-VLA includes the minimal DINOv3 ConvNeXt source required by the CGA+DINOv3
normal encoder under `pointact/third_party/dinov3`. It is vendored from
`facebookresearch/dinov3` commit
`6876159a11b4df116f30f667f8c9888617df0751`.

That source is governed by the DINOv3 License Agreement reproduced at
`pointact/third_party/dinov3/LICENSE.md`. DINOv3 model weights are not included.

## CGA-Transformer

The adapted CGA-Transformer encoder is derived from the MIT-licensed
`singobl/CGA-Transformer` project. Its license is reproduced at
`pointact/model/vla_pointact/action_head_3d/CGA_TRANSFORMER_LICENSE.md`.

## PolarAPP

The optional task-aware polarization encoder under
`pointact/model/vla_pointact/action_head_3d/polarapp_tasknet_encoder.py` adapts
the SfP TaskNet blocks from the MIT-licensed `roydon-luo/PolarAPP` project,
commit `1f2b4b86eb8e2b3a863a7fb53162b87fac5a3860`. Its license is reproduced at
`pointact/model/vla_pointact/action_head_3d/POLARAPP_LICENSE.md`. PolarAPP model
weights and datasets are not included.
