# CGA-Transformer polar tokens with Utonia PointACT

This is an additional mode. The existing SfP-Wild + Concerto/Utonia modes and
the point-cloud-only defaults are unchanged.

Use `--polar_backbone cga_transformer --ptv3_backend utonia`. The image path
accepts the same calibrated seven-channel tensor as the SfP-Wild experiment:

```text
[I_un, DoLP, cos(2 AoLP), sin(2 AoLP), view_x, view_y, view_z]
```

Two learned 1x1 adapters produce the eleven-channel signal and physical-prior
branches expected by CGA. The side-effect-free CGA fusion block, U-Net encoder,
and bottleneck transformer are adapted from the MIT-licensed
`singobl/CGA-Transformer` repository. They expose feature maps with channels
`[64, 128, 256, 512, 512]` at strides `[1, 2, 4, 8, 16]`.

Those shapes match the existing SfP-Wild feature contract. Consequently, the
calibrated point-to-feature projection, local polar-token selection, joint
`[action/state, point, polar]` attention, point-feature rasterization, dense
depth decoder, and all reconstruction/consistency losses are reused without a
new mapping convention.

The upstream CGA repository does not include a released checkpoint in this
workspace. The supplied launcher therefore makes from-scratch training an
explicit choice (`CGA_ALLOW_RANDOM_INIT=True`, `CGA_FREEZE=False`). If a
compatible checkpoint becomes available, set `CGA_CHECKPOINT`; the loader
imports all shape-compatible fusion, encoder, and transformer weights while
leaving the PointACT-specific seven-to-eleven-channel adapters trainable.

Launch with:

```bash
PTV3_INIT_CKPT_FILE=/path/to/utonia.pth \
REPORT_TO=wandb \
bash experiments/10_rlbench/train_10task_polar_rlbench9_v2_cga_utonia.sh
```

The matching private dataset repository is
`yuan1119/pointact-cga-utonia-rlbench10-v1`. Its ZIP retains the validated
RLBench9-v2 data and documents that `I_un` is an RGB-luminance proxy rather
than corrected physical S0.
