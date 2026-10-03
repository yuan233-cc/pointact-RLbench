# PolarAPP task-aware hierarchy for PointACT

This optional Polar backbone uses the SfP TaskNet from PolarAPP and preserves
its task-aware decoder features. Existing `sfp_wild`, `cga_transformer`, and
`cga_dinov3_normal` configurations are unchanged.

## Architecture

PolarAPP returns three task features:

- `TaF1 = x1_dec`: full resolution, 96 channels;
- `TaF2 = x2_dec`: half resolution, 96 channels;
- `TaF3 = x3`: quarter resolution, 192 channels.

The PointACT-specific FPN combines these into five feature banks at strides
`(1, 2, 4, 8, 16)`. Each PTv3 stage keeps its existing stage-specific linear
adapter and local calibrated routing. PolarAPP's stride-2 convolutions have
input-pixel center offset zero, so the routing context explicitly carries
offsets `(0, 0, 0, 0, 0)` instead of using SfP-Wild's max-pooling offsets.

The released TaskNet can be frozen while the new FPN remains trainable.
Its released `refinement + output` normal head is also restored from the
checkpoint, rather than reconstructing normals from the FPN.

## Required options

```bash
--polar_enabled True \
--polar_backbone polarapp_taskaware \
--polarapp_checkpoint /path/to/PolarAPP/SfP/TaskNet/TaskNet.pth \
--polarapp_freeze True \
--polarapp_pyramid_channels 128 \
--polarapp_input_mode sfp_proxy
```

`polarapp_allow_random_init=True` is available only for explicit smoke tests.
Normal training requires the released TaskNet checkpoint.

## Input modes

`sfp_proxy` preserves compatibility with PointACT's existing seven-channel
SfP archives. It converts

```text
[I_un, DoLP, cos(2AoLP), sin(2AoLP), vx, vy, vz]
```

to

```text
[I_un-as-S0-proxy, DoP, sin(2AoP), cos(2AoP), normalized_x, normalized_y, 1].
```

This does **not** reproduce PolarAPP's DemNet. The current PointACT archive
does not contain its four full-resolution reconstructed analyzer images, so
`I_un` is an explicit S0 proxy and the input distribution differs from the
released TaskNet's training pipeline.

`tasknet7` performs no conversion and requires `polar_images` to already use
the exact PolarAPP TaskNet seven-channel layout above. Camera calibration and
`polar_pixel_transform` must refer to the same image grid.

The `sfp_proxy` conversion is implemented in
`PolarAppTaskAwareEncoder.prepare_tasknet_input`; no offline archive rewrite is
required. The calibrated `(vx,vy,vz)` rays remain available to PointACT's
routing metadata but are intentionally replaced at the TaskNet input by the
normalized image coordinates used by the released PolarAPP model.

## Joint depth and normal consistency

TaskNet now supports the same PointACT depth-completion branch as SfP-Wild:

1. `TaF1/TaF2/TaF3` are converted into five raster feature levels.
2. The five PTv3 encoder stages are projected onto the matching image grids.
3. `PolarPointDepthDecoder` fuses both sources and predicts dense metric depth.
4. The depth map is differentiated into pinhole-camera normals.
5. Those normals are compared with the detached normal prediction from the
   released TaskNet normal head.

Enable it with:

```bash
--use_polar_depth_self_supervision True \
--polar_consistency_weight 1.0 \
--sparse_depth_consistency_weight 1.0 \
--depth_smoothness_weight 0.01
```

For the current RLBench V2 sidecars, the ready-to-run launcher is
`experiments/10_rlbench/train_10task_polar_rlbench9_v2_tasknet_depth.sh`.
It defaults to the local released `TaskNet.pth`; override it with
`TASKNET_CHECKPOINT=/other/path/TaskNet.pth` when needed.

The classification-head variant uses the same polar/depth branch:

```bash
bash experiments/10_rlbench/train_10task_polar_rlbench9_v2_tasknet_depth_classify_h200.sh
```

Its defaults target one 140 GiB H200: batch size 32, 192 FPN channels, 64
local polar tokens per group, radius 2, and 12 data workers with prefetch 4.
These are starting values rather than a guarantee: the first full-resolution
optimizer step is the memory test. Override `PER_DEVICE_BATCH_SIZE=24` and then
`16` if needed. The V2 point records already contain fewer than 4096 points, so
raising `max_npoints` would not add information.

Before classifier training, the launcher creates a run-local data config with
identity action normalization. State normalization is retained. This is
required because the classification losses consume raw XYZ/Euler/gripper
targets, whereas the source V2 config points at regression action statistics.

This mode requires a real `polarapp_checkpoint` containing both
`refinement.*` and `output.*`; random TaskNet initialization is rejected even
when `polarapp_allow_random_init=True`. The dataset must provide aligned
`polar_images`, `polar_K`, `T_camera_from_model`, `view_valid`,
`observed_depth`, and `observed_depth_valid` (plus `pixel_valid` and
`polar_pixel_transform` when applicable).

An audit against the rendered V2 normal sidecars found that the released
TaskNet head is opposite to the SfP-Wild comparison frame
`(+left,+down,+forward)` on all three axes. TaskNet targets are therefore
converted once as `(-nx, -ny, -nz)`. Depth-derived OpenCV normals are converted
as `(-nx, ny, nz)`. Both then enter the cosine loss in the same frame.

The normal head is always frozen and its output is detached because it is the
pseudo-label teacher. With `polarapp_freeze=False`, the TaskNet feature body
can still receive gradients from action/depth feature fusion; the normal target
cannot move to reduce its own consistency loss.

## Suggested training sequence

1. Start with `polarapp_freeze=True` and train the FPN, PTv3 stage adapters,
   and joint-attention parameters.
2. Compare against the unchanged PTv3 and SfP-Wild baselines.
3. If routing and optimization are stable, set `polarapp_freeze=False` and use
   `polarapp_lr` (for example, `1e-5`) to apply a lower learning rate only to
   TaskNet. The new FPN and PTv3 continue to use the main learning rate.

Polar query outputs are still discarded by the existing joint-attention
implementation. This mode provides stage-wise TaskNet-to-PTv3 fusion, not a
2D feature-grid writeback path.
