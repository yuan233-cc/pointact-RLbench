# TaskNet inherited-support bbox fusion

This is an opt-in candidate implementation. The formal training launchers
still default to `polar_fusion_mode=projection`. No cluster job, checkpoint,
or source V2 dataset was changed. Correctness tests, coverage measurements,
benchmarks and sanity training are deferred; no speedup or accuracy claim is
made yet.

## Data and geometry

Use `data-10task-polar-rlbench9-v2-tasknet-bbox.yaml`. Its
`use_point_image_support=True` preserves the CURRENT observation's
`point_pixel_indices`, not `point_source_pixel_indices`. The pixel ID is
`v * image_width + u`. The dataset attaches it as a row-aligned column before
workspace filtering, random sampling, rotation and centering, and removes
that column before feeding point features to the model. The collator validates
row counts and concatenates pixel IDs in exactly the same sample order as
points. `point_pixel_image_hw` retains the pixel-ID source dimensions.

The current V2 corruption is already baked into the LMDB: no new corruption
is introduced here. The chosen sidecar must describe those input points, not a
clean/source point cloud. Runtime shape/range assertions cannot prove that an
arbitrary replacement sidecar has correct semantic correspondence.

At input, each canonical point row gets `[u,v,u,v]`. PTv3 serialization only
stores order/inverse maps; it does not physically reorder `coord`/`feat`.
Attention gathers support with the same serialized/padded order. Concerto
and Utonia pooling both reuse their existing `cluster/indices/idx_ptr` to
reduce child boxes by min/min/max/max. `input_to_stage` composes that exact
cluster map across stages. No second voxelization is added.

The first version supports ONE frontview per sample. Boxes are inclusive
image pixel-center bounds. Expansion is centered, uses inclusive width/height,
and clips to image bounds. A singleton box samples its pixel repeatedly; no
unreported minimum-region heuristic is applied. A resize/crop requires an
explicit `polar_pixel_transform`; pixels outside the resulting image fail
validation rather than silently selecting unrelated features. Point-frame
rotation/centering does not alter the saved observation pixel correspondence.

## Dense bank and attention

TaskNet is evaluated once to obtain the full-image TaF maps. Their native
channels and convolution-grid strides, for the released default model, are:

| Level | Channels | Stride | Grid center offset |
| --- | --- | --- | --- |
| TaF1 / 0 | 96 | 1 | 0 |
| TaF2 / 1 | 96 | 2 | 0 |
| TaF3 / 2 | 192 | 4 | 0 |

The first real forward logs actual map shapes. For a 256x256 image the expected
resolutions are 256x256, 128x128, 64x64; these are source-code expectations, not
a newly executed shape test. `[0,0,1,2,2]` is a configurable candidate mapping
for the five PTv3 stages, not a measured optimum.

Every block follows its CURRENT serialization/padding membership:

```text
current point supports -> group union -> expanded bbox -> fixed regular grid
    -> batched bilinear grid_sample from full native TaskNet bank
    -> stage-shared adapter/norm on sampled tokens only
    -> 2D position + view-0 embedding + polarization modality embedding
    -> packed [Action | Point | Polar] joint self-attention
    -> canonical point restore + per-sample group-mean action output
```

The bank is not masked by depth/point/polar-valid masks. Missing-depth pixels
inside a sampled region can supply tokens. A finite regular grid does NOT
guarantee that every missing pixel is sampled. Each block resamples its own
groups; no old group-level polar tokens are carried across different orders.
Polar query output is exposed for inspection but not written back to the 2D
bank or reused in later blocks. The wrapper retains only a detached last-block
copy to avoid retaining an old autograd graph.

The native bank is converted once to float32 for bilinear interpolation and
coordinate precision during bf16 training. The token adapter and joint
attention follow the model's precision/autocast settings. Whole images are
not expanded to deep PTv3 channels. Sampling grids are packed by sample and
processed in one `grid_sample`; reduction and A/P/Z packing do not loop over
groups in Python. Original PTv3 padding still contains its existing per-sample
logic and some dynamic-size host synchronization remains.

The bbox branch does not construct/call the legacy `PolarTokenRouter` and
does not perform pooled-XYZ projection, per-point neighborhood expansion,
candidate unique or FPS. PTv3's own pooling `unique` is deliberately retained.
The old projection branch is retained for ablations.

## Depth decoder and freezing

The existing five-level trainable FPN is still used for the depth decoder,
not for selecting bbox polar tokens. Decoder point maps are built by following
each retained INPUT pixel's `input_to_stage` index to the corresponding deep
point feature. Collisions are averaged, and feature-grid borders are clamped
consistently with dense sampling. Pooled XYZ is not reprojected here either.

Depth holdout filters points AND their saved pixel IDs together before support
construction. If every point in a sample would be hidden, one point is kept
and its pixel is removed from the holdout target mask, avoiding direct target
leakage. Existing coarse decoder feature masking is retained. Sparse observed
depth remains a label, never a decoder input.

Action loss and the weighted normal-consistency, sparse-depth and smoothness
losses are unchanged. The TaskNet normal teacher remains detached in the
existing shared normal frame. `TASKNET_FREEZE=True` is retained by the new
classify launcher: released TaskNet parameters stay frozen; PTv3, fusion
adapters/attention, FPN and depth/action heads retain their existing training
policy. The new architecture changes stage-adapter input shapes, so an old
full projection checkpoint is not a drop-in strict-resume checkpoint. Use the
released TaskNet/PTv3 initializers for a separate new run.

## Configuration and deferred tools

New model/training configuration fields:

- `polar_fusion_mode`: `projection` (default) or `bbox`.
- `polar_bbox_grid_size`: 2, 4 or 6; K is respectively 4, 16 or 36.
- `polar_bbox_expansion`: five stage-specific finite values >=1.
- `polar_bbox_feature_levels`: five values chosen from 0,1,2.

The dedicated classify candidate entry is
`experiments/10_rlbench/train_10task_polar_rlbench9_v2_tasknet_bbox_classify_h200.sh`.
It selects bbox data/model settings and a separate output directory. It has
NOT been launched. The existing classification/regression scripts also accept
`POLAR_FUSION_MODE`, `POLAR_BBOX_GRID_SIZE`, `POLAR_BBOX_EXPANSION` and
`POLAR_BBOX_FEATURE_LEVELS`; lists use space-separated environment values.
Their default mode remains projection.

The geometry-only diagnostic `scripts/probe_tasknet_bbox_coverage.py` is ready
for later use with the production dataset and exact PTv3 pooling/padding.
It does not run TaskNet or attention. It compares each canonical stage point's
primary (unpadded) group across order families, with padding replicas still
contributing to group boxes. By default it reproduces the 0.7 depth-keep
policy; pass `--depth-keep-probability 1` for the unmasked-data ablation.

It reports IoU and bidirectional coverage at alpha=1,1.25,1.5,1.75,2, with
mean/median/P5/P10/P90/P95/min and fractions >=.90/.95/.99, grouped by
stage/order pair/sample/batch and aggregated. Region reports include image
and selected-feature-map area fraction and aspect ratio. Coverage is
canonical-stage-point weighted; area is group weighted. Feature-area bounds
are transformed by the native stride/offset and clipped to the feature grid.
Order shuffling is disabled ONLY in the diagnostic to preserve family labels;
training serialization behavior remains unchanged.

Future invocation (not executed in this change):

```bash
PYTHONPATH=. python scripts/probe_tasknet_bbox_coverage.py \
    --samples 16 --batch-size 4 --backend concerto \
    --output outputs/tasknet_bbox_coverage/report.json
```

The report refuses to overwrite an existing file. Review coverage against
region size before selecting alpha_s. All default alpha values are currently
1.0 (no expansion), deliberately NOT asserted as the final recommendation.
Profiler labels `polar_bbox/union`, `/grid`, `/grid_sample`, `/token_adapter`,
`/projection_pack` and `/joint_attention` support a later isolated benchmark.
Tests, real coverage, throughput/peak-memory comparison and sanity training
must still be completed before replacing formal training.
