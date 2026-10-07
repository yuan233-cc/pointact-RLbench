# Workspace geometry pretraining

`scripts/train_workspace_geometry.py` is a geometry-only entry point. It does
not construct Qwen or use action tokens. A frozen normal teacher supplies dense
polar features and detached normals; Concerto's five encoder stages, stage
polar adapters/cross-attention, and the metric depth decoder are trained.

## Teacher inputs

* TaskNet: `tasknet_frontview_native_stokes`, native S0/DoLP/AoLP, physically
  consistent Stokes resize to 64px. Native normals convert to canonical camera
  coordinates with `(-nx,+ny,-nz)`. Stage mapping is `[0,0,1,2,2]`.
* CGA+DINOv3: the selected mixed-10tasks checkpoint's **11-channel native_cga**
  recipe, original RGB-luminance proxy sidecar and same-frame RGB. Generate
  analyzers and the 11-channel physical prior using the existing pretraining
  converter. Preserve teacher resolution at 256px and its right/down/forward,
  camera-facing convention. Pool frozen encoder features by four for efficient
  attention, using `[0,1,2,3,4]`. Do not silently replace proxy intensity by
  native rendered S0: that would change this checkpoint's input distribution.
  A direct comparison to archived offline records verified integer-centered
  rays (`u-cx`, `v-cy`), not the converter's later +0.5 convention. The geometry
  trainer preserves these rays; depth backprojection also uses the archived
  Coppelia integer-center calibration. TaskNet optical input retains its own
  established preprocessing independently of sensor backprojection.
  Frozen CGA inference is chunked at 224 frames to avoid large convolution
  indexing tensors; pooled features are reassembled before the full PTv3 batch.

PTv3 uses the released large Concerto widths `64,128,256,512,768`, initialized
from its checkpoint with the first six stem channels copied and added polar
columns zeroed. Inputs are workspace-filtered incomplete XYZRGB+polar points,
up to 4096 per sample; saved current-observation pixel IDs remain aligned.

## Normal-filtered input and exactly two active objectives

The default `weighted_workspace` protocol has no held-out depth, no GT targets,
and no smoothness loss. A detached sensor/teacher normal confidence `q` is
computed once before PTv3. Boundary-safe neighbors within six pixels reject
depth jumps over `0.01 + 0.02 * depth` metres. Agreement within 30 degrees
produces positive confidence; only those point rows enter PTv3. Pixel IDs are
filtered with the same row mask. Empty trusted samples are excluded from the
batch and counted in `skipped_samples`; an entirely empty batch raises an error.
Rejected points are never substituted to manufacture a point count. Centering
uses retained points and composes camera transforms in FP32.

`loss = normal_consistency / (1-cos(30deg)) + weighted_point_fit`.
Both coefficients are one. The point term is Huber with beta=1 applied to 3D
distance divided by `--point-fit-scale-m` (default 0.05 m). Along each calibrated
pixel ray this distance is `abs(predicted_Z - observed_Z) * norm(ray)`. A 30-degree
normal error scores about 1; a 5 cm point error scores 0.5. These fixed reference
scales make terms comparable, not guaranteed equal on every batch. Logs retain
raw normal error and weighted metric point MAE. Huber retains a corrective
gradient for large offsets, unlike the old redescending Cauchy objective.

* Normal consistency: depth-derived normals versus frozen teacher normals,
  masked to the workspace image envelope, ray/AABB intersection and optical
  validity. Require the normal stencil to remain inside that mask.
* Weighted point fit: ALL valid observed workspace pixels before point
  subsampling, including observations rejected from PTv3. Weight is
  `0.1 + 0.9*q` by default (`--inconsistent-point-weight` controls the floor).
  Normalize by the sum of weights. Saved pixel correspondence replaces costly
  nearest-neighbor matching. Raster collisions use the nearest visible sensor
  depth, not occluded surfaces. No observed or clean depth is fed to the decoder.

The weights preserve the sensor's uncertainty: local normal agreement
cannot certify absolute depth or reject every coherent region-wide offset.
No complete-depth or GT-normal sidecar is opened by this training dataset.
Episode-disjoint validation holds out episode IDs ending in 9.
Validation uses the same normal-filtering protocol, with no hidden-depth mask.
Observed-point fitting measures reconstruction, not unseen-hole accuracy.
Legacy checkpoints without `supervision_mode` still evaluate with their original
three-term structured-holdout protocol; their training semantics are not changed.

## Runtime

Use `--probe-seconds 600` for a bounded real training probe. It saves `last.pt`
and `timing_estimate.json`; the formal run can explicitly resume this checkpoint
into a new directory. Estimated wall time adds exactly 2400 seconds to the
measured remaining time including a 5% validation allowance. Only `last.pt` and
`best.pt` are retained. Cluster wrappers release final allocations when the
trainer exits, including failure, and keep W&B runtime/credentials in Job-local
`/tmp/yuan`.
