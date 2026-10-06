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

PTv3 uses the released large Concerto widths `64,128,256,512,768`, initialized
from its checkpoint with the first six stem channels copied and added polar
columns zeroed. Inputs are workspace-filtered incomplete XYZRGB+polar points,
up to 4096 per sample; saved current-observation pixel IDs remain aligned.

## Exactly three active objectives

`loss = normal_consistency + 0.2 * holdout_depth + 0.05 * visible_anchor_depth`.
Smoothness is disabled. Geometry operations and log residuals run in FP32.

* Normal consistency: depth-derived normals versus frozen teacher normals,
  masked to the workspace image envelope, ray/AABB intersection and optical
  validity. Require the normal stencil to remain inside that mask.
* Holdout depth: randomly select whole 8px blocks, remove their source points
  **before PTv3**, and retain exactly that mask for loss. Targets are only the
  incomplete sensor measurements before subsampling, never clean depth.
* Visible anchors: detached confidence from boundary-safe sensor normals and
  teacher agreement within 30 degrees. Find valid neighbors within six pixels
  and reject depth jumps; use Cauchy log-depth residuals. No forced minimum
  anchor count and no fallback that labels unverified points as trusted.

Both depth terms preserve the sensor's uncertainty: local normal agreement
cannot certify absolute depth or reject every coherent region-wide offset.
No complete-depth or GT-normal sidecar is opened by this training dataset.
Episode-disjoint validation holds out episode IDs ending in 9.

## Runtime

Use `--probe-seconds 600` for a bounded real training probe. It saves `last.pt`
and `timing_estimate.json`; the formal run can explicitly resume this checkpoint
into a new directory. Estimated wall time adds exactly 2400 seconds to the
measured remaining time including a 5% validation allowance. Only `last.pt` and
`best.pt` are retained. Cluster wrappers release final allocations when the
trainer exits, including failure, and keep W&B runtime/credentials in Job-local
`/tmp/yuan`.
