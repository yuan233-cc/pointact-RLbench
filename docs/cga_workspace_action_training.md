# CGA+DINOv3 workspace geometry + action classification

This recipe adds learned action queries to the existing Concerto workspace
fusion path. It does not add physical reconstruction or residual-depth inputs.
No complete depth/normal GT is used for training.

## Entry point

Run from the PointACT conda environment:

```bash
bash experiments/10_rlbench/train_10task_cga_workspace_action_classify.sh
```

This is a training launcher, not a Slurm allocation request. It has not been
submitted to a cluster. Check dataset/model paths before launching.
The action recipe defaults to 30,000 optimizer steps (override `MAX_STEPS`).
The batch-size default is only a starting point; select the H200 batch and
allocation duration from a measured throughput probe, not local test timing.

The default teacher is
`pretrained/cga_dinov3_mixed_10tasks_v2_20261004/best.pt`.
Set `CGA_CHECKPOINT` and `DINOV3_WEIGHTS` to override teacher paths.
The recipe initializes PTv3 from Concerto; it does not automatically import the
recent standalone geometry checkpoint or resume its optimizer.

## Input and coordinates

- Frozen native-CGA teacher: 256x256, 11-channel observation and 11-channel prior,
  plus aligned original RGB for DINOv3.
- Preserve this checkpoint's I_un luminance-proxy convention; do not substitute
  TaskNet native S0. Analyzer channels are approximated from that intensity and
  corrected DoLP/AoLP, not claimed to be measured analyzer images.
- Camera axes: right/down/forward; integer-centered rays and depth geometry.
- Use original PointACT bounds from `get_rlbench_robot_workspace()`.
- Rasterize all incomplete observations before workspace crop/subsampling.
  Missing depth is NaN. Admit holes only through observed in-workspace
  neighborhood support and outside-observation exclusion.
- Large enclosed holes additionally use connected-component boundary support:
  bridge at most two pixels around observed in-workspace support, reject
  exterior/image-border components, require observations on both sides of both
  image axes, and veto a component with outside-workspace boundary observations.
  Regions larger than 35% of the image are conservatively rejected. No clean
  depth, simulated geometry, or predicted depth determines membership. This
  remains a neighborhood inference, not proof of an unknown point's 3D position.
  Previously generated workspace masks/teacher caches must be regenerated for
  this rule; the action recipe computes masks from observations online.
- The dataset centers points, action positions and camera transforms together.
  Normal filtering does not recenter the retained points a second time.

## Forward and losses

1. Run the frozen teacher once: retain dense normals and five feature levels.
2. Pool each native feature level by four; map stages 0/1/2/3/4 one-to-one.
3. Compute boundary-safe normal agreement from ORIGINAL observed depth.
   Filter PTv3 rows by their saved pixel identities, in training and inference.
   Never pad with rejected observations; an empty retained sample is an
   explicit error rather than silently misaligning its action label.
4. Concerto retains point/action self-attention. Point queries read workspace
   polar memory; action queries also read it by default. Polar memory does not
   write back. Trainable adapters still receive gradients through K/V.
5. Rasterize five fused point stages using inherited pixel support and pooling
   lineage, not fresh projection of pooled 3D centroids. The depth decoder
   consumes point feature maps and validity masks; no direct dense Polar/RGB or
   raw depth input is newly added.
6. Pool decoded features into a completion token; its learned projection is
   added to action embeddings before the classification head.

Default objective:

```text
L = L_position_CE + L_rotation_CE + L_gripper_BCE
    + 1.0 * (L_normal + L_weighted_point_fit)
```

Normal error is normalized by `1-cos(30 degrees)`. Within valid workspace
normal stencils, observed pixels have weight 1 and missing observed-depth
pixels have weight 3 (`POLAR_HOLE_NORMAL_WEIGHT`). Weighted averaging makes
this a relative pixel importance, not a global threefold loss multiplier.
Rejected-but-finite measurements are not reclassified as holes.

Point fit compares corresponding observed/predicted positions along the same
camera ray, normalized by 0.05 meters, with SmoothL1 and fixed detached weight
`0.1 + 0.9 * normal_confidence`. All in-workspace valid observations remain
targets, including observations removed from PTv3. Holes have no point-fit
target. No heldout-depth, anchor-hard-constraint or smoothness loss is used.
Predicted depth outside the fixed observation-derived workspace mask is NaN.

Trainable: Concerto, polar adapters/cross-attention, action embeddings/head,
depth decoder, completion-to-action projection. Freeze CGA+DINOv3/normal head.
VLM image input is disabled; the existing frozen Qwen text path is retained.

## Validation performed

- Unit checks of hole weights, masked losses, finite gradients and pixel-row
  filtering.
- Real small CUDA Concerto/action/decoder training backward and inference;
  only the image teacher is mocked for that integration test.
- Real V2 input comparison with standalone geometry preprocessing: observation
  max difference 1.2e-7, physical prior difference zero (frame 9-0).
- Full production-scale training convergence and H200 throughput have not
  been tested by these checks.
