# Observation-defined operation-space geometry training

TaskNet and CGA+DINOv3 geometry runs use the original PointACT
`get_rlbench_robot_workspace()` function, not the broader training crop.
Bounds are recorded in each new run config together with mask version.

The incomplete point sidecar and its original pixel IDs are rasterized BEFORE
workspace filtering or point subsampling. A Z buffer selects the nearest
available measurement; absent depth is NaN. Points outside the operation space
remain available as negative evidence but never enter PTv3 or depth loss.

The fixed image mask contains valid measured positions inside the operation
space and conservatively inferred missing pixels. Missing pixels require at
least four of the eight nearest inside observations within 12 pixels, support
above/below/left/right, and no outside observation in the surrounding 25x25
window. Unknown regions without sufficient support remain excluded. No clean
depth, GT normals, scene segmentation, predicted geometry or ray-AABB mask is
used to construct it. The existing sidecar was cropped to the broader training
space; absent pixels therefore mean missing observations, not certified sensor
failures. Conservative local support does not certify hidden surfaces.

Only inside observed points are candidates for the existing normal-guided PTv3
filter. Stage Polar attention uses the fixed mask, including inferred holes.
Decoder feature inputs/intermediate maps are masked at each scale. It still
uses ordinary dense convolution kernels (not sparse image computation); returned
depth outside the final mask is NaN and prediction_valid_mask is explicit.

There are exactly two losses: normal consistency in the fixed image mask
(including inferred holes; finite-difference neighbors must remain in mask),
and confidence-weighted point fitting only at valid inside measurements.
NaN holes never supply a metric depth target. Mask construction is detached;
the student cannot evade loss by moving a predicted point outside the box.
Frozen teachers retain their checkpoint preprocessing and may encode global
image context; only masked tokens enter trainable point fusion.
