# SfP-Wild Polar tokens

This is an experimental extension, not a structure claimed by the PointACT or
SfP-Wild papers. The original PointACT path remains the default
(`polar_enabled=false`).

The implemented path is limited to
`VLAEncDec3DWithActionRegressionModel`, the Concerto or Utonia backend, and
encoder-only PTv3. Unsupported combinations fail during model construction.

## Input contract

Each batch must contain:

- `polar_images`: `[B,V,7,H,W]`, ordered as
  `[I_un, DoLP, cos(2 AoLP), sin(2 AoLP), vx, vy, vz]`;
- `polar_K`: `[B,V,3,3]` intrinsics for the undistorted image geometry;
- `T_camera_from_model`: `[B,V,4,4]`;
- `view_valid`: `[B,V]`;
- optional `pixel_valid`: `[B,V,H,W]`;
- optional `polar_pixel_transform`: `[B,V,3,3]`, mapping pixels produced by
  `polar_K` into the resized/cropped/padded SfP input.

SfP-Wild loads `I_un` and `DoLP` from its saved polarization array without an
extra normalization step; AoLP is in radians. The helper
`assemble_onlyiun_pol_vd` builds the seven channels and generates rays from the
actual `K`. It follows the released `vd_local.npy` convention: +x is image-left,
+y image-bottom, and +z forward. Images must be undistorted before this path.

Point augmentation must retain
`T_model_from_world=[R_aug,-center;0,1]`, then compute:

```text
T_camera_from_model = T_camera_from_world @ inverse(T_model_from_world)
```

No identity calibration is synthesized. RGB augmentation is not applied to
the seven-channel Polar tensor.

## Configuration

```text
--polar_enabled true
--sfp_checkpoint /absolute/path/to/sfp_wild_checkpoint.pth
--sfp_freeze true
--sfp_feature_levels x1 x2 x3 x4 x5
--polar_neighbor_radius 1
--polar_max_tokens_per_group 32
--polar_max_views 8
--polar_writeback false
```

Select the point backbone with `--ptv3_backend concerto` or
`--ptv3_backend utonia`. Utonia retains its 3D rotary encoding for point
queries and keys; projected Polar tokens use their normalized image location,
camera-view embedding, and modality embedding.

The official checkpoint link is in the SfP-Wild README. A missing checkpoint
is an error. `sfp_allow_random_init=true` exists only for explicit from-scratch
tests and must not be described as pretrained initialization.

After a forward pass, per-block routing counters are available at
`model.ptv3_model.last_polar_route_stats`. `save_polar_route_overlay` can draw
projected points and selected feature-grid centers for calibration checks.

## Normal decoder

`SfpWildFeatureEncoder` also restores the released `up1`--`up4` U-Net decoder
and three-channel `outc` normal head. They use the original parameter names, so
a full SfP-Wild checkpoint loads both the feature encoder and normal decoder.
An older encoder-only checkpoint remains valid and reports the decoder keys as
missing instead of silently treating them as pretrained.

The existing Polar-token path continues to call `forward_features()` and does
not execute the decoder. Call `forward_with_normals()` to obtain `(x1..x5,
raw_normals)`, or `decode_normals(levels, normalize=True)` for unit camera-frame
normals. The integrated PointACT model currently freezes these decoder
parameters because normal ground truth and its auxiliary loss are not yet part
of the batch contract; this avoids unused trainable parameters in distributed
action-only training.
