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
actual `K` using pixel centers in the canonical pretraining convention: +x is
image-right, +y image-bottom, and +z forward. Images must be undistorted before
this path. Immediately before a released SfP-Wild checkpoint, PointACT negates
only `ray_x` to match its legacy `vd_local.npy`; decoded SfP normals are mapped
back by the inverse x conversion.

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
--polar_token_mode local
--polar_max_views 8
--polar_writeback false
```

Select the point backbone with `--ptv3_backend concerto` or
`--ptv3_backend utonia`. Utonia retains its 3D rotary encoding for point
queries and keys; projected Polar tokens use their normalized image location,
camera-view embedding, and modality embedding.

`polar_token_mode=local` preserves calibrated projection routing and caps each
group at `polar_max_tokens_per_group`. `polar_token_mode=all` is the additional
global branch: every valid SfP grid token for that sample is copied into every
serialized point group, exactly as action tokens are copied. The all-token
branch intentionally ignores the local token cap and can be very expensive at
high-resolution feature levels.

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

The action-only Polar-token path continues to call `forward_features()`. With
polar/depth self-supervision enabled, `decode_normals(levels, normalize=True)`
provides detached unit-normal targets for the normals differentiated from
predicted depth. The integrated model keeps `up1`--`outc` frozen and in eval
mode; a full checkpoint is required for this objective.
