# CGA + DINOv3 normal pretraining

The standalone path never loads the VLA. It requires independently supervised
surface normals; candidate normals in the physical-prior branch are inputs and
must not be reused as labels.

The required DINOv3 ConvNeXt-Base source is vendored under
`pointact/third_party/dinov3` at upstream commit `6876159`. Its weights are not
stored in this repository; set `model.dinov3_weights` to the separately obtained
official checkpoint. The vendored source is governed by the DINOv3 License
Agreement included in that directory.

Each manifest is JSON/JSONL/CSV/text. JSON entries use this form:

```json
{"samples": [{"path": "records/000001.npz", "group": "episode_000"}]}
```

`group` identifies an entire object, scene, or episode. Training exits if a
group occurs in both train and validation manifests.

`data.normal_gt_source` is mandatory in the training configuration. Values
identifying damaged/corrupted/observed depth or candidate normals are rejected.
For transparent simulation, inspect the rendered visible-surface object mask
before training so rear background normals are not marked valid.

Native CGA records may use the original keys `images`, `est`, `spec`, `Iun`,
`cos1`, `cos2`, `DoP`, `image_coordinate`, `label`, and `mask`. DINO training
also requires aligned `rgb`. Robot records use `polar_observation` (7 channels),
`physical_prior` (11 channels), `rgb`, `normal_gt`, `normal_valid_mask`, camera
intrinsics/coordinate metadata, and optionally `T_camera_from_world`.

`scripts/prepare_cga_priors.py` packages already computed SfP candidates and
specular confidence. It deliberately refuses to synthesize candidates from
depth or a learned observation adapter.

For HAMMER, SfPUEL, and Mitsuba-rendered RLBench data, use
`scripts/prepare_polar_normal_sources.py`. It computes the CGA inputs directly
from the four analyzer images, without using depth or normal GT as an input:

- `Iun = (I0 + I45 + I90 + I135) / 2`;
- `DoP`, `cos(2*AoLP)`, and `sin(2*AoLP)` from the linear Stokes terms;
- two specular and one diffuse ambiguous normal using the Fresnel model at
  refractive index 1.5;
- the CGA paper's specular confidence: the 3x3 local minimum of the per-pixel
  maximum-minus-minimum analyzer intensity.

All converted records use camera coordinates with `+x` right, `+y` down, and
`+z` forward. Supervision and physical candidates are face-forwarded against
the camera ray. SfPUEL groups are keyed by the first filename component so
material/view variants of one synthetic object cannot cross splits. HAMMER
groups are complete continuous trajectory directories. RLBench groups must be
complete `task + episode + seed` identities.

```bash
python scripts/prepare_polar_normal_sources.py sfpuel \
  --input-root /data/SfPUEL-training \
  --output /data/cga/sfpuel \
  --group-prefix-components 1

python scripts/prepare_polar_normal_sources.py rlbench \
  --input-root /data/reach_target_episode_polar_normal \
  --output /data/cga/rlbench_reach_target_ep0 \
  --group reach_target_ep0_seed7
```

HAMMER stores the four analyzer observations as quadrants. The public PPFT
preprocessing marks its angle-to-quadrant assignment as a guess, so the
converter has no default. First score all 24 layouts against independent normal
GT, inspect the score gap and visual QA, then pass the selected layout
explicitly:

```bash
python scripts/prepare_polar_normal_sources.py hammer-calibrate \
  --input-root /data/HAMMER --limit 8

python scripts/prepare_polar_normal_sources.py hammer \
  --input-root /data/HAMMER \
  --output /data/cga/hammer \
  --quadrant-layout 0,45,90,135
```

The last layout is only an argument example, not a validated HAMMER mapping.

```bash
python scripts/prepare_cga_priors.py \
  --manifest /data/raw_manifest.json \
  --output /data/cga_prior_records

python scripts/train_polar_normal.py \
  --config configs/polar_normal/cga_dinov3_convnext_base.yaml

python scripts/train_polar_normal.py \
  --config configs/polar_normal/cga_dinov3_convnext_base.yaml \
  --resume outputs/cga_dinov3_normal/last.pt

python scripts/eval_polar_normal.py \
  --checkpoint outputs/cga_dinov3_normal/best.pt
```

For the initial overfit check, set `data.overfit_samples` to a small value in
two manifests with disjoint group names. Set `model.use_dino: false` and use
`configs/polar_normal/cga_only.yaml` for the CGA-only ablation.

For PointACT, configure `polar_backbone: cga_dinov3_normal`, provide
`cga_dino_normal_checkpoint` and `dinov3_weights`, and store
`polar_rgb` plus `polar_physical_prior` in the aligned SfP sidecar. The policy
calls `forward_features()`, so DINO and F3/F4/F5 fusion run while the normal
decoder is skipped.

When `data.datasets` is a list, `train_polar_normal.py` concatenates its
datasets and uses inverse-dataset-size sampling, so every dataset has equal
expected sampling probability. Validation is reported per dataset and the best
checkpoint uses macro-average normal MAE. Every listed dataset must use the
same observation mode. A source may be training-only while too few independent
groups exist for a leakage-free validation split; at least one source must
still provide a validation manifest. All supplied train/validation manifests
must be group-disjoint.
