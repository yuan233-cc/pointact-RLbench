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
`+z` forward. Supervision is face-forwarded against the camera ray when
intrinsics are known; physical candidates use the camera-facing `-z`
hemisphere. Pixel centers are `(x+0.5,y+0.5)`.
SfPUEL groups are keyed by the first filename component so
material/view variants of one synthetic object cannot cross splits. HAMMER
groups are complete scenes, keeping all their trajectories on one side of the
split. The empty-object `_naked` trajectories are excluded. RLBench groups must be
complete `task + episode + seed` identities.

HAMMER does not ship a `norm/` modality in the public archive. Its adapter
derives supervision only from the clean polarization-camera `_gt` z-depth and
the sequence's own `intrinsics.txt`, using centered 3D finite differences and
rejecting invalid pixels and depth discontinuities. No noisy D435/L515/ToF
sensor depth is used as normal supervision.

SfPUEL's official loader decodes 16-bit normal PNGs as RGB divided by 65535,
then mapped from `[0,1]` to `[-1,1]`. The raw map has image-right `x`,
image-up `y`, and camera-facing `+z`; the adapter applies `(x,y,z) ->
(x,-y,-z)` before unit normalization. Because SfPUEL does not provide camera
intrinsics, its `image_coordinate` is the constant optical-axis vector
`(0,0,1)` rather than an invented per-pixel perspective ray. The axis choice
was checked against the raw normal-map colors and the polarization prior on
the local subset; it should be revisited if another SfPUEL release changes
the normal-map convention.

The RLBench Mitsuba source stores negative `fx` and `fy` but renders rays
using their absolute values. Its raw AOV camera `x/y` axes must both be
negated to match image-right/image-down coordinates. The adapter converts
the focal signs and AOV axes, then keeps only valid pixels where the AOV
normal agrees within 15 degrees with a normal independently reconstructed
from rendered z-depth. This geometry check removes discontinuities and
shading normals that differ strongly from the visible surface geometry.
The 44-frame local subset retains about 80% of pixels after this check.

```bash
python scripts/prepare_polar_normal_sources.py sfpuel \
  --input-root /data/SfPUEL-training \
  --output /data/cga/sfpuel \
  --group-prefix-components 1 \
  --spatial-stride 4

python scripts/prepare_polar_normal_sources.py rlbench \
  --input-root /data/reach_target_episode_polar_normal \
  --output /data/cga/rlbench_reach_target_ep0 \
  --group reach_target_ep0_seed7
```

HAMMER stores the four analyzer observations as quadrants. [LUCID's Phoenix
camera example](https://thinklucid.com/polarized-camera-resource-center/3d-depth-from-polarization-sfp/)
reads the four quadrants in top-left/top-right/bottom-left/bottom-right order
as 0/45/90/135 degrees; its [Bayer polarization pixel format](https://support.thinklucid.com/knowledgebase/pixel-formats-area-scan/)
also names the channels in that order. The [PPFT HAMMER preprocessing](https://github.com/lastbasket/Polarization-Prompt-Fusion-Tuning/blob/master/scripts/data_processing/process_hammer.py)
uses the same ordering but explicitly calls it a guess. Because HAMMER itself
does not publish capture-format metadata, the converter still requires an
explicit layout:

```bash
python scripts/prepare_polar_normal_sources.py hammer-calibrate \
  --input-root /data/HAMMER --limit 8

python scripts/prepare_polar_normal_sources.py hammer \
  --input-root /data/HAMMER \
  --output /data/cga/hammer \
  --quadrant-layout 0,45,90,135 \
  --spatial-stride 4
```

On the local 16-frame, eight-sequence subset, a physics-candidate score ranked
`45,90,0,135` first, only 0.08 degrees ahead of the next layout. This is too
small a margin to overrule LUCID's documented camera output ordering, and the
score depends on material/reflection assumptions. The Stokes orthogonal-pair
sum differences are also small across candidate pairings. The calibration
JSONs remain as audit artifacts. Use `0,45,90,135` for the current HAMMER
records, while retaining the qualification that HAMMER authors have not
confirmed whether their saved PNGs were reordered after capture. Geometry
supervision from clean depth is independent of this quadrant choice.
`--spatial-stride 4` matches the public PPFT HAMMER loader's spatial reduction
and prevents packed float records from becoming unnecessarily large.

Create leakage-safe splits only at whole-group granularity:

```bash
python scripts/split_polar_normal_manifest.py \
  --manifest /data/cga/hammer/manifest.json \
  --train-output /data/cga/hammer/train_manifest.json \
  --val-output /data/cga/hammer/val_manifest.json \
  --val-fraction 0.2 --seed 7
```

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
