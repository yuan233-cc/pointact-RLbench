# Polar + PointACT fusion self-supervision

This branch predicts a dense, camera-visible metric-depth image from one
co-located polar/depth view. It fills holes in the visible depth map; a single
view cannot determine the hidden back side of an object without a learned
shape prior or additional views.

## Data flow

For every sample, the dataset now loads:

- SfP-Wild input: `I_un`, DoLP, `cos(2AoLP)`, `sin(2AoLP)`, and three viewing-ray channels.
- The incomplete input point cloud and its aligned flattened pixel indices.
- Camera intrinsics and `T_camera_from_world`.

Before point sampling, augmentation, and centering, the loader transforms the
observed points into the camera frame and z-buffers them into
`observed_depth`. This is a sparse measurement anchor, not a complete-depth
label. No new trajectory collection or rendering is needed for this part.

PointACT records the fused point hidden states after each of its five encoder
stages. The calibrated camera projects those hidden states onto the matching
SfP-Wild feature grids. Features that land in the same cell are averaged and a
validity mask distinguishes projected cells from holes. A U-Net-style decoder
then combines the five projected PointACT maps with the five SfP-Wild image
feature maps. Its sigmoid output is mapped to the configured metric range. The
default range is 0.05–4.5 m.

The decoder never receives `observed_depth`. That tensor is generated from the
incomplete input point cloud and is used only as a held-out metric-depth target.
During training, 30% of the observed target pixels are held out. Input points
that project to those pixels are removed before the PointACT forward pass, and
the corresponding multiscale projected cells are also zeroed before decoding.
This prevents PointACT attention from seeing the target depth while preserving
gradients through the remaining polar/point fusion context.

## Objective

Predicted depth is back-projected with the calibrated intrinsics. Central
finite differences give a camera-facing normal at each interior pixel. With
incidence angle `theta` and refractive index `eta`, the code evaluates both
the dielectric diffuse and specular Fresnel DoLP candidates.

The observed doubled-angle phase vector is

```text
(cos(2AoLP), sin(2AoLP))
```

For the predicted normal azimuth, diffuse and specular AoLP differ by `pi/2`.
Their doubled-angle vectors therefore have opposite signs. The phase loss uses
the absolute dot product to handle this ambiguity. DoLP is compared separately
against the lower-error diffuse/specular Fresnel candidate, with a default
weight of 0.25 because fixed-eta DoLP is sensitive to material. Separating
phase and DoLP also prevents a fronto-parallel zero-DoLP prediction from
trivially minimizing a raw normalized-Stokes loss. It also uses:

- masked point modelling: points at 30% of observed depth pixels are removed
  before PointACT and supervise a robust log-depth error, which fixes metric
  scale without feeding a depth image to the decoder;
- edge-aware inverse-depth smoothness, weighted by `I_un` image gradients.

The implemented objective is inspired by CroMo's differentiable
geometry-to-polarization constraint. It is adapted to this dataset because
there is no iToF observation, full point cloud, or GT normal.

References: [CroMo](https://arxiv.org/abs/2203.12485) and
[S²P³](https://link.springer.com/article/10.1007/s11263-023-01965-w).

`I_un` in the current sidecar is RGB luminance, not physical unpolarized
intensity/S0. Therefore the loss compares DoLP and doubled-angle AoLP phase,
and does not reconstruct the four analyzer intensities. `I_un` is still used by SfP-Wild
and as the edge image for smoothness. If physical S0 is rendered later, an
analyzer-intensity loss can be added safely.

## Training

The existing launcher enables the objective:

```bash
SFP_CHECKPOINT=/path/to/onlyiun_pol_vd_checkpoint \
bash experiments/10_rlbench/train_10task_polar_rlbench9_v2_sfp_wild_proxy.sh
```

Set `PTV3_BACKEND=utonia` and `PTV3_INIT_CKPT_FILE` to the Utonia checkpoint
to use Utonia's native 54/108/216/432/576 channel widths and
3/6/12/24/32 attention heads. The same fused stage maps feed the dense depth
decoder, so the reconstruction loss still updates Polar-to-point attention
and the completion feature still conditions the action tokens.

Useful environment overrides are:

```text
POLAR_DEPTH_LOSS_WEIGHT=0.1
POLAR_CONSISTENCY_WEIGHT=1.0
SPARSE_DEPTH_WEIGHT=1.0
DEPTH_SMOOTHNESS_WEIGHT=0.01
POLAR_REFRACTIVE_INDEX=1.5
POLAR_MIN_DOLP=0.02
POLAR_DOLP_WEIGHT=0.25
POLAR_DEPTH_KEEP_PROBABILITY=0.7
POLAR_DEPTH_MIN=0.05
POLAR_DEPTH_MAX=4.5
MAX_STEPS=40000
```

With `SFP_FREEZE=True`, the pretrained SfP encoder remains frozen. The
self-supervised loss updates the PointACT point/polar fusion blocks directly,
because their five output feature maps are decoder inputs. It also updates the
depth decoder. PointACT's original per-stage linear adapters remain necessary
to match SfP channel dimensions to each attention stage; the extra shared
identity adapters from the earlier implementation have been removed.

The final 32-channel decoder feature is spatially pooled, projected to the
PointACT output width, and added to every action token before the action head.
The projection is zero-initialized so loading/training starts with baseline
action behavior. It then learns from action loss, allowing completed geometry
to affect action prediction. With `SFP_FREEZE=False`, both losses additionally
update the SfP encoder.

The fixed dielectric Fresnel model is only an approximation for conductors,
mixed pixels, and unknown refractive indices. The per-pixel diffuse/specular
minimum and robust losses reduce this sensitivity, but material-dependent eta
or a learned confidence head is the next extension if these pixels dominate.

## Dataset and pipeline audit

The configured v2 dataset was scanned end to end:

- 5,051 frames and 13,391,770 incomplete points;
- every point-cloud row has a matching current-pixel and source-pixel entry;
- every frame has a 256×256 dense-polar record and an SfP input record with
  valid `I_un`, `K`, and `T_camera_from_world` shapes;
- no missing keys, non-finite point values, or out-of-range pixels were found;
- 97,933 same-pixel collisions are resolved by nearest-depth z-buffering.

About 27.72% of incomplete points project to a different pixel from their
clean source because this dataset intentionally simulates holes, wrong depth,
shape distortion, and floating points. These points are valid model inputs but
not trustworthy depth labels. For that reason, the sparse loss uses masked
holdout pixels and a Cauchy penalty with bounded outlier influence. This makes
the dataset trainable for robust fusion, but it is not a clean depth-accuracy
benchmark.

The SfP `+left,+down,+forward` x-axis convention was checked against 95,084
valid auxiliary renderer normals. Negating the OpenCV camera x component gave
a median absolute doubled-angle phase agreement of 0.955. The auxiliary GT
normals are used only for this audit and are not loaded by training.

The self-supervised loss now reaches PointACT attention through the rasterized
fused point hidden states. The decoder's completion feature also reaches the
action head in the same forward pass. Predicted dense points are still not
appended to the serialized point set; doing that would require a second
PointACT pass and substantially more memory.

No local official SfP-Wild checkpoint was found during the audit. The launcher
therefore intentionally refuses to start until `SFP_CHECKPOINT` names a real
checkpoint. Random SfP initialization remains suitable only for smoke tests.
