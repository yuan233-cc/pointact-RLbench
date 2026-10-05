# RLBench v2 scene-level CGA normal pretraining input

`RLBenchPolarNormalLmdbDataset` joins `polar_frontview_dense`,
`sfp_frontview_rgb_luminance_proxy`, and `normal_frontview_dense` by the
same `episode-frame` key, and decodes front RGB from the matching AV1 video.
The split is by complete episode, across all ten tasks and full scenes.

Generate local train/validation manifests after staging the dataset:

```bash
python scripts/make_rlbench_polar_normal_manifests.py \
  robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps_polar_rlbench9_v2 \
  robot_data/polar_normal_rlbench_v2_manifests
```

The loader generates the 7-channel robot observation and 11-channel CGA
physical prior **on demand**; it does not duplicate 5051 dense frames on
disk. The prior uses only DoLP, AoLP, proxy intensity, and camera calibration;
it does not use the normal label. Candidate normals use refractive index 1.5
by default. The coordinate frame is PointACT SfP: +x left, +y down, and
camera-facing normals have negative z.

For mixed pretraining with native-CGA HAMMER and SfPUEL, set
`input_mode: native_cga` on the RLBench dataset entry. This produces an
11-channel observation by reconstructing four **proxy** analyzer intensities
from DoLP/AoLP and RGB-luminance `I_un`. It converts rays, ambiguous-normal
candidates, and normal labels together to +x right, +y down, +z forward.
The original 7-channel `robot` mode remains the default for RLBench-only
experiments. The mixed mode does not turn the proxy intensities into measured
polarization; keep per-dataset validation separate.

Important limitation: the corrected v2 renderer did not retain its four
analyzer intensity images. `I_un` is an RGB-luminance proxy. Consequently,
the reconstructed Stokes contrast and 3×3 specular-confidence channel are
**approximations**, not measured physical intensity. DoLP/AoLP are from the
corrected polarization rerender. The target normals are geometric normals
from archived Coppelia depth and same-object finite differences, not Mitsuba
shading-normal AOVs. Masked normal loss should be used; these labels can be
noisy at edges and are not a replacement for a held-out normal benchmark.

To check the pipeline locally:

```bash
python scripts/train_polar_normal.py --config configs/polar_normal/rlbench_v2_cga_smoke.yaml
```

For the full CGA+DINOV3 run, set `model.dinov3_weights` to a real local
ConvNeXt-Base DINOv3 checkpoint and use
`configs/polar_normal/rlbench_v2_cga_dinov3.yaml`. The provided path is a
placeholder; the full run cannot start until the weights and a GPU runtime
are available.
