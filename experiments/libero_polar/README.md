# LIBERO Spatial polar / incomplete-point replay

This directory adds data-generation code **without modifying the official
LIBERO checkout**. It restores each HDF5 demonstration's initial MuJoCo state,
replays its recorded actions, and captures aligned RGB, metric depth, geom-ID
segmentation, world XYZ, polarization, clean 9-D points, and corrupted 9-D
points at the current state before each action. It also produces morphology-
completed 9-D points and training-only visible interaction reconstruction GT.

## Why replay is required

The official LIBERO HDF5 files contain RGB and low-dimensional simulator state,
but not metric depth, camera calibration, segmentation, polarization, or a
complete independently renderable scene snapshot for every observation. The
initial simulator state plus the action sequence is therefore used to recreate
the dynamic scene. Replay is not needed for an RGB-only baseline.

Replay is also not mathematically mandatory here: the official HDF5 contains a
flattened MuJoCo state for every step. `--state-source recorded` restores those
states independently and is useful for render-only generation without
cumulative controller drift. The default `--state-source replay` follows the
OpenVLA regeneration procedure and additionally checks whether the recorded
actions still reach success in the installed environment. Compare a few 128px
frames against the archived RGB before choosing `recorded` for a full run;
renderer/version differences mean restoration need not be pixel-identical.

## Polarization backends

- `analytic` estimates normals from MuJoCo depth and applies a deterministic
  Fresnel screen-space model. It is fast and useful for alignment tests, but is
  **not** path-traced polarization.
- `native` adapts MuJoCo visual geoms to the independent CUDA spectral Mueller
  renderer already used by the RLBench polar pipeline. It does not patch
  LIBERO. It renders only Stokes/polarization; MuJoCo depth remains authoritative
  for every point and the native depth/mask/point buffers are disabled. Textures
  are currently represented by material base color. Use this backend for the
  intended training dataset.

The saved metadata always identifies the backend, so the two cannot be silently
mixed.

## Output

Every compact `frames/NNNNNN.npz` contains:

- dense `rgb`, `depth_m`, `geom_id` and camera matrices;
- `DoLP`, `AoLP`, doubled-angle channels and validity mask;
- `clean9` and `incomplete9`, whose rows are
  `[x,y,z,r,g,b,DoLP,cos(2AoLP),sin(2AoLP)]`;
- `filled9`, which appends morphology-estimated points only at known simulated
  corruption holes;
- clean/incomplete source-pixel provenance, geom IDs, corruption codes, robot
  state, and the original 7-D LIBERO action;
- `filled_source_pixel_index`, `filled_current_pixel_index`,
  `filled_synthetic_mask`, and `filled_corruption_code`;
- `interaction_target_points` (visible clean bowl + plate XYZ, 5 mm voxelized,
  at most 512) and `interaction_input_mask` (one binary label per `filled9`
  row).

For the full 500-episode dataset, `--omit-redundant-aolp` omits only the dense
`AoLP` array while retaining `cos2AoLP` and `sin2AoLP`. Recover radians in
`[0, pi)` with `mod(0.5 * atan2(sin2AoLP, cos2AoLP), pi)`. This preserves the
physical polarization representation and saves about 10 GiB. `--resume` skips
completed successful episodes, regenerates a partial episode, and
`--min-free-gib` stops before starting another episode instead of filling the
filesystem.

Pass `--save-dense-debug` for a small QA run to additionally retain dense world
XYZ and Stokes `S0..S3`. These arrays are omitted by default because they can be
reconstructed from depth/calibration or are not model inputs, and would add
hundreds of gigabytes to a 500-episode 256px dataset.

When corruption moves a point, its RGB and polarization channels are sampled
again at its new projected pixel. `incomplete_source_pixel_index` records the
original pixel and `incomplete_current_pixel_index` records the new one. Points
that leave the image are removed; points on an invalid polar pixel remain, with
their three polar feature channels set to zero.

The incomplete cloud uses exact MuJoCo geom names. For Spatial, the manipulated
object is `akita_black_bowl_1` and the related support is `plate_1`. Corruption
parameters are deterministic per episode and cover robot holes, target
dropout/wrong depth, support distortion, and sparse floating points.

Deletion strength is explicit and archived in both the episode summary and
dataset manifest. The generator accepts `--robot-drop-fraction`,
`--target-affected-fraction`, and `--target-drop-fraction`; their defaults are
`0.13`, `0.55`, and `0.65`. Existing clean/polar replay data can be used to
build a directly comparable corruption variant with
`rebuild_corruption_variant.py`, without rerunning the native renderer.

## Geometry, completion, and reconstruction semantics

`clean9` geometry comes only from LIBERO's MuJoCo metric depth and camera
calibration. Polar validity never removes a valid MuJoCo point; unavailable
polar channels are set to zero. `incomplete9` is created by changing/removing
rows of that clean cloud, never from native-renderer geometry.

The depth completer matches the RLBench filled9 preprocessing: it z-buffers the
corrupted cloud, applies depth-dependent cross dilation, closing, median/local
propagation, and bilateral smoothing, then unprojects estimates. Candidates are
restricted to clean voxel-source pixels removed by the simulated corruption.
This makes `filled9` an offline controlled-dataset input. The clean hole list is
not available at deployment; a live experiment needs a separate hole detector
or learned completion model.

Reconstruction GT is also training-only. Its geometry and labels use untouched
MuJoCo depth plus exact geom segmentation, never native polar depth. The visible
target set combines the manipulated `akita_black_bowl_1` with the related
`plate_1`; half the 512-point budget is reserved for the bowl when possible.
`interaction_input_mask` labels each filled input point positive only when its
current projection belongs to either interaction object and agrees with the
original MuJoCo depth within 5 cm. Neither target points nor labels are policy
inputs at inference.

## Local smoke test

```bash
export PYTHONNOUSERSITE=1
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=egl
export PYTHONPATH=/media/hyunjun/NewDisk1/Yuan_Feng/6dor/LIBERO_official
export LIBERO_CONFIG_PATH=/tmp/libero_geovla_check

/home/hyunjun/anaconda3/envs/cavla3d/bin/python \
  experiments/libero_polar/replay_libero_spatial.py \
  --raw-dir /media/hyunjun/NewDisk1/Yuan_Feng/6dor/Open6DOR_V2_Execution/libero_datasets/datasets/libero_spatial \
  --output /tmp/libero_spatial_polar_smoke \
  --task-ids 0 --episodes-per-task 1 --max-steps 2 \
  --resolution 64 --polar-backend analytic
```

For physical rendering, remove `--max-steps`, use resolution 256, and select:

```bash
--polar-backend native \
--native-renderer-repo /media/hyunjun/NewDisk1/Yuan_Feng/rlbench_custom_render/RLBench \
--spp 512 --max-depth 8
```

The native renderer defaults to `libero_spatial_materials.json` and accepts an
alternative through `--materials`. Keys may be exact MuJoCo geom names or glob
patterns. The supplied profile assigns glazed ceramic to both black bowls, the
plate and ramekin; coated paper/plastic to the cookie box; coated wood to the
table and cabinet; painted plastic to the robot/mount and stove base; and rough
aluminum to gripper parts and the stove burner. These are engineering priors,
not measured properties. Calibrate and archive the file before reporting a
physical-polar experiment.

The bundled profile covers all visual geoms in the current LIBERO-Spatial
scene; collision geoms are neither rendered nor counted as material overrides.

Every frame runs physical consistency checks: finite Stokes values, DoLP in
`[0,1]`, recomputed DoLP agreement, the Stokes realizability cone
`S0² >= S1²+S2²+S3²`, nonnegative ideal-analyzer intensities, and unit-length
`[cos(2AoLP), sin(2AoLP)]` on pixels where AoLP is defined. The report is stored
under `frames[*].polar.physics_qa` in the episode summary.
Per-object pixel counts and DoLP mean/median/P95/max are stored alongside it in
`frames[*].polar.region_qa`; the MuJoCo geom-ID/name table is archived in each
episode summary.

Set `NATIVE_POLAR_NVCC` when `nvcc` is not on `PATH`. Full 500-episode capture
should run on the cluster. This NPZ representation is an auditable intermediate
format; stream it into GeoVLA RLDS shards after one-episode alignment QA, rather
than creating a second dense intermediate copy.

The adapter intentionally imports only the renderer submodules and bypasses the
RLBench package initializer, so PyRep/CoppeliaSim are not dependencies of the
LIBERO generation process.

## Full Spatial generation

The resumable 10-task, 500-episode native-polar run is encapsulated in:

```bash
experiments/libero_polar/generate_full_native.sh
```

It uses 256 px, 512 spp, path depth 8, replay state, no-op filtering, the
archived material overrides, compact double-angle polarization storage, and a
5 GiB free-space guard. Check progress with:

```bash
python experiments/libero_polar/full_dataset_status.py \
  robot_data/libero/libero_spatial_polar_native_256_full_500ep_v1
```
