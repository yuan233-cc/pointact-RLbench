# RLBench Benchmark

This directory contains the installation, data preparation, training, and evaluation instructions for the RLBench benchmark.


## Install RLBench

```bash
conda create -n rlbench python==3.10
```

Follow [instructions](https://github.com/vlc-robot/robot-3dlotus/blob/main/INSTALL.md) to install Pyrep and RLBench.

## RLBench Data Generation

The processed RLBench-10Task training data used in our [PointACT](https://arxiv.org/abs/2605.21414) paper can be downloaded directly:
```bash
hf download --repo-type dataset cshizhe/RLBench-10Task \
    --local-dir robot_data/rlbench/lerobot_point_lmdb
```

To generate your own RLBench data, follow the steps below.

1. Follow the [robot-3dlotus data generation pipeline](https://github.com/vlc-robot/robot-3dlotus/blob/main/DATAGEN.md) to generate microsteps and RGB-D keysteps.

2. Convert the keystep data to LeRobot v2.1 format:
```bash
conda activate pointact

export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

cd data_prep/rlbench_to_lerobot

keystep_dir=$SCRATCH/datasets/robot_data/rlbench/train_dataset/keysteps_bbox
microstep_dir=$SCRATCH/datasets/robot_data/rlbench/train_dataset/microsteps
task_instruction_file=$HOME/codes/robot-3dlotus/assets/taskvars_instructions_new.json
output_dir=$SCRATCH/datasets/robot_data/rlbench/lerbot_point_lmdb
repo_id=push_button+0

python convert_keystep_to_lerobot_v21.py \
    --keystep_dir ${keystep_dir} \
    --task_instruction_file ${task_instruction_file} \
    --taskvars push_button+0 \
    --keep_stop_action \
    --export_point_cloud --microstep_dir ${microstep_dir} \
    --output_dir ${output_dir} --repo_id ${repo_id}
```

3. Generate state/action statistics:

```bash
python data_prep/prepare_robot_state_action_stats.py \
    --dataset_dirs robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps \
    --output_file robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps/robot_state_action_stats/rot6d_points_frontview.json \
    --point_cloud_dir points_frontview \
    --state_xyz_slice 0 3 \
    --action_xyz_slice 0 3 \
    --state_rotation_slice 3 7 \
    --action_rotation_slice 3 7 \
    --rotation_type quat \
    --target_rotation_type rot6d

python data_prep/prepare_robot_state_action_stats.py \
    --dataset_dirs robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps \
    --output_file robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps/robot_state_action_stats/rot6d.json \
    --state_rotation_slice 3 7 \
    --action_rotation_slice 3 7 \
    --rotation_type quat \
    --target_rotation_type rot6d
```

## Training

For the aligned `phone_on_base` polar + incomplete-point-cloud example, run
`python experiments/10_rlbench/create_phone_polar_incomplete_episode.py` from
the repository root. The default output is
`robot_data/rlbench/lerobot_point_lmdb/phone_on_base_1episode_polar_incomplete9_seed24`.
It contains six keyframes from one successful episode. Its training LMDB stores
`[x, y, z, r, g, b, DoLP, cos(2 AoLP), sin(2 AoLP)]` under
`points_frontview_polar_incomplete9`; `points_frontview_polar_clean9` is a
reference copy. After geometry corruption, RGB and polar are sampled at each
point's current projected image pixel. Points projecting onto invalid polar
pixels are omitted. The color augmentation applies only to RGB, and the
example configuration disables geometry rotation because the polar angle is
defined in the original camera frame. The original six-channel `xyzrgb` mode
remains the default.

The matching example configuration is
`data_configs/data-phone-polar-incomplete9-one-episode.yaml`. On a machine with
a CUDA GPU and the pretrained Qwen model available locally, run
`bash experiments/10_rlbench/train_phone_polar_incomplete9_one_episode.sh` to
exercise one training epoch with `--ptv3_input_channels 9`. Set
`PTV3_INIT_CKPT_FILE` to a Concerto checkpoint to initialize the XYZRGB prefix
and zero-initialize the three polar weights, even when the Concerto stem itself
has nine inputs. One episode is a
training smoke test, not a useful final policy dataset.

To compare HouseCat-style depth filling on that episode, export a separate
dataset and select its filled-point configuration:

```bash
python experiments/10_rlbench/create_phone_polar_incomplete_episode.py \
  --fill-depth-holes \
  --output robot_data/rlbench/lerobot_point_lmdb/phone_on_base_1episode_polar_housecat_filled9_seed24_v3
DATA_PATH=experiments/10_rlbench/data_configs/data-phone-polar-filled9-one-episode.yaml \
  bash experiments/10_rlbench/train_phone_polar_incomplete9_one_episode.sh
```

`points_frontview_polar_filled9` retains the original incomplete points. It
projects them into a sparse depth image and estimates missing depths by
HouseCat-style multiscale morphology. Only pixels corresponding to voxel
samples actually lost to corruption are back-projected; ordinary empty pixels
from voxel downsampling are not filled. RGB and polar come from that pixel;
unavailable polar components are zero. The offline target-pixel mask is derived
from the pre-corruption source cloud, so this preprocessing is not a deployable
completion method without an inference-time way to locate missing pixels. The
renderer’s `depth_m` is not used as a completion target.
`point_depth_filled_mask` identifies generated rows for analysis but is not a
model input. The exporter computes separate state/action statistics for the
filled cloud. The filled XYZ are local estimates and can be wrong at object
boundaries.

To build a fresh, aligned ten-task dataset (100 successful demonstrations per
task), run `python experiments/10_rlbench/collect_10task_polar_episodes.py` on a
machine with the local custom RLBench renderer and CoppeliaSim installation.
The collector records each new successful trajectory once, saves only selected
keyframe scene snapshots, and renders polar maps at 512 samples per pixel.
It resumes episodes with completed `summary.json` and `render_summary.json`
files. Then run `python experiments/10_rlbench/export_10task_polar_incomplete9.py`.
The export creates a LeRobot dataset with separate LMDBs for incomplete nine
channel training points, intact reference points, full front-camera polar maps,
and original/current pixel indices. See the generated dataset README for the
exact channel and action semantics. The training configuration is
`data_configs/data-10task-polar-incomplete9.yaml`; run
`bash experiments/10_rlbench/train_10task_polar_incomplete9.sh` after export.
For the ten-task filled variant, append the corrected filled-point LMDB to the
existing dataset, leaving the clean and incomplete clouds untouched:

```bash
python experiments/10_rlbench/append_10task_polar_filled9.py
bash experiments/10_rlbench/train_10task_polar_filled9.sh
```

The filled-point configuration reads the same LeRobot episodes and actions,
selects `points_frontview_polar_filled9`, and uses its own normalization file.
It keeps the nine-channel XYZRGB+polar PointACT path; material conditioning is
disabled. The offline fill uses clean-source pixel provenance to locate holes,
which is not available during live inference.

The six-channel `xyz_polar` ablation uses the same rows and geometry but replaces
RGB with `[DoLP, cos(2 AoLP), sin(2 AoLP)]`. It does not send an RGB image to the
VLM and does not apply RGB augmentation to the polar tuple. Before the PTV3
stem, the tuple is represented in `[0, 1]` and passed through the same `2*x-1`
mapping as RGB, resulting in `[2*DoLP-1, cos(2 AoLP), sin(2 AoLP)]`. The three
polar channels inherit Concerto's pretrained RGB input weights:

```bash
bash experiments/10_rlbench/train_10task_xyzpolar_filled6.sh
# Optional training-only interaction reconstruction:
bash experiments/10_rlbench/train_10task_xyzpolar_filled6_target_reconstruction.sh
```

For a controlled three-way classification ablation, train the XYZRGB control
with the matched launcher below and compare it with
`train_10task_polar_rlbench9_v2.sh` and `train_10task_xyzpolar_filled6.sh`:

```bash
bash experiments/10_rlbench/train_10task_xyzrgb_filled6_matched.sh
```

All three configurations read identical filled9 rows and use the same workspace,
point budget, stochastic point retention, action statistics, no VLM image, no
rotation, and spatial-sampling RNG. The two modes containing RGB retain the
original PointACT RGB augmentation; the XYZ+polar mode has no RGB to augment.
RGB and polar model features use the same `[-1, 1]` range. The nine-channel
model copies Concerto's XYZRGB weights and zero-initializes its three added
polar columns; the two six-channel models copy all six Concerto XYZRGB input
columns. Existing checkpoints trained before these configuration fields were
added retain their original preprocessing and should not be mixed into this
controlled comparison.

For evaluation, use `run_filled9_rlbench.py` as the client and
`run_xyzpolar_filled6_server.py` as the policy server; the latter performs the
same `[XYZ, polar]` column selection as training.

SfP-Wild checkpoints use the live native-polarization renderer through a
separate adapter. It converts the rendered RGB, DoLP, AoLP and calibrated front
camera into the same seven-channel SfP tensor used by training, while the point
branch receives the same corrupted incomplete nine-channel cloud as training:

```bash
bash experiments/10_rlbench/eval_sfp_wild_rlbench_local.sh \
  /absolute/path/to/checkpoint-STEP STEP
```

The checkpoint must be an SfP-Wild action-regression run with polar depth
self-supervision enabled. Existing point-polar classification checkpoints keep
using `run_filled9_rlbench.py`; their inference path is unchanged.
The three controlled launchers share one training command. Its defaults are
1000 epochs, batch size 512 per GPU, learning rate `1e-4`, cosine scheduling,
seed/data seed 42, and a checkpoint every 500 steps. Override `OUTPUT_DIR`,
`EPOCHS`, `PER_DEVICE_BATCH_SIZE`, `TRAIN_SEED`, or `DATA_SEED` through
environment variables when needed. The one-episode launcher remains a
one-epoch smoke test.
These are newly generated trajectories, so episode numbers do not match the
original RGB-only dataset. Optical material parameters are assumptions stored
with the export, and the full polar maps cover valid camera-visible pixels.
`bash experiments/10_rlbench/finalize_10task_polar_dataset.sh` resumes the
collection, exports and verifies the 1000-episode dataset, writes a checked ZIP
archive, and uploads it to a private `yuan1119` Hugging Face dataset repository
when that account is logged in locally.

We support EO1, EO1-Point, QwenGR00T, QwenGR00T-Point, Pi0, and PointAct.
For PointAct, you can switch between classification and regression action heads. You can also remove images from the VLM by setting `video_key_ids_for_vlm: []` in the data configuration file. In RLBench, the 3D point cloud alone is often sufficient for most tasks, so removing images can substantially speed up training, roughly 13 hours on 1 H100 GPU, while keeping comparable performance.

Before launching a run, update the relevant configuration files in `experiments/10_rlbench/data_configs`.
SLURM launch examples are available in `job_scripts`.

We train all models with an effective batch size of 128 using 1 or 2 H100 GPUs.

```bash
# EO1
bash experiments/10_rlbench/train_eo1.sh
# EO1-Point
bash experiments/10_rlbench/train_eo1_point.sh
# QwenGR00T
bash experiments/10_rlbench/train_vla2.sh
# QwenGR00T-Point
bash experiments/10_rlbench/train_vla2_point.sh
# Pi0
bash experiments/10_rlbench/train_pi0.sh
# PointACT
bash experiments/10_rlbench/train_pointact_clf_concerto.sh
bash experiments/10_rlbench/train_pointact_clf_utonia.sh
```

## Inference

We use client-server evaluation so the policy can run in the `pointact` environment while the RLBench simulator runs in its own environment.

```bash
bash experiments/10_rlbench/eval_hybridvla_10tasks.sh <YOUR-EXPR-DIR> <CKPT-STEP> <SEED> <PRED-ROTATION-TYPE> "--args.num_episodes 100"
```

## Results

It is normal that success rates vary by about 1-5% across repeated evaluations.
After code optimization especially for EO1-Point which is unstable, we obtain better results than those reported in the paper :

| Model | Trainable Parameters | SR | 
| --- | --- | --- | 
| Pi0 (freeze vision encoder) | 2,720,744,480 | 68.1 |
| EO1 (freeze vision encoder) | 3,139,696,928 | 69.5 |
| EO1-Point (freeze vision encoder) | 3,348,939,040 | 73.5 |
| QwenGR00T (freeze VLM) | 1,068,806,144 | 52.4 |
| QwenGR00T-Point (freeze VLM) | 1,278,048,256 | 71.5 |
| PointACT (concerto, clf, point cloud only, freeze VLM) | 320,024,389 | 86.3 |
| PointACT (utonia, clf, point cloud only, freeze VLM) | 213,370,207 | 87.6 |
