#!/usr/bin/env bash
# Launch TaskNet/PointACT classification in the existing Aachen SSH allocation.
set -euo pipefail
umask 077

[[ "${SLURM_JOB_ID:-}" == 26196 ]] || { echo 'Expected allocation 26196' >&2; exit 2; }
: "${RUN_NAME:?Set a unique RUN_NAME}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2

repo=/mnt/home/weihangli/pointact_project/code/pointact_tasknet_v2_0f65a30
image=/mnt/home/weihangli/pointact_project/containers/pointact.sqsh
dataset=/tmp/yuan/datasets/job-26196/rlbench-v2/hybridvla_10tasks_train_keysteps_polar_rlbench9_v2
output=/local/weihangli/checkpoints/$RUN_NAME
netrc=/tmp/yuan/credentials/job-26196/netrc
wandb_root=/tmp/yuan/wandb/job-26196/$RUN_NAME

test -f "$dataset/meta/info.json"
test -f "$repo/pretrained/PolarAPP/TaskNet.pth"
test -f "$repo/pretrained/Pointcept-Concerto/concerto_large.pth"
test -f "$netrc"
test "$(stat -c %a "$netrc")" = 600
test -d "$output" && test -w "$output"
test -d /local/weihangli/checkpoints
test -w /local/weihangli/checkpoints
test "$(sha256sum "$repo/pointact/model/vla_pointact/action_head_3d/polar_router.py" | cut -d ' ' -f 1)" = db7fa1f2b93de07cc8c2926f0ba93cb86e9c1684b7ac46ccc3fd42ec2ff35f83

mkdir -p "$wandb_root" "$wandb_root/data" "$wandb_root/cache" "$wandb_root/config" /tmp/yuan/hf/job-26196
export ENROOT_DATA_PATH=/tmp/yuan/pointact_enroot/job-26196/data
export ENROOT_CACHE_PATH=/tmp/yuan/pointact_enroot/job-26196/cache
export ENROOT_RUNTIME_PATH=/tmp/yuan/pointact_enroot/job-26196/runtime
export ENROOT_TEMP_PATH=/tmp/yuan/pointact_enroot/job-26196/tmp

exec enroot start --root --rw \
    --env NETRC="$netrc" \
    --env TASK_RUN_NAME="$RUN_NAME" \
    --env TASK_MAX_STEPS="${MAX_STEPS:-20000}" \
    --env TASK_SAVE_STEPS="${SAVE_STEPS:-250}" \
    --env TASK_LOGGING_STEPS="${LOGGING_STEPS:-2}" \
    --mount /mnt:/mnt --mount /tmp:/tmp --mount /local:/local \
    "$image" bash -lc '
        set -euo pipefail
        export PATH=/opt/conda/envs/pointact/bin:/opt/conda/bin:$PATH
        export CONDA_PREFIX=/opt/conda/envs/pointact
        export PYTHONNOUSERSITE=1
        export PYTHONPATH=/mnt/home/weihangli/pointact_project/code/pointact_tasknet_v2_0f65a30
        export POINTACT_POLAR_ROUTE_CACHE=1
        export NETRC=/tmp/yuan/credentials/job-26196/netrc
        export WANDB_ENTITY=fengy7732-technical-university-of-munich
        export WANDB_PROJECT=pointact-rlbench-tasknet-v2
        export WANDB_MODE=online
        export WANDB_LOG_MODEL=false
        export WANDB_DIR=/tmp/yuan/wandb/job-26196/$TASK_RUN_NAME
        export WANDB_DATA_DIR=$WANDB_DIR/data
        export WANDB_CACHE_DIR=$WANDB_DIR/cache
        export WANDB_CONFIG_DIR=$WANDB_DIR/config
        export HF_HOME=/tmp/yuan/hf/job-26196
        export REPORT_TO=wandb
        export TASKNET_FREEZE=True
        export DATASET_ROOT=/tmp/yuan/datasets/job-26196/rlbench-v2/hybridvla_10tasks_train_keysteps_polar_rlbench9_v2
        export TASKNET_CHECKPOINT=/mnt/home/weihangli/pointact_project/code/pointact_tasknet_v2_0f65a30/pretrained/PolarAPP/TaskNet.pth
        export PTV3_INIT_CKPT_FILE=/mnt/home/weihangli/pointact_project/code/pointact_tasknet_v2_0f65a30/pretrained/Pointcept-Concerto/concerto_large.pth
        export VLM_PATH=/mnt/home/weihangli/pointact_project/models/Qwen2.5-VL-3B-Instruct
        export OUTPUT_DIR=/local/weihangli/checkpoints/$TASK_RUN_NAME
        export RUN_NAME=$TASK_RUN_NAME
        export PER_DEVICE_BATCH_SIZE=64
        export MAX_STEPS=$TASK_MAX_STEPS
        export SAVE_STEPS=$TASK_SAVE_STEPS
        export LEARNING_RATE=1.4e-4
        export LOGGING_STEPS=$TASK_LOGGING_STEPS
        export WARMUP_STEPS=0.03
        export DATALOADER_NUM_WORKERS=12
        export DATALOADER_PREFETCH_FACTOR=2
        cd /mnt/home/weihangli/pointact_project/code/pointact_tasknet_v2_0f65a30
        exec bash experiments/10_rlbench/train_10task_polar_rlbench9_v2_tasknet_depth_classify_h200.sh
    '
