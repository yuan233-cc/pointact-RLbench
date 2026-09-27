#!/usr/bin/env bash
set -euo pipefail

# Matched XYZRGB control on the repaired v2 point cloud. Both point-cloud yaw
# rotation and RGB augmentation are disabled; all other filled6 settings match
# the previous no-rotation classifier run.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export DATASET_CONFIG="${DATASET_CONFIG:-experiments/10_rlbench/data_configs/data-10task-rlbench-filled6-clf-no-rot-no-color-aug.yaml}"
export DATASET_NAME="${DATASET_NAME:-keysteps-euler-points.filled6.frontview-no-vlm-rgb.rot0.no-color-aug}"
export OUTPUT_DIR="${OUTPUT_DIR:-${repo_root}/checkpoints/rlbench/pointact-rlbench-filled6-v2-clf-no-rot-no-color-aug}"

export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-128}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export MERGER_LR="${MERGER_LR:-5e-5}"
export VISION_LR="${VISION_LR:-2e-5}"
export EPOCHS="${EPOCHS:-1000}"
export MAX_STEPS="${MAX_STEPS:--1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
export COLOR_AUG=False

exec bash "${repo_root}/experiments/10_rlbench/train_pointact_clf_concerto.sh"
