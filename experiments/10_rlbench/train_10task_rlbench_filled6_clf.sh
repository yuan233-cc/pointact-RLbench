#!/usr/bin/env bash
set -euo pipefail

# Train the standard six-channel PointACT classifier on the repaired v2 filled
# point clouds. Polar columns are stripped and RGB images are not sent to VLM.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export DATASET_CONFIG="${DATASET_CONFIG:-experiments/10_rlbench/data_configs/data-10task-rlbench-filled6-clf.yaml}"
export DATASET_NAME="${DATASET_NAME:-keysteps-euler-points.filled6.frontview-no-vlm-rgb.30}"
export OUTPUT_DIR="${OUTPUT_DIR:-${repo_root}/checkpoints/rlbench/pointact-rlbench-filled6-v2-clf}"

# Match the complete-point-cloud classifier run (job 25521); only the dataset
# source changes. Environment overrides remain available for cluster launches.
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-128}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export MERGER_LR="${MERGER_LR:-5e-5}"
export VISION_LR="${VISION_LR:-2e-5}"
export EPOCHS="${EPOCHS:-1000}"
export MAX_STEPS="${MAX_STEPS:--1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"

exec bash "${repo_root}/experiments/10_rlbench/train_pointact_clf_concerto.sh"
