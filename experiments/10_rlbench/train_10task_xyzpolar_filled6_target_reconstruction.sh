#!/usr/bin/env bash
# XYZ + polarization classifier with training-only interaction reconstruction.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_PATH="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-xyzpolar-filled6-target-reconstruction.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/pointact-rlbench-xyzpolar-filled6-reconstruction-bs128}"
export USE_TARGET_RECONSTRUCTION=True
export PTV3_APPLY_POINT_CA=True
export TARGET_RECONSTRUCTION_WEIGHT="${TARGET_RECONSTRUCTION_WEIGHT:-2.5}"
export TARGET_RECONSTRUCTION_MAX_POINTS="${TARGET_RECONSTRUCTION_MAX_POINTS:-512}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-128}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
export SAVE_STEPS="${SAVE_STEPS:-32}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"

exec bash "$repo_root/experiments/10_rlbench/train_10task_xyzpolar_filled6.sh"
