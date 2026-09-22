#!/usr/bin/env bash
# Same filled9 policy, with a training-only interaction-surface U-Net loss.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_PATH="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-polar-filled9-target-reconstruction.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/pointact-rlbench-polar-filled9-target-reconstruction}"
export USE_TARGET_RECONSTRUCTION=True
export PTV3_APPLY_POINT_CA=True
export TARGET_RECONSTRUCTION_WEIGHT="${TARGET_RECONSTRUCTION_WEIGHT:-2.5}"
export TARGET_RECONSTRUCTION_MAX_POINTS="${TARGET_RECONSTRUCTION_MAX_POINTS:-512}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
exec bash "$repo_root/experiments/10_rlbench/train_10task_polar_filled9.sh"
