#!/usr/bin/env bash
# Nine-channel classifier on the repaired RLBench geometry and polarization.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_PATH="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-polar-rlbench9-v2-filled.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/pointact-rlbench-polar-rlbench9-v2-filled}"
export PTV3_INPUT_CHANNELS=9
export PTV3_INIT_COPY_INPUT_CHANNELS=6
exec bash "$repo_root/experiments/10_rlbench/train_10task_polar_filled9.sh"
