#!/usr/bin/env bash
# Filled9 training with the dense three-channel polar image sent to the VLM.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_PATH="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-polar-rlbench9-v2-filled-vlm-polar.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$(cd "$script_dir/../.." && pwd)/checkpoints/rlbench/pointact-rlbench-polar-filled9-vlm-polar-bs512-lr1e4}"
export COLOR_AUG=False
export IMAGE_AUG=False

exec "$script_dir/train_10task_polar_filled9.sh"
