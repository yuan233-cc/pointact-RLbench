#!/usr/bin/env bash
# Same repaired nine-channel classifier with interaction-surface reconstruction.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_PATH="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-polar-rlbench9-v2-filled-target-reconstruction.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/pointact-rlbench-polar-rlbench9-v2-reconstruction-bs128}"
exec bash "$repo_root/experiments/10_rlbench/train_10task_polar_filled9_target_reconstruction.sh"
