#!/usr/bin/env bash
# Strict XYZRGB control for the filled9 polar-feature ablation suite.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_PATH="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-rlbench-filled6-clf-no-rot.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/pointact-rlbench-xyzrgb-filled6-matched-bs512-lr1e4}"
export PTV3_INPUT_CHANNELS=6
export PTV3_INIT_COPY_INPUT_CHANNELS=6

exec bash "$repo_root/experiments/10_rlbench/train_10task_polar_filled9.sh"
