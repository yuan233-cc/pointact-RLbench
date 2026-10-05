#!/usr/bin/env bash
# Opt-in candidate only. Do not replace or resume the formal projection run.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export POLAR_FUSION_MODE=bbox
export DATA_PATH="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-polar-rlbench9-v2-tasknet-bbox.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/pointact-rlbench9-v2-tasknet-bbox-classify-h200}"
export TASKNET_FREEZE="${TASKNET_FREEZE:-True}"
# Alpha remains neutral until a real-data coverage probe has been run.
export POLAR_BBOX_EXPANSION="${POLAR_BBOX_EXPANSION:-1.0 1.0 1.0 1.0 1.0}"
export POLAR_BBOX_GRID_SIZE="${POLAR_BBOX_GRID_SIZE:-4}"
export POLAR_BBOX_FEATURE_LEVELS="${POLAR_BBOX_FEATURE_LEVELS:-0 0 1 2 2}"
exec bash "$repo_root/experiments/10_rlbench/train_10task_polar_rlbench9_v2_tasknet_depth_classify_h200.sh" "$@"
