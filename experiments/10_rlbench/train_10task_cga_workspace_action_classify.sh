#!/usr/bin/env bash
# Aligned frozen native-CGA teacher + Concerto + action classification.
# This launches training only; it does not request any cluster allocation.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export POLAR_BACKBONE=cga_dinov3_normal
export CGA_INPUT_MODE=native_cga
export PTV3_BACKEND=concerto
export POLAR_FUSION_MODE=workspace
export POLAR_DEPTH_SUPERVISION_MODE=weighted_workspace
export POLAR_BBOX_FEATURE_LEVELS="0 1 2 3 4"
export POLAR_WORKSPACE_ATTEND_ACTION="${POLAR_WORKSPACE_ATTEND_ACTION:-True}"
export POLAR_HOLE_NORMAL_WEIGHT="${POLAR_HOLE_NORMAL_WEIGHT:-3.0}"
export POLAR_DEPTH_LOSS_WEIGHT="${POLAR_DEPTH_LOSS_WEIGHT:-1.0}"
export SPARSE_DEPTH_WEIGHT="${SPARSE_DEPTH_WEIGHT:-1.0}"
export DEPTH_SMOOTHNESS_WEIGHT=0.0
export PTV3_PATCH_SIZE="${PTV3_PATCH_SIZE:-256}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export MAX_STEPS="${MAX_STEPS:-30000}"
export DATA_PATH="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-polar-rlbench9-v2-cga-workspace-action.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/cga-workspace-action-classify}"
exec bash "$repo_root/experiments/10_rlbench/train_10task_polar_rlbench9_v2_tasknet_depth_classify_h200.sh" "$@"
