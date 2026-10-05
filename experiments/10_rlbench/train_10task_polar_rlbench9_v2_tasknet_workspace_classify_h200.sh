#!/usr/bin/env bash
# TaskNet(64x64, frozen) + Concerto workspace-memory geometry experiment.
# Point self-attention is unchanged; only point Q reads the stage's dense polar KV.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PTV3_BACKEND=concerto
export POLAR_FUSION_MODE=workspace
export TASKNET_INPUT_SIZE="${TASKNET_INPUT_SIZE:-64}"
export TASKNET_FREEZE="${TASKNET_FREEZE:-True}"
export POLAR_WORKSPACE_ATTEND_ACTION="${POLAR_WORKSPACE_ATTEND_ACTION:-False}"
export PTV3_APPLY_POINT_CA="${PTV3_APPLY_POINT_CA:-False}"
export DATA_PATH="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-polar-rlbench9-v2-tasknet-workspace.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/pointact-rlbench9-v2-tasknet-workspace-classify-h200}"

# Geometry objective: TaskNet normal consistency remains dominant, while a
# small held-out sparse-depth term anchors the decoder's metric depth scale.
export POLAR_DEPTH_LOSS_WEIGHT="${POLAR_DEPTH_LOSS_WEIGHT:-1.0}"
export POLAR_CONSISTENCY_WEIGHT="${POLAR_CONSISTENCY_WEIGHT:-1.0}"
export SPARSE_DEPTH_WEIGHT="${SPARSE_DEPTH_WEIGHT:-0.1}"
export DEPTH_SMOOTHNESS_WEIGHT="${DEPTH_SMOOTHNESS_WEIGHT:-0.0}"

# Dense workspace attention is quadratic in points x workspace pixels.  Start at
# eight samples/H200; raise only after the measured peak-memory smoke run.
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

exec bash "$repo_root/experiments/10_rlbench/train_10task_polar_rlbench9_v2_tasknet_depth_classify_h200.sh" "$@"
