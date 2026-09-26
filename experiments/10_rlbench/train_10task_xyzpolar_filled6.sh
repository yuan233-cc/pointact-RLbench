#!/usr/bin/env bash
# Six-channel PointACT classifier using XYZ + aligned polarization, without RGB.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_PATH="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-xyzpolar-filled6.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/pointact-rlbench-xyzpolar-filled6-bs512-lr1e4}"
export PTV3_INPUT_CHANNELS=6
# The polar tuple is mapped through the same [0, 1] -> [-1, 1] interface as
# RGB, so initialize all six target columns from Concerto's XYZRGB stem.
export PTV3_INIT_COPY_INPUT_CHANNELS=6

exec bash "$repo_root/experiments/10_rlbench/train_10task_polar_filled9.sh"
