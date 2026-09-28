#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
RAW_DIR="${RAW_DIR:-${ROOT}/6dor/Open6DOR_V2_Execution/libero_datasets/datasets/libero_spatial}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/hyunjun/wli_data/libero_spatial_clean_geovla_npz_v1}"

export PYTHONNOUSERSITE=1
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=egl
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/libero_clean_mplconfig}"
export PYTHONPATH="${ROOT}/6dor/LIBERO_official"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/tmp/libero_geovla_check}"

cd "${ROOT}/robot-PointAct"
exec /home/hyunjun/anaconda3/envs/cavla3d/bin/python \
  experiments/libero_polar/replay_libero_spatial_clean.py \
  --raw-dir "${RAW_DIR}" \
  --output "${OUTPUT_DIR}" \
  --task-ids 0 1 2 3 4 5 6 7 8 9 \
  --episodes-per-task 50 \
  --resolution 256 \
  --writer-workers 4 \
  --max-pending-writes 16 \
  --min-free-gib 100 \
  --resume
