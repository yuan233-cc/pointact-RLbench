#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
RAW_DIR="${RAW_DIR:-${ROOT}/6dor/Open6DOR_V2_Execution/libero_datasets/datasets/libero_spatial}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/hyunjun/wli_data/libero_spatial_clean_geovla_npz_v1}"
PARALLEL_TASKS="${PARALLEL_TASKS:-4}"

export PYTHONNOUSERSITE=1
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=egl
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/libero_clean_mplconfig}"
export PYTHONPATH="${ROOT}/6dor/LIBERO_official"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/tmp/libero_geovla_check}"

cd "${ROOT}/robot-PointAct"

generate_task() {
  local task_id="$1"
  /home/hyunjun/anaconda3/envs/cavla3d/bin/python \
    experiments/libero_polar/replay_libero_spatial_clean.py \
    --raw-dir "${RAW_DIR}" \
    --output "${OUTPUT_DIR}" \
    --task-ids "${task_id}" \
    --episodes-per-task 50 \
    --resolution 256 \
    --writer-workers 2 \
    --max-pending-writes 8 \
    --min-free-gib 100 \
    --resume \
    --worker-fragment
}
export -f generate_task
export RAW_DIR OUTPUT_DIR

seq 0 9 | xargs -n 1 -P "${PARALLEL_TASKS}" bash -c 'generate_task "$1"' _

# Re-open the completed summaries once to produce the authoritative 500-episode manifest.
exec bash experiments/libero_polar/generate_full_clean_geovla.sh
