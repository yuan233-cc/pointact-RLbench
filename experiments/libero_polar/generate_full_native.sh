#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
RAW_DIR="${RAW_DIR:-${ROOT}/6dor/Open6DOR_V2_Execution/libero_datasets/datasets/libero_spatial}"
SPP="${SPP:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/robot-PointAct/robot_data/libero/libero_spatial_polar_native_256px_${SPP}spp_full_500ep_v1}"

export PYTHONNOUSERSITE=1
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=egl
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/libero_polar_mplconfig}"
export PYTHONPATH="${ROOT}/6dor/LIBERO_official"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/tmp/libero_geovla_check}"
export NATIVE_POLAR_NVCC="${NATIVE_POLAR_NVCC:-/usr/local/cuda/bin/nvcc}"

cd "${ROOT}/robot-PointAct"
exec /home/hyunjun/anaconda3/envs/cavla3d/bin/python \
  experiments/libero_polar/replay_libero_spatial.py \
  --raw-dir "${RAW_DIR}" \
  --output "${OUTPUT_DIR}" \
  --task-ids 0 1 2 3 4 5 6 7 8 9 \
  --episodes-per-task 50 \
  --resolution 256 \
  --polar-backend native \
  --native-renderer-repo "${ROOT}/rlbench_custom_render/RLBench" \
  --materials "${ROOT}/robot-PointAct/experiments/libero_polar/libero_spatial_materials.json" \
  --spp "${SPP}" \
  --max-depth 8 \
  --device 0 \
  --omit-redundant-aolp \
  --min-free-gib 5 \
  --resume
