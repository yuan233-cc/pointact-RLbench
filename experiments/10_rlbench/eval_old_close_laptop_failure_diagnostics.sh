#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000"
OUTPUT_DIR="${PROJECT_ROOT}/RLBench_close_tasks_consistent_two_checkpoints_20260919/old_complete_laptop_diagnostic"
POINTACT_PYTHON="/home/hyunjun/anaconda3/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
HOST="127.0.0.1"
PORT="15541"

export COPPELIASIM_ROOT="${WORKSPACE_ROOT}/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"
export PYTHONNOUSERSITE="1"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}/logs"
cd "${PROJECT_ROOT}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${POINTACT_PYTHON}" -u experiments/10_rlbench/run_reseedable_incomplete25_server.py \
  --args.seed 7 \
  --args.pretrained-path "${CHECKPOINT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.num-denoise-steps 10 \
  --args.missing-rate 0 \
  >"${OUTPUT_DIR}/logs/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 480); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}"
    exit 1
  fi
  if nc -z "${HOST}" "${PORT}" 2>/dev/null; then
    break
  fi
  sleep 5
done

"${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_rlbench_client.py \
  --args.taskvar close_laptop_lid+0 \
  --args.seed 7 \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.replan-steps 1 \
  --args.max-steps 25 \
  --args.pretrained-path "${CHECKPOINT}" \
  --args.clip-within-workspace \
  --args.pred-rot-type euler \
  --args.save-dir "${OUTPUT_DIR}" \
  --args.num-workers 1 \
  --args.num-episodes 20 \
  --args.reset-policy-each-episode \
  --args.save-video \
  --args.project-action-on-image \
  --args.continuous-video-fps 2 \
  --args.save-obs-outs \
  2>&1 | tee "${OUTPUT_DIR}/logs/evaluation.log"
