#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000"
OUTPUT="${PROJECT_ROOT}/PTV3_wine_umbrella_realistic_missing_20260917"
POINTACT_PYTHON="/home/hyunjun/anaconda3/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
PORT=15508

export COPPELIASIM_ROOT="${WORKSPACE_ROOT}/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${OUTPUT}/.matplotlib"

mkdir -p "${OUTPUT}"
cd "${PROJECT_ROOT}"
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${POINTACT_PYTHON}" -u experiments/10_rlbench/run_reseedable_ptv3_action_attention_server.py \
  --args.seed 7 --args.pretrained-path "${CHECKPOINT}" --args.num-denoise-steps 10 \
  --args.host 127.0.0.1 --args.port "${PORT}" \
  --args.capture-dir "${OUTPUT}/attention_captures" --args.max-captures 8 &
SERVER_PID=$!
READY=0
for _ in $(seq 1 480); do
  if nc -z 127.0.0.1 "${PORT}" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then wait "${SERVER_PID}"; exit 1; fi
  sleep 5
done
if [[ "${READY}" != "1" ]]; then exit 1; fi

"${RLBENCH_PYTHON}" -u experiments/10_rlbench/capture_wine_umbrella_realistic_attention.py

kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
SERVER_PID=""

"${POINTACT_PYTHON}" -u experiments/10_rlbench/render_wine_umbrella_realistic_3d_attention.py
