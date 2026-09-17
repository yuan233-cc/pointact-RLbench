#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000"
OUTPUT="${PROJECT_ROOT}/PTV3_close_fridge_incomplete_study_20260916/success_eval"
POINTACT_PYTHON="${WORKSPACE_ROOT}/.conda/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
BACKEND_PORT=15495
PROXY_PORT=15496

export COPPELIASIM_ROOT="${WORKSPACE_ROOT}/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT}"
cd "${PROJECT_ROOT}"
BACKEND_PID=""; PROXY_PID=""
cleanup() {
  if [[ -n "${PROXY_PID}" ]] && kill -0 "${PROXY_PID}" 2>/dev/null; then kill "${PROXY_PID}" 2>/dev/null || true; wait "${PROXY_PID}" 2>/dev/null || true; fi
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then kill "${BACKEND_PID}" 2>/dev/null || true; wait "${BACKEND_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

"${POINTACT_PYTHON}" -u scripts/run_server.py \
  --args.seed 7 --args.pretrained_path "${CHECKPOINT}" --args.num_denoise_steps 10 \
  --args.host 127.0.0.1 --args.port "${BACKEND_PORT}" &
BACKEND_PID=$!
for _ in $(seq 1 480); do
  if ss -ltnH "sport = :${BACKEND_PORT}" | rg -q .; then break; fi
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then wait "${BACKEND_PID}"; exit 1; fi
  sleep 5
done

for RATE in 0.00 0.25 0.50 0.75; do
  RATE_DIR="${OUTPUT}/missing_${RATE}"
  mkdir -p "${RATE_DIR}"
  "${POINTACT_PYTHON}" -u experiments/10_rlbench/run_point_dropout_proxy.py \
    --args.backend-host 127.0.0.1 --args.backend-port "${BACKEND_PORT}" \
    --args.host 127.0.0.1 --args.port "${PROXY_PORT}" --args.missing-rate "${RATE}" \
    --args.seed 701 --args.log-file "${RATE_DIR}/proxy.jsonl" &
  PROXY_PID=$!
  for _ in $(seq 1 120); do
    if ss -ltnH "sport = :${PROXY_PORT}" | rg -q .; then break; fi
    if ! kill -0 "${PROXY_PID}" 2>/dev/null; then wait "${PROXY_PID}"; exit 1; fi
    sleep 2
  done
  "${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_rlbench_client.py \
    --args.taskvar close_fridge+0 --args.seed 7 --args.host 127.0.0.1 --args.port "${PROXY_PORT}" \
    --args.replan-steps 1 --args.max-steps 25 --args.pretrained-path "${CHECKPOINT}" \
    --args.clip-within-workspace --args.pred-rot-type euler --args.save-dir "${RATE_DIR}" \
    --args.num-workers 1 --args.num-episodes 20
  kill "${PROXY_PID}" 2>/dev/null || true
  wait "${PROXY_PID}" 2>/dev/null || true
  PROXY_PID=""
done
