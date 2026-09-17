#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000"
OUTPUT="${PROJECT_ROOT}/PTV3_wine_umbrella_realistic_missing_20260917"
POINTACT_PYTHON="/home/hyunjun/anaconda3/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
PORT=15507

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

"${POINTACT_PYTHON}" -u experiments/10_rlbench/run_reseedable_policy_server.py \
  --args.seed 7 --args.pretrained-path "${CHECKPOINT}" --args.num-denoise-steps 10 \
  --args.host 127.0.0.1 --args.port "${PORT}" &
SERVER_PID=$!
READY=0
for _ in $(seq 1 480); do
  if nc -z 127.0.0.1 "${PORT}" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then wait "${SERVER_PID}"; exit 1; fi
  sleep 5
done
if [[ "${READY}" != "1" ]]; then exit 1; fi

for SPEC in "stack_wine:wine_bottle" "take_umbrella_out_of_umbrella_stand:umbrella"; do
  TASK="${SPEC%%:*}"
  TARGET="${SPEC##*:}"
  for SEVERITY in 0.00 0.25 0.50 0.75; do
    CONDITION_DIR="${OUTPUT}/${TASK}/severity_${SEVERITY}"
    mkdir -p "${CONDITION_DIR}"
    "${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_task_realistic_missing_client.py \
      --args.task "${TASK}" --args.target-object "${TARGET}" \
      --args.output-dir "${CONDITION_DIR}" --args.host 127.0.0.1 --args.port "${PORT}" \
      --args.severity "${SEVERITY}" --args.variation 0 --args.seed 7 \
      --args.num-episodes 10 --args.max-steps 25
  done
done

"${POINTACT_PYTHON}" -u experiments/10_rlbench/summarize_wine_umbrella_realistic_missing.py
