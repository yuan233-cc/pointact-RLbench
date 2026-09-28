#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/RLBench_take_frame_failure_attention_paired_seed7_20260920}"
POINTACT_PYTHON="/home/hyunjun/anaconda3/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
OLD_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000"
NEW_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-incomplete25-clf-concerto-hf/checkpoint-40000"
HOST="127.0.0.1"
PORT="${EVAL_PORT:-15550}"
EPISODES="${EPISODES:-50}"
CORRUPTION_SEED=2026091801
CAPTURE_REQUESTS=(0 1 2 3 9 19 24)

export COPPELIASIM_ROOT="${WORKSPACE_ROOT}/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"
export PYTHONNOUSERSITE="1"

cd "${PROJECT_ROOT}"
if nc -z "${HOST}" "${PORT}" 2>/dev/null; then
  echo "[LOCAL] Port ${PORT} is occupied." >&2
  exit 1
fi
mkdir "${OUTPUT_ROOT}"
cp "${BASH_SOURCE[0]}" "${OUTPUT_ROOT}/launch.sh"
cp experiments/10_rlbench/run_take_frame_geometry_client.py "${OUTPUT_ROOT}/geometry_client_snapshot.py"
cp experiments/10_rlbench/run_ptv3_action_attention_server.py "${OUTPUT_ROOT}/attention_server_snapshot.py"
cp experiments/10_rlbench/ptv3_action_attention.py "${OUTPUT_ROOT}/attention_capture_snapshot.py"
cp experiments/10_rlbench/analyze_take_frame_failure_attention.py "${OUTPUT_ROOT}/analysis_snapshot.py"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local ready=0
  for _ in $(seq 1 480); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}"
      return 1
    fi
    if nc -z "${HOST}" "${PORT}" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 5
  done
  [[ "${ready}" == "1" ]] || return 1
}

run_condition() {
  local label="$1"
  local checkpoint="$2"
  local server_script="$3"
  local condition_root="${OUTPUT_ROOT}/${label}"
  mkdir -p "${condition_root}/logs" "${condition_root}/attention" "${condition_root}/eval"
  echo "[LOCAL] ${label}: take_frame_off_hanger x ${EPISODES}, paired episode seeds."

  local server_args=(
    --args.seed 7
    --args.pretrained-path "${checkpoint}"
    --args.host "${HOST}"
    --args.port "${PORT}"
    --args.num-denoise-steps 10
    --args.capture-dir "${condition_root}/attention"
    --args.max-captures 0
    --args.capture-request-indices-per-episode "${CAPTURE_REQUESTS[@]}"
  )
  if [[ "${label}" == "new_incomplete25" ]]; then
    server_args+=(--args.missing-rate 0.25 --args.corruption-seed "${CORRUPTION_SEED}")
  fi
  "${POINTACT_PYTHON}" -u "${server_script}" "${server_args[@]}" \
    >"${condition_root}/logs/server.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server

  "${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_take_frame_geometry_client.py \
    --args.taskvar take_frame_off_hanger+0 \
    --args.seed 7 \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.replan-steps 1 \
    --args.max-steps 25 \
    --args.pretrained-path "${checkpoint}" \
    --args.clip-within-workspace \
    --args.pred-rot-type euler \
    --args.save-dir "${condition_root}/eval" \
    --args.num-workers 1 \
    --args.num-episodes "${EPISODES}" \
    --args.reset-policy-each-episode \
    --args.reset-environment-rng-each-episode \
    2>&1 | tee "${condition_root}/logs/evaluation.log"

  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

run_condition \
  old_complete "${OLD_CHECKPOINT}" \
  experiments/10_rlbench/run_reseedable_ptv3_action_attention_server.py
run_condition \
  new_incomplete25 "${NEW_CHECKPOINT}" \
  experiments/10_rlbench/run_incomplete25_ptv3_action_attention_server.py

"${POINTACT_PYTHON}" -u experiments/10_rlbench/analyze_take_frame_failure_attention.py \
  --root "${OUTPUT_ROOT}" | tee "${OUTPUT_ROOT}/analysis.log"

echo "[LOCAL] Completed: ${OUTPUT_ROOT}"
