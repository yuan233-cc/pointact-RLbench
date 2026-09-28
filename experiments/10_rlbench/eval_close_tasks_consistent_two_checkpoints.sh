#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
OUTPUT_ROOT="${PROJECT_ROOT}/RLBench_close_tasks_consistent_two_checkpoints_20260919"
POINTACT_PYTHON="/home/hyunjun/anaconda3/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
OLD_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000"
NEW_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-incomplete25-clf-concerto-hf/checkpoint-40000"
HOST="127.0.0.1"
PORT="${EVAL_PORT:-15540}"
EPISODES="${EPISODES:-20}"
CORRUPTION_SEED="${CORRUPTION_SEED:-2026092001}"
TASKS=(close_box close_laptop_lid toilet_seat_down)

export COPPELIASIM_ROOT="${WORKSPACE_ROOT}/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"
export PYTHONNOUSERSITE="1"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

run_condition() {
  local label="$1"
  local checkpoint="$2"
  local missing_rate="$3"
  local condition_root="${OUTPUT_ROOT}/${label}"
  if [[ -e "${condition_root}" ]]; then
    echo "Refusing to overwrite ${condition_root}" >&2
    return 1
  fi
  mkdir -p "${condition_root}/logs"

  "${POINTACT_PYTHON}" -u experiments/10_rlbench/run_reseedable_incomplete25_server.py \
    --args.seed 7 \
    --args.pretrained-path "${checkpoint}" \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.num-denoise-steps 10 \
    --args.missing-rate "${missing_rate}" \
    --args.corruption-seed "${CORRUPTION_SEED}" \
    >"${condition_root}/logs/server.log" 2>&1 &
  SERVER_PID=$!

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
  if [[ "${ready}" != "1" ]]; then
    echo "Policy server did not become ready within 40 minutes." >&2
    return 1
  fi

  for task in "${TASKS[@]}"; do
    local run_dir="${condition_root}/${task}"
    mkdir -p "${run_dir}"
    "${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_rlbench_client.py \
      --args.taskvar "${task}+0" \
      --args.seed 7 \
      --args.host "${HOST}" \
      --args.port "${PORT}" \
      --args.replan-steps 1 \
      --args.max-steps 25 \
      --args.pretrained-path "${checkpoint}" \
      --args.clip-within-workspace \
      --args.pred-rot-type euler \
      --args.save-dir "${run_dir}" \
      --args.num-workers 1 \
      --args.num-episodes "${EPISODES}" \
      --args.reset-policy-each-episode \
      2>&1 | tee "${condition_root}/logs/${task}.log"
  done

  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

run_condition old_complete "${OLD_CHECKPOINT}" 0
run_condition new_incomplete25 "${NEW_CHECKPOINT}" 0.25

"${POINTACT_PYTHON}" -u experiments/10_rlbench/summarize_close_tasks_consistent.py \
  --output-root "${OUTPUT_ROOT}" \
  --episodes "${EPISODES}"

echo "Completed: ${OUTPUT_ROOT}"
