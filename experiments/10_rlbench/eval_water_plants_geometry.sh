#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/RLBench_water_plants_geometry_diagnostic_20260919}"
POINTACT_PYTHON="/home/hyunjun/anaconda3/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
OLD_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000"
NEW_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-incomplete25-clf-concerto-hf/checkpoint-40000"
HOST="127.0.0.1"
PORT="${EVAL_PORT:-15530}"
NUM_EPISODES="${NUM_EPISODES:-20}"

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

run_checkpoint() {
  local label="$1"
  local checkpoint="$2"
  local server_script="$3"
  local run_dir="${OUTPUT_ROOT}/${label}"
  shift 3
  local extra_server_args=("$@")

  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
    return 1
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/attention_captures"

  "${POINTACT_PYTHON}" -u "${server_script}" \
    --args.seed 7 \
    --args.pretrained-path "${checkpoint}" \
    --args.num-denoise-steps 10 \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    "${extra_server_args[@]}" \
    >"${run_dir}/logs/server.log" 2>&1 &
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

  "${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_water_plants_geometry_client.py \
    --args.taskvar water_plants+0 \
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
    --args.num-episodes "${NUM_EPISODES}" \
    --args.reset-policy-each-episode \
    --args.save-video \
    --args.project-action-on-image \
    --args.continuous-video-fps 2 \
    2>&1 | tee "${run_dir}/logs/evaluation.log"

  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

run_checkpoint \
  old_complete \
  "${OLD_CHECKPOINT}" \
  experiments/10_rlbench/run_reseedable_ptv3_action_attention_server.py \
  --args.capture-dir "${OUTPUT_ROOT}/old_complete/attention_captures" \
  --args.max-captures 0

run_checkpoint \
  new_incomplete25_matched \
  "${NEW_CHECKPOINT}" \
  experiments/10_rlbench/run_incomplete25_replay_attention_server.py \
  --args.corruption-seed 20260917 \
  --args.replay-training-corruption \
  --args.capture-dir "${OUTPUT_ROOT}/new_incomplete25_matched/attention_captures" \
  --args.max-captures 0

echo "Completed geometry diagnostic: ${OUTPUT_ROOT}"
