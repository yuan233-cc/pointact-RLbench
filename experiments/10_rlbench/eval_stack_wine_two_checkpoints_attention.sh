#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
OUTPUT_ROOT="${PROJECT_ROOT}/RLBench_stack_wine_two_checkpoint_eval_20260918"
POINTACT_PYTHON="/home/hyunjun/anaconda3/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
HOST="127.0.0.1"
PORT="15518"

OLD_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000"
NEW_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-incomplete25-clf-concerto-hf/checkpoint-40000"

export COPPELIASIM_ROOT="${WORKSPACE_ROOT}/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"
export PYTHONNOUSERSITE="1"
export MPLCONFIGDIR="${OUTPUT_ROOT}/.matplotlib"

mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}"
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
  local run_dir="${OUTPUT_ROOT}/${label}"

  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
    return 1
  fi
  for required in config.json processor_config.json model.safetensors trainer_state.json; do
    if [[ ! -s "${checkpoint}/${required}" ]]; then
      echo "Missing checkpoint file: ${checkpoint}/${required}" >&2
      return 1
    fi
  done
  mkdir -p "${run_dir}/attention_captures" "${run_dir}/logs"

  "${POINTACT_PYTHON}" -u experiments/10_rlbench/run_reseedable_ptv3_action_attention_server.py \
    --args.seed 7 \
    --args.pretrained-path "${checkpoint}" \
    --args.num-denoise-steps 10 \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.capture-dir "${run_dir}/attention_captures" \
    --args.max-captures 0 \
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
    echo "Attention server did not become ready within 40 minutes." >&2
    return 1
  fi

  "${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_rlbench_client.py \
    --args.taskvar stack_wine+0 \
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
    --args.num-episodes 20 \
    --args.save-video \
    --args.project-action-on-image \
    --args.continuous-video-fps 2 \
    2>&1 | tee "${run_dir}/logs/evaluation.log"

  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""

  "${POINTACT_PYTHON}" -u experiments/10_rlbench/render_stack_wine_attention_rollouts.py \
    --run-dir "${run_dir}" \
    --checkpoint-label "${label}" \
    --checkpoint-path "${checkpoint}" \
    --expected-episodes 20 \
    --fps 3 \
    2>&1 | tee "${run_dir}/logs/render_attention.log"
}

run_checkpoint "baseline_job25521" "${OLD_CHECKPOINT}"
run_checkpoint "incomplete25_clf_concerto" "${NEW_CHECKPOINT}"

"${POINTACT_PYTHON}" -u experiments/10_rlbench/summarize_stack_wine_checkpoint_comparison.py \
  --output-root "${OUTPUT_ROOT}" \
  --runs \
    "${OUTPUT_ROOT}/baseline_job25521" \
    "${OUTPUT_ROOT}/incomplete25_clf_concerto"

echo "Completed: ${OUTPUT_ROOT}"
