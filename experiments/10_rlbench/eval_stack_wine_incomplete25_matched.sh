#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
OUTPUT_ROOT="${PROJECT_ROOT}/RLBench_stack_wine_two_checkpoint_eval_incomplete25_matched_20260918"
RUN_DIR="${OUTPUT_ROOT}/incomplete25_clf_concerto_matched"
CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-incomplete25-clf-concerto-hf/checkpoint-40000"
BASELINE_RUN="${PROJECT_ROOT}/RLBench_stack_wine_two_checkpoint_eval_20260918/baseline_job25521"
POINTACT_PYTHON="/home/hyunjun/anaconda3/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
HOST="127.0.0.1"
PORT="15519"

export COPPELIASIM_ROOT="${WORKSPACE_ROOT}/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"
export PYTHONNOUSERSITE="1"
export MPLCONFIGDIR="${OUTPUT_ROOT}/.matplotlib"

if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/attention_captures" "${RUN_DIR}/logs" "${MPLCONFIGDIR}"
cd "${PROJECT_ROOT}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${POINTACT_PYTHON}" -u experiments/10_rlbench/run_incomplete25_ptv3_action_attention_server.py \
  --args.seed 7 \
  --args.pretrained-path "${CHECKPOINT}" \
  --args.num-denoise-steps 10 \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.capture-dir "${RUN_DIR}/attention_captures" \
  --args.max-captures 0 \
  >"${RUN_DIR}/logs/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 480); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}"
    exit 1
  fi
  if nc -z "${HOST}" "${PORT}" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "${ready}" != "1" ]]; then
  echo "Incomplete25 attention server did not become ready." >&2
  exit 1
fi

"${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_rlbench_client.py \
  --args.taskvar stack_wine+0 \
  --args.seed 7 \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.replan-steps 1 \
  --args.max-steps 25 \
  --args.pretrained-path "${CHECKPOINT}" \
  --args.clip-within-workspace \
  --args.pred-rot-type euler \
  --args.save-dir "${RUN_DIR}" \
  --args.num-workers 1 \
  --args.num-episodes 20 \
  --args.reset-policy-each-episode \
  --args.save-video \
  --args.project-action-on-image \
  --args.continuous-video-fps 2 \
  2>&1 | tee "${RUN_DIR}/logs/evaluation.log"

kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
SERVER_PID=""

"${POINTACT_PYTHON}" -u experiments/10_rlbench/render_stack_wine_attention_rollouts.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint-label "incomplete25_clf_concerto_matched" \
  --checkpoint-path "${CHECKPOINT}" \
  --expected-episodes 20 \
  --fps 3 \
  2>&1 | tee "${RUN_DIR}/logs/render_attention.log"

"${POINTACT_PYTHON}" -u experiments/10_rlbench/summarize_stack_wine_checkpoint_comparison.py \
  --output-root "${OUTPUT_ROOT}" \
  --runs "${BASELINE_RUN}" "${RUN_DIR}"

echo "Completed: ${OUTPUT_ROOT}"
