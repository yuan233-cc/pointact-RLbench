#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
OUTPUT_ROOT="${PROJECT_ROOT}/RLBench_remaining7_tasks_50eps_episode_rng_seed2026091801_20260919"
POINTACT_PYTHON="/home/hyunjun/anaconda3/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
OLD_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000"
NEW_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-incomplete25-clf-concerto-hf/checkpoint-40000"
HOST="127.0.0.1"
PORT="${EVAL_PORT:-15546}"
EPISODES=50
CORRUPTION_SEED=2026091801
TASKS=(
  sweep_to_dustpan
  close_fridge
  phone_on_base
  take_umbrella_out_of_umbrella_stand
  take_frame_off_hanger
  stack_wine
  water_plants
)

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
  echo "[LOCAL] Port ${PORT} is already occupied; refusing a duplicate launch." >&2
  exit 1
fi
for checkpoint in "${OLD_CHECKPOINT}" "${NEW_CHECKPOINT}"; do
  for required in config.json processor_config.json model.safetensors trainer_state.json; do
    if [[ ! -s "${checkpoint}/${required}" ]]; then
      echo "[LOCAL] Missing checkpoint file: ${checkpoint}/${required}" >&2
      exit 1
    fi
  done
done

# mkdir without -p refuses existing results, including concurrent launches.
mkdir "${OUTPUT_ROOT}"
cp "${BASH_SOURCE[0]}" "${OUTPUT_ROOT}/launch.sh"
cp experiments/10_rlbench/run_reseedable_incomplete25_server.py "${OUTPUT_ROOT}/policy_server_snapshot.py"
cp experiments/10_rlbench/summarize_remaining7_tasks.py "${OUTPUT_ROOT}/summarizer_snapshot.py"

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
  if [[ "${ready}" != "1" ]]; then
    echo "[LOCAL] Policy server did not become ready within 40 minutes." >&2
    return 1
  fi
}

run_condition() {
  local label="$1"
  local checkpoint="$2"
  local missing_rate="$3"
  local condition_root="${OUTPUT_ROOT}/${label}"
  mkdir -p "${condition_root}/logs"
  echo "[LOCAL] Starting ${label}: 7 tasks x ${EPISODES} episodes; episode RNG = 7 + episode_id."
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
  wait_for_server

  for task in "${TASKS[@]}"; do
    local run_dir="${condition_root}/${task}"
    mkdir "${run_dir}"
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

run_condition old_complete "${OLD_CHECKPOINT}" 0.0
run_condition new_incomplete25 "${NEW_CHECKPOINT}" 0.25

"${POINTACT_PYTHON}" -u experiments/10_rlbench/summarize_remaining7_tasks.py \
  --output-root "${OUTPUT_ROOT}" \
  --episodes "${EPISODES}" \
  --rng-mode episode-reset \
  --corruption-seed "${CORRUPTION_SEED}"

echo "[LOCAL] Completed: ${OUTPUT_ROOT}"
