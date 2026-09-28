#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
POINTACT_PYTHON="${WORKSPACE_ROOT}/.conda/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-filled6-v2-cls-step40000}"
EPISODES="${EPISODES:-25}"
MAX_STEPS="${MAX_STEPS:-25}"
SEED="${SEED:-7}"
CORRUPTION_SEED="${CORRUPTION_SEED:-20260923}"
EVAL_EPISODE_OFFSET="${EVAL_EPISODE_OFFSET:-10000}"
WORKERS="${WORKERS:-4}"
PORT="${PORT:-15563}"
RUN_TAG="${RUN_TAG:-25ep_seed7_original_rng_parallel4_20260925}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/filled6_step40000_${RUN_TAG}"
TASKS=(
    close_box close_laptop_lid toilet_seat_down sweep_to_dustpan close_fridge
    phone_on_base take_umbrella_out_of_umbrella_stand take_frame_off_hanger
    stack_wine water_plants
)

export COPPELIASIM_ROOT="${WORKSPACE_ROOT}/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${WORKSPACE_ROOT}/rlbench_custom_render/RLBench:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="/tmp/pointact-matplotlib-filled6-step40000"

if (( WORKERS < 1 )); then
    echo "WORKERS must be positive" >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}/model.safetensors" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 2
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "Local NVIDIA GPU is unavailable; refusing to start a partial evaluation." >&2
    exit 3
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"
cp experiments/10_rlbench/filled9_inference.py "${OUTPUT_ROOT}/filled9_inference_snapshot.py"
cp experiments/10_rlbench/run_filled6_rlbench.py "${OUTPUT_ROOT}/rlbench_client_snapshot.py"
cp experiments/10_rlbench/run_filled6_server.py "${OUTPUT_ROOT}/server_snapshot.py"
cp experiments/10_rlbench/merge_filled9_parallel_results.py "${OUTPUT_ROOT}/merge_results_snapshot.py"
cp "${BASH_SOURCE[0]}" "${OUTPUT_ROOT}/launch.sh"

SERVER_PID=""
WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    for _ in $(seq 1 240); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            wait "${SERVER_PID}"
            return 1
        fi
        if ss -ltnH "sport = :${PORT}" | grep -q .; then
            return 0
        fi
        sleep 5
    done
    echo "Policy server did not become ready within 20 minutes" >&2
    return 1
}

run_task_worker() {
    local task_name="$1"
    local worker_dir="${OUTPUT_ROOT}/workers"
    local output="${worker_dir}/${task_name}.json"
    local progress="${worker_dir}/${task_name}.progress.jsonl"
    mkdir -p "${worker_dir}"
    if [[ -e "${output}" ]]; then
        echo "[LOCAL] ${task_name}: already complete"
        return 0
    fi
    if [[ -e "${progress}" ]]; then
        mv "${progress}" "${progress}.interrupted.$(date +%s)"
    fi
    "${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_filled6_rlbench.py \
        --tasks "${task_name}" \
        --episodes "${EPISODES}" \
        --max-steps "${MAX_STEPS}" \
        --seed "${SEED}" \
        --corruption-seed "${CORRUPTION_SEED}" \
        --eval-episode-offset "${EVAL_EPISODE_OFFSET}" \
        --checkpoint-step 40000 \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --output "${output}" \
        >"${OUTPUT_ROOT}/logs/${task_name}.log" 2>&1
}

echo "[LOCAL] Starting 10 tasks x ${EPISODES} episodes with ${WORKERS} workers"
"${POINTACT_PYTHON}" -u experiments/10_rlbench/run_filled6_server.py \
    --pretrained-path "${CHECKPOINT}" \
    --seed "${SEED}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --num-denoise-steps 10 \
    >"${OUTPUT_ROOT}/logs/server.log" 2>&1 &
SERVER_PID=$!
wait_for_server

failed=0
for task_name in "${TASKS[@]}"; do
    run_task_worker "${task_name}" &
    WORKER_PIDS+=("$!")
    if (( ${#WORKER_PIDS[@]} == WORKERS )); then
        for pid in "${WORKER_PIDS[@]}"; do
            wait "${pid}" || failed=1
        done
        WORKER_PIDS=()
        (( failed == 0 )) || exit 1
    fi
done
for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}" || failed=1
done
WORKER_PIDS=()
(( failed == 0 ))

"${RLBENCH_PYTHON}" experiments/10_rlbench/merge_filled9_parallel_results.py \
    --worker-dir "${OUTPUT_ROOT}/workers" \
    --output "${OUTPUT_ROOT}/filled6_step40000.json" \
    --checkpoint-step 40000 \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --corruption-seed "${CORRUPTION_SEED}" \
    --eval-episode-offset "${EVAL_EPISODE_OFFSET}"

echo "[LOCAL] Completed: ${OUTPUT_ROOT}/filled6_step40000.json"
