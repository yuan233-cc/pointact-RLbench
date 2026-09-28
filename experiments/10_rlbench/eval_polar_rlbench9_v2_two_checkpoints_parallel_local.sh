#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
POINTACT_PYTHON="${WORKSPACE_ROOT}/.conda/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
CLS_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-polar9-v2-cls-step10000"
RECON_CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-polar9-v2-recon-step17728"
EPISODES="${EPISODES:-25}"
MAX_STEPS="${MAX_STEPS:-25}"
SEED="${SEED:-7}"
CORRUPTION_SEED="${CORRUPTION_SEED:-20260923}"
EVAL_EPISODE_OFFSET="${EVAL_EPISODE_OFFSET:-10000}"
WORKERS="${WORKERS:-4}"
RUN_TAG="${RUN_TAG:-25ep_seed7_original_rng_parallel4_20260924}"
PORT="${PORT:-15562}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/polar9_v2_two_checkpoint_${RUN_TAG}"
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
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"
export PYTHONNOUSERSITE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export MPLCONFIGDIR="/tmp/pointact-matplotlib-polar9-v2-parallel"

if (( WORKERS < 1 )); then
    echo "WORKERS must be positive" >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"
cp experiments/10_rlbench/filled9_inference.py "${OUTPUT_ROOT}/filled9_inference_snapshot.py"
cp experiments/10_rlbench/run_filled9_rlbench.py "${OUTPUT_ROOT}/rlbench_client_snapshot.py"
cp experiments/10_rlbench/run_filled9_server.py "${OUTPUT_ROOT}/classification_server_snapshot.py"
cp experiments/10_rlbench/run_filled9_reconstruction_action_only_server.py \
   "${OUTPUT_ROOT}/reconstruction_server_snapshot.py"
cp experiments/10_rlbench/merge_filled9_parallel_results.py \
   "${OUTPUT_ROOT}/merge_results_snapshot.py"
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
    local label="$1"
    local step="$2"
    local task_name="$3"
    local worker_dir="${OUTPUT_ROOT}/${label}.workers"
    local output="${worker_dir}/${task_name}.json"
    local progress="${worker_dir}/${task_name}.progress.jsonl"
    mkdir -p "${worker_dir}"
    if [[ -e "${output}" ]]; then
        echo "[LOCAL] ${label}/${task_name}: already complete"
        return 0
    fi
    if [[ -e "${progress}" ]]; then
        # Replaying only the unfinished suffix would restart the task RNG stream
        # at the wrong state. Preserve the interrupted rows for diagnosis and
        # rerun this one task from episode zero.
        mv "${progress}" "${progress}.interrupted.$(date +%s)"
    fi
    "${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_filled9_rlbench.py \
        --tasks "${task_name}" \
        --episodes "${EPISODES}" \
        --max-steps "${MAX_STEPS}" \
        --seed "${SEED}" \
        --corruption-seed "${CORRUPTION_SEED}" \
        --eval-episode-offset "${EVAL_EPISODE_OFFSET}" \
        --checkpoint-step "${step}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --output "${output}" \
        >"${OUTPUT_ROOT}/logs/${label}_${task_name}.log" 2>&1
}

run_worker_batch() {
    local label="$1"
    local step="$2"
    local task_name pid failed
    WORKER_PIDS=()
    failed=0
    for task_name in "${TASKS[@]}"; do
        run_task_worker "${label}" "${step}" "${task_name}" &
        WORKER_PIDS+=("$!")
        if (( ${#WORKER_PIDS[@]} == WORKERS )); then
            for pid in "${WORKER_PIDS[@]}"; do
                wait "${pid}" || failed=1
            done
            WORKER_PIDS=()
            (( failed == 0 )) || return 1
        fi
    done
    for pid in "${WORKER_PIDS[@]}"; do
        wait "${pid}" || failed=1
    done
    WORKER_PIDS=()
    (( failed == 0 ))
}

run_checkpoint() {
    local label="$1"
    local checkpoint="$2"
    local step="$3"
    local server_script="$4"
    local output="${OUTPUT_ROOT}/${label}.json"
    if [[ -e "${output}" ]]; then
        echo "[LOCAL] Skipping completed result: ${output}"
        return 0
    fi

    echo "[LOCAL] Starting ${label}: ${#TASKS[@]} tasks x ${EPISODES} episodes, ${WORKERS} workers"
    "${POINTACT_PYTHON}" -u "${server_script}" \
        --pretrained-path "${checkpoint}" \
        --seed "${SEED}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --num-denoise-steps 10 \
        >"${OUTPUT_ROOT}/logs/${label}_server.log" 2>&1 &
    SERVER_PID=$!
    wait_for_server
    run_worker_batch "${label}" "${step}"

    "${RLBENCH_PYTHON}" experiments/10_rlbench/merge_filled9_parallel_results.py \
        --worker-dir "${OUTPUT_ROOT}/${label}.workers" \
        --output "${output}" \
        --checkpoint-step "${step}" \
        --episodes "${EPISODES}" \
        --seed "${SEED}" \
        --corruption-seed "${CORRUPTION_SEED}" \
        --eval-episode-offset "${EVAL_EPISODE_OFFSET}"

    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
}

run_checkpoint classification_step10000 "${CLS_CHECKPOINT}" 10000 \
    experiments/10_rlbench/run_filled9_server.py
run_checkpoint reconstruction_step17728 "${RECON_CHECKPOINT}" 17728 \
    experiments/10_rlbench/run_filled9_reconstruction_action_only_server.py

echo "[LOCAL] Completed both checkpoints: ${OUTPUT_ROOT}"
