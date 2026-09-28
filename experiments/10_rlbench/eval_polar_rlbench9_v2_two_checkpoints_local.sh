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
RUN_TAG="${RUN_TAG:-25ep_seed7_20260924}"
PORT="${PORT:-15560}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/polar9_v2_two_checkpoint_${RUN_TAG}"

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
export MPLCONFIGDIR="/tmp/pointact-matplotlib-polar9-v2"

cd "${PROJECT_ROOT}"
for path in "${POINTACT_PYTHON}" "${RLBENCH_PYTHON}" \
            "${CLS_CHECKPOINT}/model.safetensors" \
            "${RECON_CHECKPOINT}/model.safetensors"; do
    [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done
if ss -ltnH "sport = :${PORT}" | grep -q .; then
    echo "Port ${PORT} is already occupied" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}/logs"
cp experiments/10_rlbench/filled9_inference.py "${OUTPUT_ROOT}/filled9_inference_snapshot.py"
cp experiments/10_rlbench/run_filled9_rlbench.py "${OUTPUT_ROOT}/rlbench_client_snapshot.py"
cp experiments/10_rlbench/run_filled9_server.py "${OUTPUT_ROOT}/classification_server_snapshot.py"
cp experiments/10_rlbench/run_filled9_reconstruction_action_only_server.py \
   "${OUTPUT_ROOT}/reconstruction_server_snapshot.py"
cp "${BASH_SOURCE[0]}" "${OUTPUT_ROOT}/launch.sh"

SERVER_PID=""
cleanup() {
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

run_checkpoint() {
    local label="$1"
    local checkpoint="$2"
    local step="$3"
    local server_script="$4"
    local output="${OUTPUT_ROOT}/${label}.json"
    local progress="${OUTPUT_ROOT}/${label}.progress.jsonl"
    local resume_args=()
    if [[ -e "${output}" ]]; then
        echo "[LOCAL] Skipping completed result: ${output}"
        return 0
    fi
    if [[ -e "${progress}" ]]; then
        resume_args+=(--resume)
    fi

    echo "[LOCAL] Starting ${label}: 10 tasks x ${EPISODES} episodes"
    "${POINTACT_PYTHON}" -u "${server_script}" \
        --pretrained-path "${checkpoint}" \
        --seed "${SEED}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --num-denoise-steps 10 \
        >"${OUTPUT_ROOT}/logs/${label}_server.log" 2>&1 &
    SERVER_PID=$!
    wait_for_server

    "${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_filled9_rlbench.py \
        --episodes "${EPISODES}" \
        --max-steps "${MAX_STEPS}" \
        --seed "${SEED}" \
        --corruption-seed "${CORRUPTION_SEED}" \
        --eval-episode-offset 10000 \
        --checkpoint-step "${step}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --output "${output}" \
        "${resume_args[@]}" \
        2>&1 | tee "${OUTPUT_ROOT}/logs/${label}_evaluation.log"

    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
}

run_checkpoint \
    classification_step10000 \
    "${CLS_CHECKPOINT}" \
    10000 \
    experiments/10_rlbench/run_filled9_server.py

run_checkpoint \
    reconstruction_step17728 \
    "${RECON_CHECKPOINT}" \
    17728 \
    experiments/10_rlbench/run_filled9_reconstruction_action_only_server.py

echo "[LOCAL] Completed both checkpoints: ${OUTPUT_ROOT}"
