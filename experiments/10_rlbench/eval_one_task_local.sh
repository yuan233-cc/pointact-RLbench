#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng/robot-PointAct"
WORKSPACE_ROOT="/media/hyunjun/NewDisk1/Yuan_Feng"
CHECKPOINT="${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000"
OUTPUT_DIR="${CHECKPOINT}/preds/seed7_close_box_10eps_with_final"
POINTACT_PYTHON="${WORKSPACE_ROOT}/.conda/envs/pointact/bin/python"
RLBENCH_PYTHON="${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python"
HOST="127.0.0.1"
PORT="15477"

export COPPELIASIM_ROOT="${WORKSPACE_ROOT}/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"${POINTACT_PYTHON}" -u scripts/run_server.py \
    --args.seed 7 \
    --args.pretrained_path "${CHECKPOINT}" \
    --args.num_denoise_steps 10 \
    --args.host "${HOST}" \
    --args.port "${PORT}" &
SERVER_PID=$!

SERVER_READY=0
for _ in $(seq 1 480); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        wait "${SERVER_PID}"
        exit 1
    fi
    if ss -ltnH "sport = :${PORT}" | rg -q .; then
        SERVER_READY=1
        break
    fi
    sleep 5
done

if [[ "${SERVER_READY}" != "1" ]]; then
    echo "PointACT server did not become ready within 40 minutes." >&2
    exit 1
fi

"${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_rlbench_client.py \
    --args.taskvar close_box+0 \
    --args.seed 7 \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.replan_steps 1 \
    --args.max_steps 25 \
    --args.pretrained_path "${CHECKPOINT}" \
    --args.clip_within_workspace \
    --args.pred_rot_type euler \
    --args.save_dir "${OUTPUT_DIR}" \
    --args.num_workers 1 \
    --args.num_episodes 10 \
    --args.save_video
