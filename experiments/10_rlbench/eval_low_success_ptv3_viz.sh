#!/usr/bin/env bash
set -euo pipefail

# This script adds a feature-capturing server around the unchanged PointACT
# model and reuses the unchanged RLBench client for one short episode per task.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/rlbench/pointact-rlbench-job25521/checkpoint-40000}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/PTV3_feature_visualizations_low_success_${RUN_TAG}}"
POINTACT_PYTHON="${POINTACT_PYTHON:-${WORKSPACE_ROOT}/.conda/envs/pointact/bin/python}"
RLBENCH_PYTHON="${RLBENCH_PYTHON:-${WORKSPACE_ROOT}/.conda/envs/rlbench/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-15487}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-25}"
CAPTURE_EVERY="${CAPTURE_EVERY:-3}"
MAX_CAPTURES="${MAX_CAPTURES:-0}"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-${WORKSPACE_ROOT}/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pointact_matplotlib_${USER:-user}}"

TASKS=(
    "sweep_to_dustpan+0"
    "take_frame_off_hanger+0"
    "water_plants+0"
)

mkdir -p "${OUTPUT_DIR}/captures" "${OUTPUT_DIR}/eval" "${MPLCONFIGDIR}"
cd "${PROJECT_ROOT}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"${POINTACT_PYTHON}" -u experiments/10_rlbench/run_ptv3_feature_server.py \
    --args.seed 7 \
    --args.pretrained-path "${CHECKPOINT}" \
    --args.num-denoise-steps 10 \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.capture-dir "${OUTPUT_DIR}/captures" \
    --args.capture-every "${CAPTURE_EVERY}" \
    --args.max-captures "${MAX_CAPTURES}" &
SERVER_PID=$!

server_ready=0
for _ in $(seq 1 480); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        wait "${SERVER_PID}"
        exit 1
    fi
    if ss -ltnH "sport = :${PORT}" | rg -q .; then
        server_ready=1
        break
    fi
    sleep 5
done
if [[ "${server_ready}" != "1" ]]; then
    echo "PointACT feature server did not become ready within 40 minutes." >&2
    exit 1
fi

for taskvar in "${TASKS[@]}"; do
    task_slug="${taskvar/+/_variation_}"
    eval_dir="${OUTPUT_DIR}/eval/${task_slug}"
    mkdir -p "${eval_dir}"

    "${RLBENCH_PYTHON}" -u experiments/10_rlbench/run_rlbench_client.py \
        --args.taskvar "${taskvar}" \
        --args.seed 7 \
        --args.host "${HOST}" \
        --args.port "${PORT}" \
        --args.replan-steps 1 \
        --args.max-steps "${MAX_STEPS}" \
        --args.pretrained-path "${CHECKPOINT}" \
        --args.clip-within-workspace \
        --args.pred-rot-type euler \
        --args.save-dir "${eval_dir}" \
        --args.num-workers 1 \
        --args.num-episodes "${NUM_EPISODES}" \
        --args.save-video

done

cleanup
SERVER_PID=""

"${POINTACT_PYTHON}" experiments/10_rlbench/render_ptv3_features.py \
    --args.inputs "${OUTPUT_DIR}/captures" \
    --args.output-dir "${OUTPUT_DIR}/rendered" \
    --args.make-video

"${POINTACT_PYTHON}" experiments/10_rlbench/render_ptv3_geometry_analysis.py \
    --args.inputs "${OUTPUT_DIR}/captures" \
    --args.output-dir "${OUTPUT_DIR}/geometry_analysis" \
    --args.top-k 5

echo "PTV3 visualizations written to ${OUTPUT_DIR}/rendered"
echo "PTV3 geometry analysis written to ${OUTPUT_DIR}/geometry_analysis"
