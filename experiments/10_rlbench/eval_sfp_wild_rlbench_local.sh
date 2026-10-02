#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace_root="$(dirname "$project_root")"
pointact_python="${POINTACT_PYTHON:-$workspace_root/.conda/envs/pointact/bin/python}"
rlbench_python="${RLBENCH_PYTHON:-$workspace_root/.conda/envs/rlbench/bin/python}"
checkpoint="${1:-${CHECKPOINT:-}}"
if [[ -z "$checkpoint" ]]; then
    echo "Usage: $0 CHECKPOINT_DIR [CHECKPOINT_STEP]" >&2
    exit 2
fi
checkpoint_step="${2:-${CHECKPOINT_STEP:--1}}"
episodes="${EPISODES:-1}"
max_steps="${MAX_STEPS:-25}"
seed="${SEED:-7}"
corruption_seed="${CORRUPTION_SEED:-20260923}"
port="${PORT:-15570}"
run_tag="${RUN_TAG:-$(basename "$checkpoint")_seed${seed}}"
output_root="${OUTPUT_ROOT:-$project_root/outputs/sfp_wild_${run_tag}}"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$workspace_root/.deps/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${project_root}:${workspace_root}/rlbench_custom_render/RLBench:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pointact-matplotlib-sfp-wild}"

cd "$project_root"
for path in "$pointact_python" "$rlbench_python" "$checkpoint/config.json"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
if ss -ltnH "sport = :${port}" | grep -q .; then
    echo "Port ${port} is already occupied" >&2
    exit 2
fi

mkdir -p "$output_root/logs"
cp experiments/10_rlbench/filled9_inference.py "$output_root/filled9_inference_snapshot.py"
cp experiments/10_rlbench/run_sfp_wild_rlbench.py "$output_root/rlbench_client_snapshot.py"
cp experiments/10_rlbench/run_sfp_wild_server.py "$output_root/server_snapshot.py"
cp pointact/robot_envs/rlbench_utils/sfp_adapter.py "$output_root/sfp_adapter_snapshot.py"
cp "$0" "$output_root/launch.sh"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$pointact_python" -u experiments/10_rlbench/run_sfp_wild_server.py \
    --pretrained-path "$checkpoint" \
    --seed "$seed" \
    --host 127.0.0.1 \
    --port "$port" \
    >"$output_root/logs/server.log" 2>&1 &
server_pid=$!

for _ in $(seq 1 240); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid"
        exit 1
    fi
    if ss -ltnH "sport = :${port}" | grep -q .; then
        break
    fi
    sleep 5
done
if ! ss -ltnH "sport = :${port}" | grep -q .; then
    echo "Policy server did not become ready within 20 minutes" >&2
    exit 1
fi

output="$output_root/result.json"
resume_args=()
if [[ -e "$output_root/result.progress.jsonl" ]]; then
    resume_args+=(--resume)
fi
"$rlbench_python" -u experiments/10_rlbench/run_sfp_wild_rlbench.py \
    --episodes "$episodes" \
    --max-steps "$max_steps" \
    --seed "$seed" \
    --corruption-seed "$corruption_seed" \
    --checkpoint-step "$checkpoint_step" \
    --host 127.0.0.1 \
    --port "$port" \
    --output "$output" \
    "${resume_args[@]}" \
    2>&1 | tee "$output_root/logs/evaluation.log"
