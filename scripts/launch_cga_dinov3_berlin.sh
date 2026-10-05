#!/usr/bin/env bash
set -euo pipefail
umask 077

hostname
[[ "${SLURM_JOB_ID:-}" == 26204 ]] || { echo 'Expected allocation 26204' >&2; exit 2; }
image=/mnt/home/weihangli/pointact_project/containers/pointact.sqsh
code=/mnt/home/weihangli/pointact_project/code/pointact_cga_pretrain_20261004
config="$code/configs/polar_normal/cga_dinov3_mixed_10tasks_berlin_offline.yaml"
weight=/nfs/aachen/weihangli/checkpoints/pretrained/dinov3_convnext_base_lvd1689m/dinov3_convnext_base_pretrain_lvd1689m.pth
output=/nfs/aachen/weihangli/checkpoints/cga_dinov3_mixed_10tasks_v2_offline

test -s "$image"
test -s "$config"
test -s "$weight"
test -d "$output" && test -w "$output"
test ! -e "$output/last.pt"
test ! -e "$output/metrics.jsonl"

job_root=/tmp/yuan/pointact_enroot/job-26204-cga-train
mkdir -p "$job_root/data" "$job_root/cache" "$job_root/runtime" "$job_root/tmp"
chmod 700 "$job_root/runtime" "$job_root/tmp"
export ENROOT_DATA_PATH="$job_root/data"
export ENROOT_CACHE_PATH="$job_root/cache"
export ENROOT_RUNTIME_PATH="$job_root/runtime"
export ENROOT_TEMP_PATH="$job_root/tmp"

exec enroot start --root --rw \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env PYTHONPATH="$code" \
    --mount /mnt:/mnt --mount /tmp:/tmp --mount /nfs:/nfs \
    "$image" bash -c '
        set -e
        export PATH=/opt/conda/envs/pointact/bin:/opt/conda/bin:$PATH
        export CONDA_PREFIX=/opt/conda/envs/pointact
        export PYTHONNOUSERSITE=1
        export PYTHONUNBUFFERED=1
        cd /mnt/home/weihangli/pointact_project/code/pointact_cga_pretrain_20261004
        exec python scripts/train_polar_normal.py --config configs/polar_normal/cga_dinov3_mixed_10tasks_berlin_offline.yaml
    '
