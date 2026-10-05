#!/usr/bin/env bash
set -euo pipefail
umask 077

hostname
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo 'numeric SLURM_JOB_ID required' >&2; exit 2; }
test "$(hostname -s)" = bremen
code=/mnt/home/weihangli/pointact_project/code/pointact_cga_pretrain_20261004
image=/mnt/home/weihangli/pointact_project/containers/pointact.sqsh
config="$code/configs/polar_normal/cga_dinov3_mixed_10tasks_ddp2_bs192_wandb_remote.yaml"
output=/nfs/aachen/weihangli/checkpoints/cga_dinov3_mixed_10tasks_v2_ddp2_bs192_wandb_20261004
netrc="/tmp/yuan/credentials/job-${SLURM_JOB_ID}/netrc"
wandb_root="/tmp/yuan/wandb/job-${SLURM_JOB_ID}/cga-dinov3-bs192"

test -s "$image"
test -s "$config"
test -d "$output" && test -w "$output"
test ! -e "$output/metrics.jsonl"
test ! -e "$output/last.pt"
test -s "$netrc"
test "$(stat -c %a "$netrc")" = 600
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 2
mkdir -p "$wandb_root"/{data,cache,config}
chmod 700 "$wandb_root" "$wandb_root"/{data,cache,config}

job_root="/tmp/yuan/pointact_enroot/job-${SLURM_JOB_ID}-cga-ddp2-wandb"
mkdir -p "$job_root"/{data,cache,runtime,tmp}
chmod 700 "$job_root"/{runtime,tmp}
export ENROOT_DATA_PATH="$job_root/data"
export ENROOT_CACHE_PATH="$job_root/cache"
export ENROOT_RUNTIME_PATH="$job_root/runtime"
export ENROOT_TEMP_PATH="$job_root/tmp"

exec enroot start --root --rw \
    --env PYTHONPATH="$code" \
    --env POLAR_CONFIG="$config" \
    --env NETRC="$netrc" \
    --env WANDB_ENTITY=fengy7732-technical-university-of-munich \
    --env WANDB_PROJECT=pointact-cga-dinov3-polar-normal \
    --env WANDB_MODE=online \
    --env WANDB_DIR="$wandb_root" \
    --env WANDB_DATA_DIR="$wandb_root/data" \
    --env WANDB_CACHE_DIR="$wandb_root/cache" \
    --env WANDB_CONFIG_DIR="$wandb_root/config" \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env NCCL_P2P_DISABLE=0 --env NCCL_IB_DISABLE=0 \
    --mount /mnt:/mnt --mount /tmp:/tmp --mount /nfs:/nfs \
    "$image" bash -c '
        set -e
        export PATH=/opt/conda/envs/pointact/bin:/opt/conda/bin:$PATH
        export CONDA_PREFIX=/opt/conda/envs/pointact
        export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        cd /mnt/home/weihangli/pointact_project/code/pointact_cga_pretrain_20261004
        exec torchrun --standalone --nproc_per_node=2 \
            scripts/train_polar_normal_ddp.py --config "$POLAR_CONFIG"
    '
