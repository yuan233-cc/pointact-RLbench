#!/usr/bin/env bash
set -euo pipefail

hostname
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 2
code=/mnt/home/weihangli/pointact_project/code/pointact_cga_pretrain_20261004
image=/mnt/home/weihangli/pointact_project/containers/pointact.sqsh
if [[ "$(hostname -s)" == aachen ]]; then
    storage=/local/weihangli
    mount=/local:/local
    config="$code/configs/polar_normal/cga_dinov3_mixed_10tasks_ddp2_aachen_offline.yaml"
else
    storage=/nfs/aachen/weihangli
    mount=/nfs:/nfs
    config="$code/configs/polar_normal/cga_dinov3_mixed_10tasks_ddp2_remote_offline.yaml"
fi
test -s "$image"
test -s "$config"
test -s "$storage/checkpoints/pretrained/dinov3_convnext_base_lvd1689m/dinov3_convnext_base_pretrain_lvd1689m.pth"
test -w "$storage/checkpoints"
test ! -e "$storage/checkpoints/cga_dinov3_mixed_10tasks_v2_offline_ddp2_20261004"

job_root="/tmp/yuan/pointact_enroot/job-${SLURM_JOB_ID}-cga-ddp2-preflight"
mkdir -p "$job_root/data" "$job_root/cache" "$job_root/runtime" "$job_root/tmp"
chmod 700 "$job_root/runtime" "$job_root/tmp"
export ENROOT_DATA_PATH="$job_root/data"
export ENROOT_CACHE_PATH="$job_root/cache"
export ENROOT_RUNTIME_PATH="$job_root/runtime"
export ENROOT_TEMP_PATH="$job_root/tmp"

timeout --foreground --signal=TERM --kill-after=10s 360s \
    enroot start --root --rw \
    --env PYTHONPATH="$code" \
    --env POLAR_CONFIG="$config" \
    --mount /mnt:/mnt --mount /tmp:/tmp --mount "$mount" \
    "$image" bash -c '
        set -e
        export PATH=/opt/conda/envs/pointact/bin:/opt/conda/bin:$PATH
        export CONDA_PREFIX=/opt/conda/envs/pointact
        export PYTHONNOUSERSITE=1
        export PYTHONUNBUFFERED=1
        cd /mnt/home/weihangli/pointact_project/code/pointact_cga_pretrain_20261004
        python -c "import pointact, torch; print(\"pointact\", list(pointact.__path__), \"gpus\", torch.cuda.device_count(), flush=True)"
        torchrun --standalone --nproc_per_node=2 scripts/check_two_gpu_nccl.py
        torchrun --standalone --nproc_per_node=2 scripts/train_polar_normal_ddp.py \
            --config "$POLAR_CONFIG" --smoke-optimizer-steps 2
    '
