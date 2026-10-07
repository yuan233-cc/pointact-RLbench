#!/usr/bin/env bash
# GPU host wrapper: own the foreground Enroot process and release on completion.
set -euo pipefail
hostname
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
storage=/local
storage_mount=(--mount /local:/local)
if [[ "$(hostname -s)" != aachen ]]; then
  storage=/nfs/aachen
  storage_mount=(--mount /nfs:/nfs)
fi
code="$1"
data="$2"
backbone="$3"
teacher="$4"
output="$5"
teacher="${teacher/#\/local\//$storage/}"
output="${output/#\/local\//$storage/}"
shift 5
job_root="/tmp/yuan/geometry-job-${SLURM_JOB_ID}"
mkdir -p "$job_root"/{data,cache,runtime,tmp,wandb}
chmod 700 "$job_root/runtime" "$job_root/tmp"
export ENROOT_DATA_PATH="$job_root/data"
export ENROOT_CACHE_PATH="$job_root/cache"
export ENROOT_RUNTIME_PATH="$job_root/runtime"
export ENROOT_TEMP_PATH="$job_root/tmp"
credential="/tmp/yuan/credentials/job-${SLURM_JOB_ID}/netrc"
test -s "$credential"
test ! -e "$output"
set +e
enroot start --root --rw --mount /mnt:/mnt "${storage_mount[@]}" --mount /tmp:/tmp \
  --env PYTHONPATH="$code" --env CODE_ROOT="$code" --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  --env NETRC="$credential" --env WANDB_DIR="$job_root/wandb" \
  --env WANDB_DATA_DIR="$job_root/wandb/data" --env WANDB_CACHE_DIR="$job_root/wandb/cache" \
  --env WANDB_CONFIG_DIR="$job_root/wandb/config" \
  /mnt/home/weihangli/pointact_project/containers/pointact.sqsh \
  bash -c 'export PATH=/opt/conda/envs/pointact/bin:/opt/conda/bin:$PATH; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; cd "$CODE_ROOT"; exec python scripts/train_workspace_geometry.py "$@"' bash \
  --backbone "$backbone" --dataset-root "$data" --teacher-checkpoint "$teacher" \
  --cga-records "$storage/weihangli/datasets/RLBenchPolarNormal10TasksV2_CGAOffline_20261004/records" \
  --concerto-checkpoint /mnt/home/weihangli/pointact_project/code/pvla_tasknet_native_48401a6/pretrained/Pointcept-Concerto/concerto_large.pth \
  --dino-weights "$storage/weihangli/checkpoints/pretrained/dinov3_convnext_base_lvd1689m/dinov3_convnext_base_pretrain_lvd1689m.pth" \
  --output-dir "$output" "$@"
status=$?
set -e
# Probes are managed by the caller; final runs release even after an error.
if [[ "${GEOMETRY_RELEASE_JOB:-0}" == 1 ]]; then
  scancel "$SLURM_JOB_ID"
fi
exit "$status"
