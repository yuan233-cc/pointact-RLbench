#!/usr/bin/env bash
set -euo pipefail
hostname
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
job_root="/tmp/yuan/geometry-job-${SLURM_JOB_ID}"
mkdir -p "$job_root"/{data,cache,runtime,tmp}
chmod 700 "$job_root/runtime" "$job_root/tmp"
export ENROOT_DATA_PATH="$job_root/data" ENROOT_CACHE_PATH="$job_root/cache"
export ENROOT_RUNTIME_PATH="$job_root/runtime" ENROOT_TEMP_PATH="$job_root/tmp"
exec enroot start --root --rw --mount /mnt:/mnt --mount /local:/local --mount /tmp:/tmp \
  --env NETRC="/tmp/yuan/credentials/job-${SLURM_JOB_ID}/netrc" \
  /mnt/home/weihangli/pointact_project/containers/pointact.sqsh \
  /opt/conda/envs/pointact/bin/python "$1/scripts/login_wandb_job.py"
