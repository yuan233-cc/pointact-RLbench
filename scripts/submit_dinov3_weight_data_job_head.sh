#!/usr/bin/env bash
set -euo pipefail

hostname
test "$(hostname -s)" = head
if squeue -h -u weihangli -n dinov3-weight-data-ssh | grep -q .; then
  echo 'matching data job already exists' >&2
  exit 1
fi

sbatch --parsable \
  --partition=data \
  --qos=phds_normal \
  --nodes=1 \
  --ntasks=1 \
  --time=04:00:00 \
  --job-name=dinov3-weight-data-ssh \
  --output=/mnt/home/weihangli/dinov3-weight-data-%j.out \
  --error=/mnt/home/weihangli/dinov3-weight-data-%j.err \
  --wrap='echo "JOB_ID:${SLURM_JOB_ID}"; PORT=$(python -c "import random; print(random.randint(20000,30000))"); start-ssh-server "$PORT"; sleep 4h'
