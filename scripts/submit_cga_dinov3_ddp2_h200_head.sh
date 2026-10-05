#!/usr/bin/env bash
set -euo pipefail

hostname
test "$(hostname -s)" = head
if squeue -h -u weihangli -n cga-dinov3-ddp2-ssh | grep -q .; then
    echo 'A CGA+DINOv3 two-GPU allocation already exists; inspect it first.' >&2
    exit 1
fi

sbatch --parsable \
    --partition=h200 \
    --qos=phds_opportunistic \
    --gres=gpu:2 \
    --nodes=1 \
    --ntasks=1 \
    --time=23:59:00 \
    --job-name=cga-dinov3-ddp2-ssh \
    --output=/mnt/home/weihangli/cga-dinov3-ddp2-%j.out \
    --error=/mnt/home/weihangli/cga-dinov3-ddp2-%j.err \
    /mnt/general/examples/ssh.sh
