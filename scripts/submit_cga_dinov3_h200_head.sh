#!/usr/bin/env bash
set -euo pipefail

hostname
test "$(hostname -s)" = head
if squeue -h -u weihangli -n cga-dinov3-pretrain-ssh | grep -q .; then
    echo 'A CGA+DINOv3 H200 SSH allocation already exists; inspect it first.' >&2
    exit 1
fi

sbatch --parsable \
    --partition=h200 \
    --qos=phds_normal \
    --nodelist=berlin \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks=1 \
    --time=23:59:00 \
    --job-name=cga-dinov3-pretrain-ssh \
    --output=/mnt/home/weihangli/cga-dinov3-pretrain-%j.out \
    --error=/mnt/home/weihangli/cga-dinov3-pretrain-%j.err \
    /mnt/general/examples/ssh.sh
