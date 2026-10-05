#!/usr/bin/env bash
set -euo pipefail

hostname
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
job_root="/tmp/yuan/pointact_enroot/job-${SLURM_JOB_ID}"
mkdir -p "$job_root/data" "$job_root/cache" "$job_root/runtime" "$job_root/tmp"
chmod 700 "$job_root/runtime" "$job_root/tmp"
export ENROOT_DATA_PATH="$job_root/data"
export ENROOT_CACHE_PATH="$job_root/cache"
export ENROOT_RUNTIME_PATH="$job_root/runtime"
export ENROOT_TEMP_PATH="$job_root/tmp"

image=/mnt/home/weihangli/pointact_project/containers/pointact.sqsh
code=/mnt/home/weihangli/pointact_project/code/pointact_cga_pretrain_20261004
test -s "$image"
test -s "$code/scripts/train_polar_normal.py"
enroot start --root --rw \
  --env PYTHONPATH="$code" \
  --mount /mnt:/mnt --mount /tmp:/tmp --mount /local:/local \
  "$image" bash -c '
    set -e
    export PATH=/opt/conda/envs/pointact/bin:/opt/conda/bin:$PATH
    command -v python
    python -c "import torch, numpy, scipy, av, lmdb, yaml; print(torch.__version__, numpy.__version__, scipy.__version__, av.__version__, lmdb.__version__)"
    test -d /local/weihangli/datasets/PolarNormalCGA_20261003_v1/hammer
    test -d /local/weihangli/datasets/PolarNormalCGA_20261003_v1/sfpuel
    test -d /local/weihangli/datasets/RLBenchPolarNormal10TasksV2_CGAOffline_20261004
    echo dataset_mounts_ok
  '

enroot start --root --rw \
  --env PYTHONPATH="$code" \
  --mount /mnt:/mnt --mount /tmp:/tmp --mount /local:/local \
  "$image" /opt/conda/envs/pointact/bin/python -c '
import runpy
import torch
import yaml

code = "/mnt/home/weihangli/pointact_project/code/pointact_cga_pretrain_20261004"
config = yaml.safe_load(open(code + "/configs/polar_normal/cga_dinov3_mixed_10tasks_aachen_offline.yaml"))
entry = runpy.run_path(code + "/scripts/train_polar_normal.py")
for split in ("train", "val"):
    for name, dataset in entry["build_datasets"](config, split):
        sample = dataset[0]
        assert sample["polar_observation"].shape == (11, 256, 256)
        assert sample["physical_prior"].shape == (11, 256, 256)
        assert sample["rgb"].shape == (3, 256, 256)
        assert sample["normal_gt"].shape == (3, 256, 256)
        assert torch.isfinite(sample["polar_observation"]).all()
        assert torch.isfinite(sample["physical_prior"]).all()
        assert sample["normal_valid_mask"].any()
        print(split, name, len(dataset), "valid", int(sample["normal_valid_mask"].sum()), flush=True)
  '
