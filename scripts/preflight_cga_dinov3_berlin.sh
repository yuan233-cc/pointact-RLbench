#!/usr/bin/env bash
set -euo pipefail

hostname
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
job_root="/tmp/yuan/pointact_enroot/job-${SLURM_JOB_ID}-cga-preflight"
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
test -s "$code/configs/polar_normal/cga_dinov3_mixed_10tasks_berlin_offline.yaml"

enroot start --root --rw \
  --env PYTHONPATH="$code" \
  --mount /mnt:/mnt --mount /tmp:/tmp --mount /nfs:/nfs \
  "$image" bash -c '
    set -e
    export PATH=/opt/conda/envs/pointact/bin:/opt/conda/bin:$PATH
    python - <<"PY"
import runpy
from pathlib import Path

import torch
import yaml

code = Path("/mnt/home/weihangli/pointact_project/code/pointact_cga_pretrain_20261004")
config = yaml.safe_load((code / "configs/polar_normal/cga_dinov3_mixed_10tasks_berlin_offline.yaml").read_text())
entry = runpy.run_path(str(code / "scripts/train_polar_normal.py"))
assert torch.cuda.is_available()
assert Path(config["model"]["dinov3_weights"]).is_file()
assert Path(config["training"]["output_dir"]).parent.is_dir()
samples = {}
for split in ("train", "val"):
    for name, dataset in entry["build_datasets"](config, split):
        sample = dataset[0]
        for key, shape in (("polar_observation", (11, 256, 256)), ("physical_prior", (11, 256, 256)), ("rgb", (3, 256, 256)), ("normal_gt", (3, 256, 256))):
            assert tuple(sample[key].shape) == shape, (split, name, key, sample[key].shape)
            assert torch.isfinite(sample[key]).all(), (split, name, key)
        assert sample["normal_valid_mask"].any(), (split, name)
        samples[(split, name)] = sample
        print(split, name, len(dataset), "valid", int(sample["normal_valid_mask"].sum()), flush=True)

model = entry["build_model"](config).cuda().train()
batch = entry["collate_training_fields"]([samples[("train", "hammer")]], use_rgb=True)
prediction = model(
    polar_observation=batch["polar_observation"].cuda(),
    physical_prior=batch["physical_prior"].cuda(),
    rgb=batch["rgb"].cuda(),
)["normal"]
loss = entry["masked_cosine_normal_loss"](
    prediction, batch["normal_gt"].cuda(), batch["normal_valid_mask"].cuda()
)
assert torch.isfinite(loss), loss
loss.backward()
print("model_forward_backward_ok", float(loss), "gpu_mem_bytes", torch.cuda.max_memory_allocated(), flush=True)
PY
  '
