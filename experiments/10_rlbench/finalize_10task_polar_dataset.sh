#!/usr/bin/env bash
# Resume collection, export the training dataset, package it, and upload to yuan1119.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace_root="$(dirname "$project_root")"
raw_root="$workspace_root/rlbench_custom_render/RLBench/output/ten_tasks_polar_train_20260921"
data_parent="$project_root/robot_data/rlbench/lerobot_point_lmdb"
dataset_name=hybridvla_10tasks_train_keysteps_polar_incomplete9_v1
dataset_root="$data_parent/$dataset_name"
package_root="$data_parent/${dataset_name}_upload"
archive="$package_root/${dataset_name}.zip"
repo_id=yuan1119/rlbench-10tasks-polar-incomplete9
pointact_python=/home/hyunjun/anaconda3/envs/pointact/bin/python

cd "$project_root"
"$pointact_python" -u experiments/10_rlbench/collect_10task_polar_episodes.py \
    --output "$raw_root" --episodes-per-task 100 --render-spp 512

if [[ ! -d "$dataset_root" ]]; then
    "$pointact_python" -u experiments/10_rlbench/export_10task_polar_incomplete9.py \
        --raw "$raw_root" --output "$dataset_root" --episodes-per-task 100
fi

"$pointact_python" - "$dataset_root" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
meta = json.loads((root / "meta/polar_incomplete_features.json").read_text())
assert meta["complete"] and meta["total_episodes"] == 1000 and len(meta["tasks"]) == 10
assert (root / "points_frontview_polar_incomplete9/data.mdb").is_file()
assert (root / "polar_frontview_dense/data.mdb").is_file()
print(f"Validated {meta['total_episodes']} episodes and {meta['total_frames']} keyframes")
PY

mkdir -p "$package_root"
if [[ ! -f "$archive" ]]; then
    archive_tmp="$package_root/${dataset_name}.part.zip"
    if [[ -e "$archive_tmp" ]]; then
        echo "Incomplete archive needs inspection: $archive_tmp" >&2
        exit 1
    fi
    (cd "$data_parent" && zip -q -r -1 "$archive_tmp" "$dataset_name")
    mv "$archive_tmp" "$archive"
fi
unzip -tq "$archive" >/dev/null
(cd "$package_root" && sha256sum -b "$(basename "$archive")" > SHA256SUMS)
cp "$dataset_root/README.md" "$package_root/README.md"

"$pointact_python" - "$repo_id" <<'PY'
import sys
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
api = HfApi()
account = api.whoami()["name"]
if account != "yuan1119":
    raise SystemExit(f"Hugging Face account is {account!r}; log in as 'yuan1119' before uploading")
repo_id = sys.argv[1]
try:
    info = api.repo_info(repo_id, repo_type="dataset")
except RepositoryNotFoundError:
    api.create_repo(repo_id, repo_type="dataset", private=True)
else:
    if not info.private:
        raise SystemExit(f"Refusing to upload dataset to an existing public repo: {repo_id}")
print(f"Uploading to private dataset repo {repo_id}")
PY

for attempt in 1 2 3; do
    if HF_XET_HIGH_PERFORMANCE=1 hf upload "$repo_id" "$package_root" . \
        --repo-type dataset --private \
        --commit-message "Add ten-task aligned RLBench polar dataset"; then
        break
    fi
    if [[ "$attempt" -eq 3 ]]; then
        echo "Hugging Face upload failed after three attempts" >&2
        exit 1
    fi
    sleep 120
done

"$pointact_python" - "$repo_id" "$dataset_name" <<'PY'
import sys
from huggingface_hub import HfApi
repo_id, name = sys.argv[1:]
files = set(HfApi().list_repo_files(repo_id, repo_type="dataset"))
required = {f"{name}.zip", "SHA256SUMS", "README.md"}
missing = required - files
if missing:
    raise SystemExit(f"Hugging Face upload incomplete: {sorted(missing)}")
print(f"Verified remote files in {repo_id}")
PY
