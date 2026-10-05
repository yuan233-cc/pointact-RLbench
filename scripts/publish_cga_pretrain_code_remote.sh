#!/usr/bin/env bash
set -euo pipefail

mode="${1:?expected init or publish}"
hostname
test "$(hostname -s)" = aachen
test "${SLURM_JOB_ID:-}" = 26195
root=/mnt/home/weihangli/pointact_project/code
final="$root/pointact_cga_pretrain_20261004"
partial="$final.partial.26195"
test "$(stat -c %U "$root")" = weihangli
test -w "$root"
test "$(realpath -m "$final")" = "$final"
test "$(realpath -m "$partial")" = "$partial"

case "$mode" in
  init)
    test ! -e "$final"
    test ! -e "$partial"
    mkdir -- "$partial"
    ;;
  publish)
    test ! -e "$final"
    test -d "$partial"
    test -s "$partial/scripts/train_polar_normal.py"
    test -s "$partial/configs/polar_normal/cga_dinov3_mixed_10tasks_aachen_offline.yaml"
    test -s "$partial/pointact/model/vla_pointact/action_head_3d/cga_dino_normal.py"
    test -s "$partial/pointact/third_party/dinov3/__init__.py"
    test -s "$partial/pointact/data/polar_normal_dataset.py"
    mv -T -- "$partial" "$final"
    printf 'published %s\n' "$final"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
