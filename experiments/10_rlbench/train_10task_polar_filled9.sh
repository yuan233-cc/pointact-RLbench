#!/usr/bin/env bash
# Run PointACT with the appended filled XYZRGB+DoLP+cos2AoLP+sin2AoLP cloud.
# No material-conditioning data fields or model flag are enabled.
set -euo pipefail

export DATA_PATH=experiments/10_rlbench/data_configs/data-10task-polar-filled9.yaml
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pointact_10task_polar_filled9}"
exec bash experiments/10_rlbench/train_phone_polar_incomplete9_one_episode.sh
