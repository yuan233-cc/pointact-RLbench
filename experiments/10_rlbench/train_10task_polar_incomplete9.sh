#!/usr/bin/env bash
# Run from the robot-PointAct repository root after exporting the ten-task dataset.
set -euo pipefail

export DATA_PATH=experiments/10_rlbench/data_configs/data-10task-polar-incomplete9.yaml
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pointact_10task_polar_incomplete9}"
exec bash experiments/10_rlbench/train_phone_polar_incomplete9_one_episode.sh
