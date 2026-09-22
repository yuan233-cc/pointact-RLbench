#!/usr/bin/env bash
# Ten-task filled polar PointACT classifier training.
# Keep the filled9 dataset and nine-channel point stem; material conditioning is off.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"

gpus="${GPUS:-1}"
if ! [[ "$gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS must be a positive integer: $gpus" >&2
    exit 2
fi
accelerate_args=(--num_processes "$gpus" --num_machines 1)
if (( gpus > 1 )); then
    accelerate_args=(--multi_gpu "${accelerate_args[@]}")
fi

data_path="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-polar-filled9.yaml}"
output_dir="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/pointact-rlbench-polar-filled9-bs512-lr1e4}"
ptv3_init_ckpt_file="${PTV3_INIT_CKPT_FILE:-$repo_root/pretrained/Pointcept-Concerto/concerto_large.pth}"
if [[ ! -f "$ptv3_init_ckpt_file" ]]; then
    echo "PTv3 initialization checkpoint does not exist: $ptv3_init_ckpt_file" >&2
    exit 2
fi

# Classification targets must be raw XYZ/Euler/gripper values. The published
# polar archive has action mean/std from regression preprocessing, so create a
# run-local corrected copy instead of mutating the mounted dataset.
data_path="$(python experiments/10_rlbench/prepare_classifier_data_config.py \
    "$data_path" "$output_dir/classifier_input_config")"

accelerate launch "${accelerate_args[@]}" scripts/train.py \
    --model_class VLAEncDec3DWithActionClassificationModel \
    --output_dir "$output_dir" \
    --vlm-name-or-path "${VLM_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}" \
    --data-path "$data_path" \
    --chunk-size 1 \
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-8}" \
    --freeze-vision-tower True \
    --freeze-llm True \
    --freeze-merger True \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --num-train-epochs "${EPOCHS:-1000}" \
    --max-steps "${MAX_STEPS:--1}" \
    --per-device-train-batch-size "${PER_DEVICE_BATCH_SIZE:-512}" \
    --gradient-accumulation-steps 1 \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --merger-lr "${MERGER_LR:-1e-4}" \
    --vision-lr "${VISION_LR:-4e-5}" \
    --weight-decay 0.001 \
    --warmup-steps "${WARMUP_STEPS:-0.03}" \
    --lr-scheduler-type cosine \
    --gradient-checkpointing "${GRADIENT_CHECKPOINTING:-False}" \
    --save-strategy steps \
    --save-steps "${SAVE_STEPS:-500}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT:-10}" \
    --logging-steps "${LOGGING_STEPS:-3}" \
    --report-to "${REPORT_TO:-tensorboard}" \
    --attn-implementation flash_attention_2 \
    --color_aug True \
    --image_aug True \
    --max_grad_norm 3 \
    --use_robot_state True \
    --ctx_embed_size 512 \
    --ptv3_backend concerto \
    --ptv3_patch_size 1024 \
    --ptv3_enc_mode True \
    --ptv3_enc_channels 64 128 256 512 768 \
    --ptv3_enc_depths 3 3 3 12 3 \
    --ptv3_enc_num_head 4 8 16 32 48 \
    --ptv3_input_channels 9 \
    --ptv3_clf_head_pos_bins 100 \
    --action_head_pos_center moe \
    --ptv3_apply_point_ca "${PTV3_APPLY_POINT_CA:-False}" \
    --use_target_reconstruction "${USE_TARGET_RECONSTRUCTION:-False}" \
    --target_reconstruction_weight "${TARGET_RECONSTRUCTION_WEIGHT:-0.1}" \
    --target_mask_loss_weight "${TARGET_MASK_LOSS_WEIGHT:-0.1}" \
    --target_reconstruction_max_points "${TARGET_RECONSTRUCTION_MAX_POINTS:-256}" \
    --ptv3_init_ckpt_file "$ptv3_init_ckpt_file"
