#!/usr/bin/env bash
# Optional RGB + dense polar + candidate-material point conditioning smoke run.
set -euo pipefail
cd "$(dirname "$0")/../.."

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

extra_args=()
if [[ -n "${MAX_STEPS:-}" ]]; then
    extra_args+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "${PTV3_INIT_CKPT_FILE:-}" ]]; then
    extra_args+=(--ptv3_init_ckpt_file "$PTV3_INIT_CKPT_FILE")
fi

accelerate launch --num_processes 1 --num_machines 1 scripts/train.py \
    --model_class VLAEncDec3DWithActionClassificationModel \
    --output_dir "${OUTPUT_DIR:-/tmp/pointact_phone_polar_material_conditioned}" \
    --vlm-name-or-path "${VLM_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}" \
    --data-path experiments/10_rlbench/data_configs/data-phone-polar-material-conditioned.yaml \
    --use_polar_material_conditioning True \
    --chunk-size 1 \
    --dataloader-num-workers 0 \
    --freeze-vision-tower True \
    --freeze-llm True \
    --freeze-merger True \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --num-train-epochs 1 \
    --per-device-train-batch-size 1 \
    --learning-rate 5e-5 \
    --merger-lr 5e-5 \
    --vision-lr 2e-5 \
    --weight-decay 0.001 \
    --warmup-steps 0 \
    --lr-scheduler-type cosine \
    --gradient-checkpointing True \
    --save-strategy no \
    --logging-steps 1 \
    --report-to none \
    --color_aug True \
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
    --ptv3_init_copy_input_channels 6 \
    --ptv3_clf_head_pos_bins 100 \
    --action_head_pos_center moe \
    --ptv3_apply_point_ca False \
    "${extra_args[@]}"
