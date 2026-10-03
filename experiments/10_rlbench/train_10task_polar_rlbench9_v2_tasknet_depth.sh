#!/usr/bin/env bash
# TaskNet + five-stage PointACT fusion + dense depth/normal consistency on RLBench V2.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"

tasknet_default="$repo_root/../PolarAPP/SfP/experiments/checkpoints/huggingface/SfP/TaskNet/TaskNet.pth"
tasknet_checkpoint="${TASKNET_CHECKPOINT:-$tasknet_default}"
if [[ ! -f "$tasknet_checkpoint" ]]; then
    echo "TaskNet checkpoint does not exist: $tasknet_checkpoint" >&2
    exit 2
fi
tasknet_freeze="${TASKNET_FREEZE:-True}"
tasknet_lr_args=()
if [[ "${tasknet_freeze,,}" != "true" ]]; then
    tasknet_lr_args=(--polarapp_lr "${TASKNET_LR:-1e-5}")
fi

ptv3_backend="${PTV3_BACKEND:-concerto}"
case "$ptv3_backend" in
    concerto)
        ptv3_default_checkpoint="$repo_root/pretrained/Pointcept-Concerto/concerto_large.pth"
        ptv3_channels=(64 128 256 512 768)
        ptv3_heads=(4 8 16 32 48)
        ;;
    utonia)
        ptv3_default_checkpoint="$repo_root/pretrained/Pointcept-Utonia/utonia.pth"
        ptv3_channels=(54 108 216 432 576)
        ptv3_heads=(3 6 12 24 32)
        ;;
    *)
        echo "Unsupported PTV3_BACKEND=$ptv3_backend (expected concerto or utonia)." >&2
        exit 2
        ;;
esac
ptv3_checkpoint="${PTV3_INIT_CKPT_FILE:-$ptv3_default_checkpoint}"
if [[ ! -f "$ptv3_checkpoint" ]]; then
    echo "PTv3 initialization checkpoint does not exist: $ptv3_checkpoint" >&2
    exit 2
fi

gpus="${GPUS:-1}"
accelerate_args=(--num_processes "$gpus" --num_machines 1)
if (( gpus > 1 )); then
    accelerate_args=(--multi_gpu "${accelerate_args[@]}")
fi

# The same V2 sidecars are used: the encoder performs the TaskNet channel
# reorder/coordinate replacement in memory, while the calibrated metadata is
# retained for 3D routing and depth reconstruction.
data_path="${DATA_PATH:-experiments/10_rlbench/data_configs/data-10task-polar-rlbench9-v2-incomplete-sfp-wild-proxy.yaml}"
output_dir="${OUTPUT_DIR:-$repo_root/checkpoints/rlbench/pointact-rlbench9-v2-tasknet-depth}"

accelerate launch "${accelerate_args[@]}" scripts/train.py \
    --model_class VLAEncDec3DWithActionRegressionModel \
    --output_dir "$output_dir" \
    --run-name "${RUN_NAME:-$(basename "$output_dir")}" \
    --vlm-name-or-path "${VLM_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}" \
    --data-path "$data_path" \
    --chunk-size 1 \
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-8}" \
    --dataloader-prefetch-factor "${DATALOADER_PREFETCH_FACTOR:-2}" \
    --dataloader-persistent-workers "${DATALOADER_PERSISTENT_WORKERS:-True}" \
    --dataloader-pin-memory "${DATALOADER_PIN_MEMORY:-True}" \
    --freeze-vision-tower True --freeze-llm True --freeze-merger True \
    --bf16 "${BF16:-True}" --tf32 "${TF32:-True}" --fp16 "${FP16:-False}" \
    --num-train-epochs "${EPOCHS:-1000}" --max-steps "${MAX_STEPS:-40000}" \
    --per-device-train-batch-size "${PER_DEVICE_BATCH_SIZE:-8}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
    --learning-rate "${LEARNING_RATE:-1e-4}" --weight-decay 0.001 \
    --optim "${OPTIM:-adamw_torch}" \
    --warmup-steps "${WARMUP_STEPS:-0.03}" --lr-scheduler-type cosine \
    --gradient-checkpointing "${GRADIENT_CHECKPOINTING:-False}" \
    --save-strategy steps --save-steps "${SAVE_STEPS:-500}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT:-10}" --logging-steps "${LOGGING_STEPS:-3}" \
    --report-to "${REPORT_TO:-tensorboard}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    --color_aug False --image_aug False --max_grad_norm 3 \
    --use_robot_state True --ctx_embed_size 512 \
    --ptv3_backend "$ptv3_backend" --ptv3_patch_size 1024 --ptv3_enc_mode True \
    --ptv3_enc_channels "${ptv3_channels[@]}" \
    --ptv3_enc_depths 3 3 3 12 3 --ptv3_enc_num_head "${ptv3_heads[@]}" \
    --ptv3_input_channels 9 --ptv3_init_copy_input_channels 6 \
    --ptv3_apply_point_ca False --ptv3_init_ckpt_file "$ptv3_checkpoint" \
    --action_regression_loss l2 --action_head_pos_center zero \
    --max_state_dim 10 --max_action_dim 10 \
    --polar_enabled True --polar_backbone polarapp_taskaware \
    --polarapp_checkpoint "$tasknet_checkpoint" \
    --polarapp_allow_random_init False --polarapp_freeze "$tasknet_freeze" \
    --polarapp_pyramid_channels "${TASKNET_PYRAMID_CHANNELS:-128}" \
    --polarapp_input_mode sfp_proxy "${tasknet_lr_args[@]}" \
    --sfp_feature_levels x1 x2 x3 x4 x5 \
    --polar_neighbor_radius "${POLAR_NEIGHBOR_RADIUS:-1}" \
    --polar_max_tokens_per_group "${POLAR_MAX_TOKENS_PER_GROUP:-32}" \
    --polar_token_mode "${POLAR_TOKEN_MODE:-local}" \
    --polar_max_views 1 --polar_writeback False \
    --use_polar_depth_self_supervision True \
    --polar_depth_loss_weight "${POLAR_DEPTH_LOSS_WEIGHT:-0.1}" \
    --polar_consistency_weight "${POLAR_CONSISTENCY_WEIGHT:-1.0}" \
    --sparse_depth_consistency_weight "${SPARSE_DEPTH_WEIGHT:-1.0}" \
    --depth_smoothness_weight "${DEPTH_SMOOTHNESS_WEIGHT:-0.01}" \
    --polar_depth_keep_probability "${POLAR_DEPTH_KEEP_PROBABILITY:-0.7}" \
    --polar_depth_min "${POLAR_DEPTH_MIN:-0.05}" \
    --polar_depth_max "${POLAR_DEPTH_MAX:-4.5}"
