ulimit -u 2048

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GPUS=${GPUS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-512}

# Build accelerate arguments based on GPU count
if [ $GPUS -eq 1 ]; then
    echo "Single GPU mode"
    ACCELERATE_ARGS="--num_processes 1 --num_machines 1"
elif [ $GPUS -gt 1 ] && [ $NUM_NODES -eq 1 ]; then
    echo "Multi-GPU single node mode"
    ACCELERATE_ARGS="--multi_gpu --num_processes ${GPUS} --num_machines 1 --machine_rank 0"
else
    echo "Multi-GPU multi-node mode"
    ACCELERATE_ARGS="--multi_gpu --num_processes ${GPUS} --num_machines ${NUM_NODES} --machine_rank ${SLURM_NODEID}  --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT}"
fi

# datasets
dataset=${DATASET_CONFIG:-experiments/10_rlbench/data_configs/data-hybridvla-point-clf-frontview.yaml}
dataset_name=${DATASET_NAME:-keysteps-euler-points.frontview-no.image-aug.30}

# hparams
lr=${LEARNING_RATE:-1e-4}
mlr=${MERGER_LR:-1e-4}
vlr=${VISION_LR:-4e-5}

chunk_size=1
epoch=${EPOCHS:-1000} #3000

model_name_or_path=
run_name=${RUN_NAME:-${dataset_name}_ck${chunk_size}_lr${lr}_gpu${GPUS}_bs${PER_DEVICE_BATCH_SIZE}_epoch${epoch}}

output_dir=${OUTPUT_DIR:-$SCRATCH/datasets/PointAct_exprs/rlbench/hybridvla_10tasks/pointact/VLAEncDec3DWithActionClassificationModel-concerto-${run_name}-freeze.vlm}
ptv3_init_ckpt_file=${PTV3_INIT_CKPT_FILE:-$SCRATCH/datasets/pretrained/Pointcept-Concerto/concerto_large.pth}
# output_base=null # with time in directory name

# Determine TF32 support
TF32_SUPPORT="False"
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
echo "GPU Compute Capability: $COMPUTE_CAP"
# Check if Ampere or newer (compute capability >= 8.0)
if [ "$MAJOR" -ge 8 ]; then
    TF32_SUPPORT="True"
fi
echo "TF32_SUPPORT: $TF32_SUPPORT"


accelerate launch $ACCELERATE_ARGS scripts/train.py \
    --model_class VLAEncDec3DWithActionClassificationModel \
    --output_dir "${output_dir}" \
    ${model_name_or_path:+--model-name-or-path $model_name_or_path} \
    --vlm-name-or-path "${VLM_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}" \
    --data-path "${dataset}" \
    --chunk-size ${chunk_size} \
    --dataloader-num-workers ${DATALOADER_NUM_WORKERS:-8} \
    --freeze-vision-tower True \
    --freeze-llm True \
    --freeze-merger True \
    --bf16 True \
    --tf32 ${TF32_SUPPORT} \
    --fp16 False \
    --num-train-epochs ${epoch} \
    --max-steps ${MAX_STEPS:--1} \
    --per-device-train-batch-size ${PER_DEVICE_BATCH_SIZE} \
    --gradient-accumulation-steps 1 \
    --learning-rate ${lr} \
    --merger-lr ${mlr} \
    --vision-lr ${vlr} \
    --weight-decay 0.001 \
    --warmup-steps ${WARMUP_STEPS:-0.03} \
    --lr-scheduler-type cosine \
    --gradient-checkpointing ${GRADIENT_CHECKPOINTING:-False} \
    --save-strategy steps \
    --logging-steps ${LOGGING_STEPS:-3} \
    --save-steps ${SAVE_STEPS:-500} \
    --save-total-limit ${SAVE_TOTAL_LIMIT:-10} \
    --run-name "${run_name}" \
    --attn-implementation flash_attention_2 \
    --log_level info \
    --report-to "${REPORT_TO:-tensorboard}" \
    --color_aug "${COLOR_AUG:-True}" \
    --max_grad_norm 3 \
    --use_robot_state True \
    --ctx_embed_size 512 \
    --ptv3_backend concerto \
    --ptv3_patch_size 1024 \
    --ptv3_enc_mode True \
    --ptv3_enc_channels 64 128 256 512 768 \
    --ptv3_enc_depths 3 3 3 12 3 \
    --ptv3_enc_num_head 4 8 16 32 48 \
    --ptv3_input_channels 6 \
    --ptv3_clf_head_pos_bins 100 \
    --action_head_pos_center moe \
    --ptv3_apply_point_ca False \
    --ptv3_init_ckpt_file "${ptv3_init_ckpt_file}"

    # VLAEncDec3DClassificationModel
    # VLAEncDec3DWithActionClassificationModel
    
