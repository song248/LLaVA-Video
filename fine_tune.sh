#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7,max_split_size_mb:64

# Set up the data folder
VIDEO_FOLDER="data"
DATA_YAML="labeled.json"

############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
# alias python=python3

################ Job Config ################
# LLM_VERSION="Qwen/Qwen1.5-1.8B"
LLM_VERSION="Qwen/Qwen3-1.7B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

BASE_RUN_NAME="llava-lora-fight-vqa"
RUN_NAME="${BASE_RUN_NAME}-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}"
# PREV_STAGE_CHECKPOINT="lmms-lab/LLaVA-Video-7B-Qwen2"
# PREV_STAGE_CHECKPOINT="Qwen/Qwen1.5-1.8B"
PREV_STAGE_CHECKPOINT="Qwen/Qwen3-1.7B"

echo "RUN_NAME: ${RUN_NAME}"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"

deepspeed --master_port 30000 \
    llava/train/train_mem.py \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version qwen_1_5 \
    --data_path $DATA_YAML \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_vision_encoder,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-5 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --fp16 False\
    --run_name $RUN_NAME \
    --output_dir ./work_dirs/$RUN_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile False \
    --dataloader_drop_last True \
    --frames_upbound 128 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 1 \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_bias "none"
exit 0;
