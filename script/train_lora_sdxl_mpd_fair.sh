MODEL_PATH="stabilityai/sd_xl_base_1.0"
OUTPUT_DIR="/nvfile-heatstorage/zangxh/intern/wangyx/models/sdxl/lora/20250313_train_lora_sdxl_fg"
VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
CACHE_DIR="/nvfile-heatstorage/zangxh/intern/wangyx/cache"

SEED=42
RESOLUTION=1024
BATCH_SIZE=1
TRAIN_EPOCHS=2
CKPT_STEPS=10000
ACCUMULATION=1
LR=1e-5
LR_SCHEDULER="constant"
WARMUP_STEPS=0
WORKERS=2
export DATA_DIR="./dataset"
DATA_SCRIPT="./dataset/wholebody_dataset.py"
IMAGE_COLUMN="image"
CAPTION_COLUMN="text_english"
MASK_HAND_COLUMN="mask_hand"
MASK_FACE_COLUMN="mask_face"

CUDA_VISIBLE_DEVICES=0 python ./examples/text_to_image/train_text_to_image_lora_sdxl_mpd_fair.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --pretrained_vae_model_name_or_path $VAE_PATH \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --seed $SEED \
    --max_train_steps=20000 \
    --random_flip \
    --resolution $RESOLUTION \
    --rank 256 \
    --train_batch_size $BATCH_SIZE \
    --num_train_epochs $TRAIN_EPOCHS \
    --checkpointing_steps $CKPT_STEPS \
    --checkpoints_total_limit 2 \
    --gradient_accumulation_steps $ACCUMULATION \
    --learning_rate $LR \
    --lr_scheduler $LR_SCHEDULER \
    --lr_warmup_steps $WARMUP_STEPS \
    --dataloader_num_workers $WORKERS \
    --mixed_precision "fp16" \
    --train_data_dir $DATA_DIR \
    --train_data_script $DATA_SCRIPT \
    --image_column $IMAGE_COLUMN \
    --caption_column $CAPTION_COLUMN \
    --mask_hand_column $MASK_HAND_COLUMN \
    --mask_face_column $MASK_FACE_COLUMN \
