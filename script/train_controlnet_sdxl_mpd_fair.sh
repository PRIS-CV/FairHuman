MODEL_PATH="stabilityai/sd_xl_base_1.0"
CONTROLNET_PATH="/nvfile-heatstorage/zangxh/intern/wangyx/models/sdxl/controlnet/20250313_train_mix_controlnet_sdxl_fg"
OUTPUT_DIR="/nvfile-heatstorage/zangxh/intern/wangyx/models/sdxl/controlnet/20250325_train_mix_controlnet_sdxl_fg"
VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
CACHE_DIR="/nvfile-heatstorage/zangxh/intern/wangyx/cache"

SEED=42
RESOLUTION=1024
BATCH_SIZE=4
TRAIN_EPOCHS=3
CKPT_STEPS=10000
ACCUMULATION=1
LR=1e-5
LR_SCHEDULER="constant"
WARMUP_STEPS=0
WORKERS=1
export DATA_DIR="./dataset"
DATA_SCRIPT="./dataset/wholebody_dataset.py"
IMAGE_COLUMN="image"
CONDITION_COLUMN="conditioning_image"
CAPTION_COLUMN="text_english"
MASK_HAND_COLUMN="mask_hand"
MASK_FACE_COLUMN="mask_face"

CUDA_VISIBLE_DEVICES=0 python ./examples/controlnet/train_controlnet_mpd_fair.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --controlnet_model_name_or_path $CONTROLNET_PATH \
    --pretrained_vae_model_name_or_path $VAE_PATH\
    --max_train_steps=51000 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --report_to="wandb" \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --seed $SEED \
    --resolution $RESOLUTION \
    --train_batch_size $BATCH_SIZE \
    --num_train_epochs $TRAIN_EPOCHS \
    --checkpointing_steps $CKPT_STEPS \
    --learning_rate $LR \
    --lr_scheduler $LR_SCHEDULER \
    --lr_warmup_steps $WARMUP_STEPS \
    --dataloader_num_workers $WORKERS \
    --mixed_precision "fp16" \
    --train_data_dir $DATA_DIR \
    --train_data_script $DATA_SCRIPT \
    --image_column $IMAGE_COLUMN \
    --conditioning_image_column $CONDITION_COLUMN \
    --caption_column $CAPTION_COLUMN \
    --mask_hand_column $MASK_HAND_COLUMN \
    --mask_face_column $MASK_FACE_COLUMN \
