#!/bin/bash
# Script to run VETime training

echo "========================================"
echo "Starting VETime Training..."
echo "========================================"

# Define parameters
MODEL_NAME="VETime"
DATASET_PATH="./dataset"
DATASET_TEST_DIR="./dataset/TSB-AD/Datasets/TSB-AD-U"
FILE_LIST="./dataset/TSB-AD/Datasets/File_List/TSB-AD-U.csv"

VISION_PATH="./checkpoints/weight_v"
VISION_NAME="mae_visualize_base.pth"
# Default pre-trained time-series encoder weights
TS_PATH="./checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth"

BATCH_SIZE=32
NUM_EPOCHS=4
NUM_WORKERS=5

# We use standard python command; accelerate is initialized inside train.py
# (Alternatively, you can run with `accelerate launch train.py ...` for distributed training)
python train.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --dataset_test_dir "$DATASET_TEST_DIR" \
    --file_list "$FILE_LIST" \
    --vision_path "$VISION_PATH" \
    --vision_name "$VISION_NAME" \
    --ts_path "$TS_PATH" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --num_workers $NUM_WORKERS

echo "Training completed."
