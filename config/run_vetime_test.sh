#!/bin/bash
# Script to run VETime testing on TSB-AD benchmark

echo "========================================"
echo "Starting VETime Testing..."
echo "========================================"

# Define parameters
MODEL_NAME="VETime"
DATASET_DIR="./dataset/TSB-AD/Datasets/TSB-AD-U"
SAVE_DIR="./output/metrics/uni/"
DEVICE="cuda:0"

VISION_NAME="mae_visualize_base.pth"
TS_PATH="./checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth"

# Change this path to evaluate a specific fine-tuned checkpoint
VETIME_PATH="./checkpoints/VETime.pth"

NUM_WORKERS=10

python Test_TSB.py \
    --model_name "$MODEL_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --save_dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --vision_name "$VISION_NAME" \
    --ts_path "$TS_PATH" \
    --vetime_path "$VETIME_PATH" \
    --num_workers $NUM_WORKERS

echo "Testing completed. Results and metrics are saved in $SAVE_DIR"
