#!/bin/bash
# Script to run baseline models on the TSB-AD benchmark

echo "Starting evaluation of Baseline models on TSB-AD-U dataset..."

# Set the path to your dataset
DATASET_DIR="./dataset/TSB-AD/Datasets/TSB-AD-U"
OUTPUT_DIR="./output/baselines_results"

# List of baselines to run
BASELINES=("MOMENT" "TimesFM" "TranAD" "USAD" "OmniAnomaly")

for MODEL in "${BASELINES[@]}"; do
    echo "========================================"
    echo "Running Baseline: $MODEL"
    echo "========================================"
    
    python ./dataset/TSB-AD-main/TSB_AD/main.py \
        --data_direc "$DATASET_DIR" \
        --AD_Name "$MODEL" \
        --output_dir "$OUTPUT_DIR"
        
    echo "Finished $MODEL evaluation."
    echo ""
done

echo "All baseline evaluations completed. Results are saved in $OUTPUT_DIR"
