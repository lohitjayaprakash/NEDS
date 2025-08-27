#!/bin/bash

# Local evaluation script for NEDS
# This script evaluates models using local data and environment

# Set default values
EID=${1:-"754b74d5-7a06-4004-ae0c-72a10b6ed2e6"}
TRAIN_MODE=${2:-"finetune"}
MODEL_MODE=${3:-"mm"}
MASK_RATIO=${4:-0.1}
TASK_VAR=${5:-"all"}
SEARCH=${6:-"False"}

# Extract basename from EID if it contains a path
if [[ "$EID" == */* ]]; then
    EID_NAME=$(basename "$EID")
else
    EID_NAME="$EID"
fi

echo "Starting model evaluation..."
echo "EID: $EID"
echo "Train mode: $TRAIN_MODE"
echo "Model mode: $MODEL_MODE"
echo "Mask ratio: $MASK_RATIO"
echo "Task variable: $TASK_VAR"
echo "Search: $SEARCH"
echo "Dataset directory: $DATASET_DIR"
echo "Aligned data directory: $(pwd)/data/datasets/${EID_NAME}_aligned"
echo "Config directory: $CONFIG_DIR"
echo "Output base path: $OUTPUT_BASE_PATH"echo "Activating virtual environment..."
# Activate the virtual environment
source neds_env/bin/activate

echo "Evaluation script: src/eval.py"

# Set finetune flag
if [ "$TRAIN_MODE" = "finetune" ]; then
    FINETUNE="--finetune"
else
    FINETUNE=""
fi

# Set search flag
if [ "$SEARCH" = "True" ]; then
    echo "Doing hyperparameter search"
    SEARCH_FLAG="--param_search"
else
    echo "Not doing hyperparameter search"
    SEARCH_FLAG=""
fi

echo "Evaluating multimodal model..."

if [ "$MODEL_MODE" = "mm" ]; then
    python src/eval.py --eid ${EID} \
                        --mask_mode temporal \
                        --mask_ratio ${MASK_RATIO} \
                        --seed 42 \
                        --base_path "$(pwd)/data/datasets" \
                        --num_sessions 1 \
                        ${FINETUNE} \
                        --model_mode ${MODEL_MODE} \
                        --wandb \
                        --overwrite \
                        --enc_task_var ${TASK_VAR} \
                        --data_path "$(pwd)/data/datasets" \
                        --aligned_data_dir "$(pwd)/data/datasets/${EID_NAME}_aligned" \
                        ${SEARCH_FLAG}
elif [ "$MODEL_MODE" = "encoding" ] || [ "$MODEL_MODE" = "decoding" ]; then
    python src/eval.py --eid ${EID} \
                        --mask_mode temporal \
                        --mask_ratio ${MASK_RATIO} \
                        --seed 42 \
                        --base_path "$(pwd)/data/datasets" \
                        --num_sessions 1 \
                        ${FINETUNE} \
                        --model_mode ${MODEL_MODE} \
                        --wandb \
                        --overwrite \
                        --enc_task_var ${TASK_VAR} \
                        --data_path "$(pwd)/data/datasets" \
                        --aligned_data_dir "$(pwd)/data/datasets/${EID_NAME}_aligned" \
                        ${SEARCH_FLAG}
else
    echo "model_mode: $MODEL_MODE not supported"
    exit 1
fi

echo "Evaluation completed!"
echo "Results should be in: data/datasets/results/"

# Deactivate virtual environment
deactivate
