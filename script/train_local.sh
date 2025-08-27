#!/bin/bash

# Simple local version of train.sh
# Usage: ./train_local.sh [eid] [model_mode] [mask_ratio] [num_epochs]
# Example: ./train_local.sh 754b74d5-7a06-4004-ae0c-72a10b6ed2e6 mm 0.1 10

# Get current directory and move to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parse command line arguments
eid=${1:-"754b74d5-7a06-4004-ae0c-72a10b6ed2e6"}  # Default EID
model_mode=${2:-"mm"}                              # Default to multimodal
mask_ratio=${3:-"0.1"}                            # Default mask ratio
num_epochs=${4:-"10"}                             # Default number of epochs

# Set up local data paths
user_name=$(whoami)
base_path="$HOME/Documents/Code/NEDS/data/datasets"
data_path="$base_path"
config_dir="$(pwd)/src/configs"
dataset_dir="$base_path/ibl_mm"
aligned_data_dir="$base_path/${eid}_aligned"

# Create output directory if it doesn't exist
mkdir -p "$base_path"

# Validate that dataset exists
if [ ! -d "$dataset_dir" ]; then
    echo "Error: Dataset directory not found: $dataset_dir"
    echo "Please run create_dataset_local.sh first to create the dataset."
    exit 1
fi

# Validate that aligned data exists
if [ ! -d "$aligned_data_dir" ]; then
    echo "Error: Aligned data directory not found: $aligned_data_dir"
    echo "Please run prepare_data_local.sh first to create the aligned data."
    exit 1
fi

echo "Starting model training..."
echo "EID: $eid"
echo "Model mode: $model_mode"
echo "Mask ratio: $mask_ratio"
echo "Number of epochs: $num_epochs"
echo "Dataset directory: $dataset_dir"
echo "Aligned data directory: $aligned_data_dir"
echo "Config directory: $config_dir"
echo "Output base path: $base_path"

# Activate virtual environment if it exists
if [ -d "neds_env" ]; then
    echo "Activating virtual environment..."
    python_cmd="./neds_env/bin/python"
elif command -v conda &> /dev/null; then
    echo "Attempting to activate conda environment 'neds'..."
    conda activate neds 2>/dev/null || echo "Warning: Could not activate conda environment 'neds'"
    python_cmd="python"
else
    python_cmd="python"
fi

# Determine which training script to use
train_mode=${5:-"train"}  # Default to regular training, can be "finetune"
if [ "$train_mode" = "finetune" ]; then
    python_file="src/finetune.py"
else
    python_file="src/train.py"
fi

# Set training parameters
num_sessions=1
dummy_size=50000
search="False"
task_var="all"

echo "Training script: $python_file"
echo "Search mode: $search"

# Run the appropriate training based on model mode
if [ "$model_mode" = "mm" ]; then
    echo "Training multimodal model..."
    $python_cmd $python_file --eid "$eid" \
                              --base_path "$base_path" \
                              --data_path "$data_path" \
                              --aligned_data_dir "$aligned_data_dir" \
                              --mask_ratio "$mask_ratio" \
                              --mixed_training \
                              --num_sessions "$num_sessions" \
                              --dummy_size "$dummy_size" \
                              --model_mode "$model_mode" \
                              --enc_task_var "$task_var" \
                              --config_dir "$config_dir" \
                              --modality "ap" "wheel-speed" "whisker-motion-energy" "choice" "block"

elif [ "$model_mode" = "encoding" ] || [ "$model_mode" = "decoding" ]; then
    echo "Training $model_mode model..."
    $python_cmd $python_file --eid "$eid" \
                              --base_path "$base_path" \
                              --data_path "$data_path" \
                              --aligned_data_dir "$aligned_data_dir" \
                              --mask_ratio "$mask_ratio" \
                              --num_sessions "$num_sessions" \
                              --dummy_size "$dummy_size" \
                              --model_mode "$model_mode" \
                              --enc_task_var "all" \
                              --config_dir "$config_dir" \
                              --modality "ap" "wheel-speed" "whisker-motion-energy" "choice" "block"
else
    echo "Error: model_mode '$model_mode' not supported"
    echo "Supported modes: mm, encoding, decoding"
    exit 1
fi

exit_code=$?

# Deactivate environment if activated
if [ "$VIRTUAL_ENV" ]; then
    deactivate
elif command -v conda &> /dev/null && [ "$CONDA_DEFAULT_ENV" = "neds" ]; then
    conda deactivate
fi

echo "Training completed with exit code: $exit_code"
exit $exit_code
