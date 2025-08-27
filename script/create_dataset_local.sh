#!/bin/bash

# Simple local version of create_dataset.sh
# Usage: ./create_dataset_local.sh [eid] [model_mode] [mask_ratio]
# Example: ./create_dataset_local.sh 754b74d5-7a06-4004-ae0c-72a10b6ed2e6 mm 0.1

# Get current directory and move to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parse command line arguments
eid=${1:-"754b74d5-7a06-4004-ae0c-72a10b6ed2e6"}  # Default EID
model_mode=${2:-"mm"}                              # Default to multimodal
mask_ratio=${3:-"0.1"}                            # Default mask ratio

# Set up local data paths
user_name=$(whoami)
base_path="$HOME/Documents/Code/NEDS/data/datasets"
aligned_data_dir="$base_path/${eid}_aligned"

# Create output directory if it doesn't exist
mkdir -p "$base_path"

# Validate that aligned data exists
if [ ! -d "$aligned_data_dir" ]; then
    echo "Error: Aligned data directory not found: $aligned_data_dir"
    echo "Please run prepare_data_local.sh first to create the aligned data."
    exit 1
fi

echo "Starting dataset creation..."
echo "EID: $eid"
echo "Model mode: $model_mode"
echo "Mask ratio: $mask_ratio"
echo "Aligned data directory: $aligned_data_dir"
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

# Run the Python script with local aligned data directory
$python_cmd src/create_dataset.py --eid "$eid" \
                                  --base_path "$base_path" \
                                  --data_path "$base_path" \
                                  --aligned_data_dir "$aligned_data_dir" \
                                  --num_sessions 1 \
                                  --model_mode "$model_mode" \
                                  --mask_ratio "$mask_ratio" \
                                  --modality "ap" "wheel-speed" "whisker-motion-energy" "choice" "block"

exit_code=$?

# Deactivate environment if activated
if [ "$VIRTUAL_ENV" ]; then
    deactivate
elif command -v conda &> /dev/null && [ "$CONDA_DEFAULT_ENV" = "neds" ]; then
    conda deactivate
fi

echo "Dataset creation completed with exit code: $exit_code"
exit $exit_code
