#!/bin/bash

# Simple local version of prepare_data.sh
# Usage: ./prepare_data_local.sh [num_sessions] [eid]
# Example: ./prepare_data_local.sh 1 EID_VALUE

# Get current directory and move to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parse command line arguments
num_sessions=${1:-1}  # Default to 1 if not provided
eid=${2:-"None"}      # Default to "None" if not provided

# Set up local data path
user_name=$(whoami)
base_path="$HOME/Documents/Code/NEDS/data/datasets"

# Create datasets directory if it doesn't exist
mkdir -p "$base_path"

# Validate num_sessions is an integer
if ! [[ "$num_sessions" =~ ^[0-9]+$ ]]; then
    echo "Error: num_sessions must be an integer"
    exit 1
fi

# Validate parameters based on session count
if [ "$num_sessions" -eq 1 ]; then
    echo "Download data for single session"
    if [ "$eid" = "None" ]; then
        echo "Error: eid must be provided for single session"
        echo "Usage: $0 1 <EID_VALUE>"
        exit 1
    fi
else
    echo "Download data for multiple sessions"
    eid="None"
fi

echo "Starting data preparation..."
echo "Number of sessions: $num_sessions"
echo "EID: $eid"
echo "Base path: $base_path"

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

# Run the Python script
$python_cmd src/prepare_data.py --n_sessions "$num_sessions" \
                                --eid "$eid" \
                                --base_path "$base_path" \
                                --n_workers 1

exit_code=$?

# Deactivate environment if activated
if [ "$VIRTUAL_ENV" ]; then
    deactivate
elif command -v conda &> /dev/null && [ "$CONDA_DEFAULT_ENV" = "neds" ]; then
    conda deactivate
fi

echo "Data preparation completed with exit code: $exit_code"
exit $exit_code
