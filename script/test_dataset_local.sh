#!/bin/bash

# Test script to verify the created dataset
# Usage: ./test_dataset_local.sh

# Get current directory and move to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Set up paths
base_path="$HOME/Documents/Code/NEDS/data/datasets"
dataset_path="$base_path/ibl_mm"

echo "Testing dataset creation..."
echo "Dataset path: $dataset_path"

# Check if dataset exists
if [ ! -d "$dataset_path" ]; then
    echo "Error: Dataset directory not found: $dataset_path"
    echo "Please run create_dataset_local.sh first."
    exit 1
fi

# Count files in each split
train_count=$(find "$dataset_path/train" -name "*.npy" | wc -l)
val_count=$(find "$dataset_path/val" -name "*.npy" | wc -l)
test_count=$(find "$dataset_path/test" -name "*.npy" | wc -l)

echo "Dataset statistics:"
echo "  Train files: $train_count"
echo "  Validation files: $val_count"
echo "  Test files: $test_count"
echo "  Total files: $((train_count + val_count + test_count))"

# Test loading one file
echo ""
echo "Testing data loading..."

# Activate virtual environment if it exists
if [ -d "neds_env" ]; then
    python_cmd="./neds_env/bin/python"
else
    python_cmd="python"
fi

# Create a simple test script
cat > /tmp/test_data_loading.py << 'EOF'
import numpy as np
import os
import sys

dataset_path = sys.argv[1]
train_path = os.path.join(dataset_path, 'train')

# Get the first file
files = [f for f in os.listdir(train_path) if f.endswith('.npy')]
if files:
    first_file = os.path.join(train_path, files[0])
    print(f"Loading file: {first_file}")
    
    # Load the data
    data = np.load(first_file, allow_pickle=True).item()
    
    print("Data keys:", list(data.keys()))
    print("Data types:", {k: type(v) for k, v in data.items()})
    
    # Check shapes of array data
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key} shape: {value.shape}")
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            print(f"  {key} length: {len(value)}")
        else:
            print(f"  {key}: {value}")
    
    print("\nData loading test successful!")
else:
    print("No .npy files found in train directory")
    exit(1)
EOF

# Run the test
$python_cmd /tmp/test_data_loading.py "$dataset_path"

# Clean up
rm /tmp/test_data_loading.py

echo ""
echo "Dataset test completed!"
