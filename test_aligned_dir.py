#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Quick test to check aligned_data_dir parameter passing
aligned_data_dir = "/Users/pj/Documents/Code/NEDS/data/datasets/754b74d5-7a06-4004-ae0c-72a10b6ed2e6_aligned"

print(f"Testing aligned_data_dir: {aligned_data_dir}")
print(f"Directory exists: {os.path.exists(aligned_data_dir)}")
print(f"Is directory: {os.path.isdir(aligned_data_dir)}")

if aligned_data_dir:
    print("aligned_data_dir is truthy")
else:
    print("aligned_data_dir is falsy")

# Test if we can import the function
try:
    from utils.dataset_utils import load_ibl_dataset
    print("Successfully imported load_ibl_dataset")
except Exception as e:
    print(f"Failed to import: {e}")
