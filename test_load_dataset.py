#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Test the eval.py argument parsing
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="754b74d5-7a06-4004-ae0c-72a10b6ed2e6")
ap.add_argument("--aligned_data_dir", type=str, default=None, help="Path to aligned data directory")

# Simulate command line args
test_args = [
    "--eid", "754b74d5-7a06-4004-ae0c-72a10b6ed2e6",
    "--aligned_data_dir", "/Users/pj/Documents/Code/NEDS/data/datasets/754b74d5-7a06-4004-ae0c-72a10b6ed2e6_aligned"
]

args = ap.parse_args(test_args)

print(f"Parsed eid: {args.eid}")
print(f"Parsed aligned_data_dir: {args.aligned_data_dir}")
print(f"aligned_data_dir is truthy: {bool(args.aligned_data_dir)}")

# Test calling load_ibl_dataset directly
try:
    from utils.dataset_utils import load_ibl_dataset
    print("About to call load_ibl_dataset with aligned_data_dir...")
    
    result = load_ibl_dataset(
        cache_dir="dummy",
        user_or_org_name="dummy", 
        aligned_data_dir=args.aligned_data_dir,
        eid=args.eid,
        num_sessions=1,
        batch_size=16,
        seed=42
    )
    print("Successfully called load_ibl_dataset!")
    print(f"Result type: {type(result)}")
    
except Exception as e:
    print(f"Error calling load_ibl_dataset: {e}")
    import traceback
    traceback.print_exc()
