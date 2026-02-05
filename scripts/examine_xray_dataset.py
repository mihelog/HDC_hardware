#!/usr/bin/env python3
"""
Examine X-ray dataset to understand structure and determine if labels exist
"""

import h5py
import numpy as np
import os
from pathlib import Path

def examine_h5_file(filepath):
    """Examine the structure of an H5 file"""
    print(f"\n{'='*80}")
    print(f"Examining: {filepath}")
    print(f"{'='*80}")

    with h5py.File(filepath, 'r') as f:
        print("\nTop-level keys:")
        for key in f.keys():
            print(f"  - {key}")

        # Recursively explore structure
        def explore_group(group, indent=0):
            for key in group.keys():
                item = group[key]
                prefix = "  " * indent
                if isinstance(item, h5py.Dataset):
                    print(f"{prefix}Dataset: {key}")
                    print(f"{prefix}  Shape: {item.shape}")
                    print(f"{prefix}  Dtype: {item.dtype}")
                    if item.size < 10:
                        print(f"{prefix}  Value: {item[:]}")
                elif isinstance(item, h5py.Group):
                    print(f"{prefix}Group: {key}")
                    explore_group(item, indent + 1)

        print("\nFull structure:")
        explore_group(f)

        # Check for common label keys
        label_keys = ['labels', 'label', 'classes', 'class', 'targets', 'target', 'y']
        print("\nSearching for label keys:")
        found_labels = False

        def search_labels(group, path=""):
            nonlocal found_labels
            for key in group.keys():
                item = group[key]
                current_path = f"{path}/{key}" if path else key
                if key.lower() in label_keys:
                    print(f"  Found potential label: {current_path}")
                    if isinstance(item, h5py.Dataset):
                        print(f"    Shape: {item.shape}, Dtype: {item.dtype}")
                        found_labels = True
                elif isinstance(item, h5py.Group):
                    search_labels(item, current_path)

        search_labels(f)

        if not found_labels:
            print("  No label datasets found in file")

        # Try to access the data
        print("\nAttempting to access 'exchange/data':")
        try:
            data = f['exchange']['data'][:]
            print(f"  Successfully loaded data")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Sample values (first few elements):")
            print(f"    Real part range: [{np.min(data.real):.6f}, {np.max(data.real):.6f}]")
            print(f"    Imag part range: [{np.min(data.imag):.6f}, {np.max(data.imag):.6f}]")
            print(f"    Magnitude range: [{np.min(np.abs(data)):.6f}, {np.max(np.abs(data)):.6f}]")
        except Exception as e:
            print(f"  Error: {e}")

def main():
    # Examine 32x32 datasets
    xray_base = Path("x-ray dataset")

    datasets_32 = [
        "cameraman_32by32/cameraman_32by32",
        "cell_32by32/cell_32by32",
        "mandrill_32by32/mandrill_32by32"
    ]

    print("\n" + "="*80)
    print("X-RAY DATASET ANALYSIS - 32x32 Images")
    print("="*80)

    for dataset_path in datasets_32:
        full_path = xray_base / dataset_path
        if full_path.exists():
            h5_files = sorted(full_path.glob("*.h5"))
            if h5_files:
                # Examine just the first file from each dataset
                examine_h5_file(h5_files[0])

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nConclusion:")
    print("  - The X-ray dataset contains ptychography diffraction patterns (complex data)")
    print("  - These are NOT labeled classification datasets")
    print("  - Shape: (1, 4488, 72, 72) - one measurement series with 4488 probe positions")
    print("  - Each file represents measurements from a SINGLE sample (cameraman, cell, mandrill)")
    print("  - NO class labels are present in the .h5 files")
    print("\nRecommendations:")
    print("  1. Cannot directly replace QuickDraw dataset which needs labeled images")
    print("  2. Options:")
    print("     a) Use reconstruction results (phantom/*.npy or n*/obj_checkpoint.npy)")
    print("        as images and create synthetic labels")
    print("     b) Use an unsupervised approach (autoencoder) to learn features")
    print("     c) Treat each sample type as a different class (3 classes: cameraman, cell, mandrill)")
    print("     d) Generate multiple crops/views from reconstructions as different samples")

if __name__ == "__main__":
    main()
