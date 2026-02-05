#!/usr/bin/env python3
"""
Analyze X-ray dataset options for HDC classifier
Determine best approach to use this data with the existing HDC system
"""

import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_h5_data():
    """Analyze the 32x32 H5 diffraction data"""
    print("\n" + "="*80)
    print("ANALYZING H5 DIFFRACTION DATA (32x32)")
    print("="*80)

    xray_base = Path("x-ray dataset")
    datasets = {
        "cameraman": "cameraman_32by32/cameraman_32by32/data_camera_man_phase_n5.5e6.h5",
        "cell": "cell_32by32/cell_32by32/data_cell_phase_n3e6.h5",
        "mandrill": "mandrill_32by32/mandrill_32by32/data_mandrill_phase_n5.5e6.h5"
    }

    all_data = {}
    for name, filepath in datasets.items():
        full_path = xray_base / filepath
        with h5py.File(full_path, 'r') as f:
            data = f['exchange']['data'][:]
            # Take magnitude since imaginary part is zero
            data = np.abs(data)
            all_data[name] = data

            print(f"\n{name.upper()}:")
            print(f"  Shape: {data.shape}")
            print(f"  Num images: {data.shape[1]}")
            print(f"  Image size: {data.shape[2]}x{data.shape[3]}")
            print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}")

            # Check a few sample images
            sample_img = data[0, 0]
            print(f"  Sample image (first):")
            print(f"    Non-zero pixels: {np.count_nonzero(sample_img)}/{sample_img.size}")
            print(f"    Unique values: {len(np.unique(sample_img))}")

    return all_data

def analyze_phantom_data():
    """Analyze the phantom (ground truth) data"""
    print("\n" + "="*80)
    print("ANALYZING PHANTOM (GROUND TRUTH) DATA")
    print("="*80)

    xray_base = Path("x-ray dataset")
    datasets = {
        "cameraman": "cameraman_32by32/cameraman_32by32/phantom",
        "cell": "cell_32by32/cell_32by32/phantom",
        "mandrill": "mandrill_32by32/mandrill_32by32/phantom"
    }

    phantom_data = {}
    for name, dirpath in datasets.items():
        full_path = xray_base / dirpath
        beta = np.load(full_path / "grid_beta.npy")
        delta = np.load(full_path / "grid_delta.npy")

        phantom_data[name] = {'beta': beta, 'delta': delta}

        print(f"\n{name.upper()}:")
        print(f"  Beta (absorption) shape: {beta.shape}")
        print(f"    Range: [{beta.min():.6f}, {beta.max():.6f}]")
        print(f"  Delta (refractive) shape: {delta.shape}")
        print(f"    Range: [{delta.min():.6f}, {delta.max():.6f}]")

    return phantom_data

def propose_labeled_dataset_approach():
    """Propose approaches to create a labeled dataset"""
    print("\n" + "="*80)
    print("PROPOSED APPROACHES FOR LABELED DATASET")
    print("="*80)

    print("\n1. SIMPLE CLASS-BASED APPROACH (Recommended for initial testing)")
    print("   " + "-"*70)
    print("   - Use the 1225 diffraction images from each sample as different classes")
    print("   - Classes: cameraman, cell, mandrill (3 classes)")
    print("   - Total images: 3,675 (1225 per class)")
    print("   - Image size: 32x32 pixels")
    print("   - Advantages:")
    print("     * Simple and direct")
    print("     * No need for autoencoders")
    print("     * Uses raw diffraction data (what hardware would see)")
    print("     * Each class has same number of samples (balanced)")
    print("   - Disadvantages:")
    print("     * Only 3 classes (less diverse than QuickDraw's 10)")
    print("     * All images from same sample type are labeled identically")

    print("\n2. AUTOENCODER-BASED CLUSTERING APPROACH")
    print("   " + "-"*70)
    print("   - Train autoencoder on all 3,675 diffraction images")
    print("   - Use encoder features to cluster into N classes (configurable)")
    print("   - Can create 5, 10, or more classes through clustering")
    print("   - Advantages:")
    print("     * Configurable number of classes")
    print("     * Unsupervised learning of meaningful features")
    print("     * Could discover patterns within each sample type")
    print("   - Disadvantages:")
    print("     * More complex pipeline")
    print("     * Requires additional training step")
    print("     * Labels are synthetic (no ground truth)")

    print("\n3. PHANTOM-BASED AUGMENTATION APPROACH")
    print("   " + "-"*70)
    print("   - Use phantom images (ground truth) as base")
    print("   - Generate synthetic classes by:")
    print("     * Cropping different regions")
    print("     * Applying different transformations")
    print("     * Using beta and delta as separate channels")
    print("   - Advantages:")
    print("     * Uses ground truth data")
    print("     * Can generate many synthetic classes")
    print("   - Disadvantages:")
    print("     * Phantom images are larger (would need resizing/cropping)")
    print("     * Not actual x-ray measurements (different from what hardware sees)")

    print("\n4. MIXED DIFFRACTION PATTERNS APPROACH")
    print("   " + "-"*70)
    print("   - Divide each sample's 1225 images into multiple classes")
    print("   - E.g., cameraman images 0-122 = class 0, 123-245 = class 1, etc.")
    print("   - Can create 10 classes (367 images each, with some having 368)")
    print("   - Advantages:")
    print("     * Gets to 10 classes like QuickDraw")
    print("     * No complex processing needed")
    print("   - Disadvantages:")
    print("     * Arbitrary class divisions")
    print("     * May not have meaningful differences between classes")

def main():
    print("\n" + "="*80)
    print("X-RAY DATASET ANALYSIS FOR HDC CLASSIFIER")
    print("="*80)

    # Analyze both types of data
    h5_data = analyze_h5_data()
    phantom_data = analyze_phantom_data()

    # Propose approaches
    propose_labeled_dataset_approach()

    # Final recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\nFor immediate compatibility with your HDC classifier:")
    print("\n  START WITH APPROACH #1 (Simple Class-Based)")
    print("  - 3 classes: cameraman, cell, mandrill")
    print("  - 1225 images per class (32x32 pixels)")
    print("  - Use diffraction magnitude data directly")
    print("  - Split: 980 training + 245 testing per class")
    print("  - Normalize pixel values to [0, 255] range")
    print("\n  This requires minimal changes to train_hdc.py:")
    print("  - Add new dataset loader for X-ray data")
    print("  - Use same CNN+HDC pipeline")
    print("  - Keep same 8-bit quantization")
    print("\nIf you need more classes (e.g., 10 like QuickDraw):")
    print("  - Use APPROACH #2 (Autoencoder clustering) or")
    print("  - Use APPROACH #4 (Mixed diffraction patterns)")
    print("\nNext steps:")
    print("  1. Create XRayDataset class similar to QuickDrawDataset")
    print("  2. Test with 3 classes first to validate pipeline")
    print("  3. If needed, implement autoencoder for configurable classes")

if __name__ == "__main__":
    main()
