#!/usr/bin/env python3
"""
X-ray Dataset H5 Image Viewer

This script opens and displays images from .h5 files in the X-ray dataset.
Each directory contains two .h5 files: regular and reference (_ref suffix).

Usage: python h5_image_viewer.py <directory_name>
"""

import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_h5_files(directory_path):
    """Find .h5 files in the given directory."""
    h5_files = []
    for file in os.listdir(directory_path):
        if file.endswith('.h5'):
            h5_files.append(os.path.join(directory_path, file))
    return sorted(h5_files)

def load_h5_data(filepath):
    """Load data from an .h5 file."""
    with h5py.File(filepath, 'r') as f:
        # Assuming data is in 'exchange/data' based on our analysis
        data = f['exchange']['data'][:]
    return data

def visualize_complex_data(data, title, subplot_pos):
    """Visualize complex data showing magnitude and phase."""
    # Take the first frame and middle slice for visualization
    slice_data = data[0, data.shape[1]//2, :, :]
    
    plt.subplot(2, 2, subplot_pos)
    plt.imshow(np.abs(slice_data), cmap='viridis')
    plt.title(f'{title} - Magnitude')
    plt.colorbar()
    
    plt.subplot(2, 2, subplot_pos + 1)
    plt.imshow(np.angle(slice_data), cmap='hsv')
    plt.title(f'{title} - Phase')
    plt.colorbar()

def main():
    parser = argparse.ArgumentParser(description='View H5 images from X-ray dataset')
    parser.add_argument('directory', help='Directory name (e.g., camera_man, cell, mandrill)')
    args = parser.parse_args()
    
    # Construct full path
    base_path = Path(__file__).parent
    target_dir = base_path / args.directory / args.directory
    
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist")
        return
    
    # Find H5 files
    h5_files = find_h5_files(target_dir)
    
    if not h5_files:
        print(f"No .h5 files found in {target_dir}")
        return
    
    print(f"Found {len(h5_files)} .h5 files:")
    for file in h5_files:
        print(f"  - {os.path.basename(file)}")
    
    # Count and display total images across all H5 files
    total_images = 0
    
    # Load and display data
    plt.figure(figsize=(15, 10))
    
    for i, filepath in enumerate(h5_files):
        filename = os.path.basename(filepath)
        print(f"\nLoading {filename}...")
        
        try:
            data = load_h5_data(filepath)
            print(f"  Shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  Non-zero elements: {np.sum(np.abs(data) > 1e-10)}")
            print(f"  Max magnitude: {np.max(np.abs(data)):.6f}")
            
            # Count images in this file (assuming images are in dimension 1)
            num_images = data.shape[1] if len(data.shape) >= 2 else 1
            print(f"  Number of images in this file: {num_images}")
            total_images += num_images
            
            # Get image dimensions (assuming last 2 dimensions are height x width)
            if len(data.shape) >= 2:
                image_height = data.shape[-2]
                image_width = data.shape[-1]
                print(f"  Image size: {image_width} x {image_height} pixels")
            
            # Calculate bit width per pixel
            dtype_str = str(data.dtype)
            if 'complex64' in dtype_str:
                bit_width = 64  # 32 bits real + 32 bits imaginary
            elif 'complex128' in dtype_str:
                bit_width = 128  # 64 bits real + 64 bits imaginary
            elif 'float64' in dtype_str:
                bit_width = 64
            elif 'float32' in dtype_str:
                bit_width = 32
            elif 'int64' in dtype_str:
                bit_width = 64
            elif 'int32' in dtype_str:
                bit_width = 32
            elif 'int16' in dtype_str:
                bit_width = 16
            elif 'int8' in dtype_str or 'uint8' in dtype_str:
                bit_width = 8
            else:
                # Extract number from dtype string as fallback
                import re
                numbers = re.findall(r'\d+', dtype_str)
                bit_width = int(numbers[0]) if numbers else 'Unknown'
            
            print(f"  Bit width per pixel: {bit_width} bits")
            
            # Determine if this is regular or reference file
            file_type = "Reference" if "_ref" in filename else "Regular"
            
            # Create visualization (only for first 2 files to fit in subplot)
            if i < 2:
                visualize_complex_data(data, f"{file_type} File", i*2 + 1)
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total number of images in directory '{args.directory}': {total_images}")
    
    plt.tight_layout()
    plt.suptitle(f'X-ray Dataset: {args.directory}', y=0.98)
    plt.show()
    
    # Compare files if we have exactly 2
    if len(h5_files) == 2:
        print(f"\nComparing the two files...")
        try:
            data1 = load_h5_data(h5_files[0])
            data2 = load_h5_data(h5_files[1])
            
            print(f"Arrays identical: {np.array_equal(data1, data2)}")
            print(f"Max absolute difference: {np.max(np.abs(data1 - data2)):.6f}")
            
            # Show difference statistics
            diff = np.abs(data1 - data2)
            print(f"Mean difference: {np.mean(diff):.6f}")
            print(f"Std difference: {np.std(diff):.6f}")
            
        except Exception as e:
            print(f"Error comparing files: {e}")

if __name__ == "__main__":
    main()