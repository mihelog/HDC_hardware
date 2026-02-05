#!/usr/bin/env python3
"""
Visualize sample images from X-ray dataset
"""

import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_samples():
    """Visualize sample images from each class"""
    xray_base = Path("x-ray dataset")
    datasets = {
        "cameraman": "cameraman_32by32/cameraman_32by32/data_camera_man_phase_n5.5e6.h5",
        "cell": "cell_32by32/cell_32by32/data_cell_phase_n3e6.h5",
        "mandrill": "mandrill_32by32/mandrill_32by32/data_mandrill_phase_n5.5e6.h5"
    }

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('X-ray Dataset Samples (32x32 diffraction patterns)', fontsize=16)

    for row, (name, filepath) in enumerate(datasets.items()):
        full_path = xray_base / filepath
        with h5py.File(full_path, 'r') as f:
            data = f['exchange']['data'][:]
            # Take magnitude
            data = np.abs(data[0])  # Remove batch dimension

            # Show 5 sample images from this class
            sample_indices = [0, 300, 600, 900, 1200]
            for col, idx in enumerate(sample_indices):
                img = data[idx]
                axes[row, col].imshow(img, cmap='viridis')
                axes[row, col].axis('off')
                if col == 0:
                    axes[row, col].set_title(f'{name.upper()}\n(img {idx})', fontsize=10)
                else:
                    axes[row, col].set_title(f'img {idx}', fontsize=10)

                # Add value statistics
                axes[row, col].text(0.5, -0.15,
                                   f'range: [{img.min():.1f}, {img.max():.1f}]',
                                   transform=axes[row, col].transAxes,
                                   fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig('scripts/xray_samples.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: scripts/xray_samples.png")
    plt.show()

def analyze_class_separability():
    """Analyze if the three classes are separable"""
    print("\n" + "="*80)
    print("CLASS SEPARABILITY ANALYSIS")
    print("="*80)

    xray_base = Path("x-ray dataset")
    datasets = {
        "cameraman": "cameraman_32by32/cameraman_32by32/data_camera_man_phase_n5.5e6.h5",
        "cell": "cell_32by32/cell_32by32/data_cell_phase_n3e6.h5",
        "mandrill": "mandrill_32by32/mandrill_32by32/data_mandrill_phase_n5.5e6.h5"
    }

    class_stats = {}
    for name, filepath in datasets.items():
        full_path = xray_base / filepath
        with h5py.File(full_path, 'r') as f:
            data = np.abs(f['exchange']['data'][0])  # (1225, 32, 32)

            # Calculate statistics
            stats = {
                'mean_intensity': data.mean(),
                'std_intensity': data.std(),
                'sparsity': np.mean(data == 0),
                'mean_nonzero_pixels': np.mean([np.count_nonzero(img) for img in data]),
                'mean_max_value': np.mean([img.max() for img in data]),
                'mean_energy': np.mean([np.sum(img**2) for img in data])
            }
            class_stats[name] = stats

    # Print comparison
    print("\nPer-class statistics (averaged over all images):")
    print("-" * 80)
    print(f"{'Metric':<25} {'Cameraman':>15} {'Cell':>15} {'Mandrill':>15}")
    print("-" * 80)

    metrics = [
        ('Mean intensity', 'mean_intensity'),
        ('Std intensity', 'std_intensity'),
        ('Sparsity (% zeros)', 'sparsity'),
        ('Avg nonzero pixels', 'mean_nonzero_pixels'),
        ('Avg max value', 'mean_max_value'),
        ('Avg energy', 'mean_energy')
    ]

    for metric_name, metric_key in metrics:
        values = [class_stats[cls][metric_key] for cls in ['cameraman', 'cell', 'mandrill']]
        if 'sparsity' in metric_key:
            print(f"{metric_name:<25} {values[0]*100:>14.2f}% {values[1]*100:>14.2f}% {values[2]*100:>14.2f}%")
        else:
            print(f"{metric_name:<25} {values[0]:>15.4f} {values[1]:>15.4f} {values[2]:>15.4f}")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("-" * 80)
    print("The three classes show different statistical properties,")
    print("suggesting they should be distinguishable by the CNN+HDC classifier.")
    print("="*80)

if __name__ == "__main__":
    visualize_samples()
    analyze_class_separability()
