#!/usr/bin/env python3
"""
Analyze similarity of images within and across the three 32x32 x-ray directories.
Determine if the three directories contain similar types of measurements or completely different content.
"""

import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

def load_dataset(filepath):
    """Load data from h5 file (ignoring _ref files)"""
    with h5py.File(filepath, 'r') as f:
        data = f['exchange']['data'][:]
        # Take magnitude and remove batch dimension
        data = np.abs(data[0])  # Shape: (1225, 32, 32)
    return data

def compute_pairwise_similarity(images1, images2, metric='correlation', num_samples=100):
    """
    Compute similarity between two sets of images

    Args:
        images1, images2: Arrays of shape (N, H, W)
        metric: 'correlation', 'cosine', or 'euclidean'
        num_samples: Number of random pairs to sample for efficiency

    Returns:
        List of similarity scores
    """
    similarities = []

    # Flatten images for comparison
    flat1 = images1.reshape(images1.shape[0], -1)
    flat2 = images2.reshape(images2.shape[0], -1)

    # Sample random pairs
    np.random.seed(42)
    for _ in range(num_samples):
        idx1 = np.random.randint(0, flat1.shape[0])
        idx2 = np.random.randint(0, flat2.shape[0])

        img1 = flat1[idx1]
        img2 = flat2[idx2]

        if metric == 'correlation':
            # Pearson correlation
            corr, _ = pearsonr(img1, img2)
            similarities.append(corr)
        elif metric == 'cosine':
            # Cosine similarity (1 - cosine distance)
            sim = 1 - cosine(img1, img2)
            similarities.append(sim)
        elif metric == 'euclidean':
            # Normalized euclidean distance (convert to similarity)
            dist = np.linalg.norm(img1 - img2)
            # Normalize by image size
            dist_norm = dist / np.sqrt(len(img1))
            # Convert to similarity (inverse)
            sim = 1 / (1 + dist_norm)
            similarities.append(sim)

    return similarities

def analyze_intra_class_variance(data, name):
    """Analyze variance within a single class"""
    print(f"\n{name.upper()} - Intra-class Analysis:")
    print("-" * 70)

    # Sample some images to compare
    num_images = data.shape[0]
    sample_indices = np.linspace(0, num_images-1, 10, dtype=int)

    # Compute pairwise correlations within this class
    correlations = []
    for i in range(len(sample_indices)):
        for j in range(i+1, len(sample_indices)):
            img1 = data[sample_indices[i]].flatten()
            img2 = data[sample_indices[j]].flatten()
            corr, _ = pearsonr(img1, img2)
            correlations.append(corr)

    print(f"  Pairwise correlations (10 samples):")
    print(f"    Mean: {np.mean(correlations):.4f}")
    print(f"    Std:  {np.std(correlations):.4f}")
    print(f"    Min:  {np.min(correlations):.4f}")
    print(f"    Max:  {np.max(correlations):.4f}")

    # Compute mean image
    mean_img = np.mean(data, axis=0)

    # Compute correlation of each image with mean
    mean_correlations = []
    for i in range(num_images):
        img = data[i].flatten()
        mean_flat = mean_img.flatten()
        corr, _ = pearsonr(img, mean_flat)
        mean_correlations.append(corr)

    print(f"  Correlation with class mean image:")
    print(f"    Mean: {np.mean(mean_correlations):.4f}")
    print(f"    Std:  {np.std(mean_correlations):.4f}")

    return correlations, mean_correlations, mean_img

def analyze_inter_class_variance(data1, name1, data2, name2):
    """Analyze variance between two classes"""
    print(f"\n{name1.upper()} vs {name2.upper()} - Inter-class Analysis:")
    print("-" * 70)

    # Compute similarities between random pairs from the two classes
    correlations = compute_pairwise_similarity(data1, data2, metric='correlation', num_samples=200)

    print(f"  Pairwise correlations (200 random pairs):")
    print(f"    Mean: {np.mean(correlations):.4f}")
    print(f"    Std:  {np.std(correlations):.4f}")
    print(f"    Min:  {np.min(correlations):.4f}")
    print(f"    Max:  {np.max(correlations):.4f}")

    return correlations

def visualize_mean_images(mean_images, names):
    """Visualize mean images from each class"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Mean Images from Each Class', fontsize=14)

    for idx, (mean_img, name) in enumerate(zip(mean_images, names)):
        axes[idx].imshow(mean_img, cmap='viridis')
        axes[idx].set_title(f'{name.upper()}')
        axes[idx].axis('off')

        # Add statistics
        axes[idx].text(0.5, -0.1,
                      f'Range: [{mean_img.min():.2f}, {mean_img.max():.2f}]',
                      transform=axes[idx].transAxes,
                      ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('scripts/mean_images_comparison.png', dpi=150, bbox_inches='tight')
    print("\nMean images saved to: scripts/mean_images_comparison.png")

def visualize_similarity_distributions(intra_stats, inter_stats, names):
    """Visualize distribution of similarities"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Intra-class correlations
    ax = axes[0, 0]
    for name, correlations in intra_stats.items():
        ax.hist(correlations, alpha=0.5, bins=30, label=name, density=True)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Density')
    ax.set_title('Intra-class Pairwise Correlations\n(How similar are images WITHIN each class?)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Inter-class correlations
    ax = axes[0, 1]
    colors = ['red', 'green', 'blue']
    for idx, (pair_name, correlations) in enumerate(inter_stats.items()):
        ax.hist(correlations, alpha=0.5, bins=30, label=pair_name,
               color=colors[idx], density=True)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Density')
    ax.set_title('Inter-class Pairwise Correlations\n(How similar are images ACROSS classes?)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Comparison of intra vs inter
    ax = axes[1, 0]
    all_intra = np.concatenate(list(intra_stats.values()))
    all_inter = np.concatenate(list(inter_stats.values()))
    ax.hist(all_intra, alpha=0.6, bins=30, label='Intra-class', color='blue', density=True)
    ax.hist(all_inter, alpha=0.6, bins=30, label='Inter-class', color='red', density=True)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Density')
    ax.set_title('Overall Intra-class vs Inter-class Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(all_intra), color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(np.mean(all_inter), color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "SUMMARY STATISTICS\n" + "="*50 + "\n\n"
    summary_text += "Intra-class (within same class):\n"
    summary_text += f"  Mean correlation: {np.mean(all_intra):.4f}\n"
    summary_text += f"  Std correlation:  {np.std(all_intra):.4f}\n\n"
    summary_text += "Inter-class (across different classes):\n"
    summary_text += f"  Mean correlation: {np.mean(all_inter):.4f}\n"
    summary_text += f"  Std correlation:  {np.std(all_inter):.4f}\n\n"

    separation = np.mean(all_intra) - np.mean(all_inter)
    summary_text += f"Separation: {separation:.4f}\n\n"

    if separation > 0.1:
        summary_text += "✓ Good separation!\n"
        summary_text += "  Classes are distinguishable.\n"
        summary_text += "  Images within a class are more similar\n"
        summary_text += "  to each other than to other classes."
    elif separation > 0.05:
        summary_text += "⚠ Moderate separation\n"
        summary_text += "  Classes may be somewhat distinguishable.\n"
    else:
        summary_text += "✗ Poor separation\n"
        summary_text += "  Classes are not well separated.\n"
        summary_text += "  Images across classes are as similar\n"
        summary_text += "  as images within same class."

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace')

    plt.tight_layout()
    plt.savefig('scripts/similarity_analysis.png', dpi=150, bbox_inches='tight')
    print("Similarity analysis saved to: scripts/similarity_analysis.png")

def main():
    print("\n" + "="*80)
    print("X-RAY DATASET IMAGE SIMILARITY ANALYSIS")
    print("Comparing 32x32 images across three directories")
    print("="*80)

    xray_base = Path("x-ray dataset")

    # Load datasets (non-ref files only)
    datasets = {
        "cameraman": xray_base / "cameraman_32by32/cameraman_32by32/data_camera_man_phase_n5.5e6.h5",
        "cell": xray_base / "cell_32by32/cell_32by32/data_cell_phase_n3e6.h5",
        "mandrill": xray_base / "mandrill_32by32/mandrill_32by32/data_mandrill_phase_n5.5e6.h5"
    }

    print("\nLoading datasets...")
    data = {}
    for name, filepath in datasets.items():
        data[name] = load_dataset(filepath)
        print(f"  {name}: {data[name].shape}")

    # Analyze intra-class variance
    print("\n" + "="*80)
    print("INTRA-CLASS ANALYSIS (Within each directory)")
    print("="*80)

    intra_correlations = {}
    mean_images = {}
    for name in data.keys():
        corr, mean_corr, mean_img = analyze_intra_class_variance(data[name], name)
        intra_correlations[name] = corr
        mean_images[name] = mean_img

    # Analyze inter-class variance
    print("\n" + "="*80)
    print("INTER-CLASS ANALYSIS (Across different directories)")
    print("="*80)

    inter_correlations = {}
    pairs = [
        ("cameraman", "cell"),
        ("cameraman", "mandrill"),
        ("cell", "mandrill")
    ]

    for name1, name2 in pairs:
        corr = analyze_inter_class_variance(data[name1], name1, data[name2], name2)
        inter_correlations[f"{name1} vs {name2}"] = corr

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    visualize_mean_images([mean_images[name] for name in data.keys()],
                          list(data.keys()))

    visualize_similarity_distributions(intra_correlations, inter_correlations,
                                      list(data.keys()))

    # Final conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    all_intra = np.concatenate(list(intra_correlations.values()))
    all_inter = np.concatenate(list(inter_correlations.values()))

    print(f"\nIntra-class correlation (same directory): {np.mean(all_intra):.4f} ± {np.std(all_intra):.4f}")
    print(f"Inter-class correlation (different directories): {np.mean(all_inter):.4f} ± {np.std(all_inter):.4f}")
    print(f"Separation metric: {np.mean(all_intra) - np.mean(all_inter):.4f}")

    if np.mean(all_intra) - np.mean(all_inter) > 0.1:
        print("\n✓ GOOD NEWS: The three directories contain sufficiently different content.")
        print("  Images within each directory are more similar to each other than to")
        print("  images from other directories. This suggests they can be used as")
        print("  separate classes for classification.")
    elif np.mean(all_intra) - np.mean(all_inter) > 0.05:
        print("\n⚠ MODERATE: The three directories show some differences.")
        print("  Classification may be possible but might be challenging.")
    else:
        print("\n✗ WARNING: The three directories contain very similar content.")
        print("  Images are as similar across directories as within directories.")
        print("  They may not work well as separate classes.")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
