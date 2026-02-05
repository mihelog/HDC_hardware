#!/usr/bin/env python3
"""
Explore alternative ways to create labels from the x-ray dataset.
Since the three directories are too similar, we need to find other ways to create classes.
"""

import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def load_dataset(filepath):
    """Load data from h5 file"""
    with h5py.File(filepath, 'r') as f:
        data = f['exchange']['data'][:]
        data = np.abs(data[0])  # Shape: (1225, 32, 32)
    return data

def approach1_spatial_regions(data, name):
    """
    Approach 1: Divide images based on spatial properties
    (e.g., center brightness, corner brightness, etc.)
    """
    print(f"\n{name.upper()} - Spatial Region Analysis:")
    print("-" * 70)

    n_images = data.shape[0]

    # Compute various spatial features
    center_brightness = []
    corner_brightness = []
    edge_brightness = []
    total_brightness = []

    for i in range(n_images):
        img = data[i]

        # Center region (middle 16x16)
        center = img[8:24, 8:24]
        center_brightness.append(np.mean(center))

        # Corner regions (8x8 each)
        corners = [img[0:8, 0:8], img[0:8, 24:32], img[24:32, 0:8], img[24:32, 24:32]]
        corner_brightness.append(np.mean([np.mean(c) for c in corners]))

        # Edge regions (exclude center)
        edges = np.concatenate([img[0:8, :].flatten(), img[24:32, :].flatten(),
                               img[:, 0:8].flatten(), img[:, 24:32].flatten()])
        edge_brightness.append(np.mean(edges))

        # Total brightness
        total_brightness.append(np.mean(img))

    center_brightness = np.array(center_brightness)
    corner_brightness = np.array(corner_brightness)
    edge_brightness = np.array(edge_brightness)
    total_brightness = np.array(total_brightness)

    # Compute ratio of center to edge
    center_to_edge_ratio = center_brightness / (edge_brightness + 1e-10)

    print(f"  Center brightness:     {np.mean(center_brightness):.4f} ± {np.std(center_brightness):.4f}")
    print(f"  Corner brightness:     {np.mean(corner_brightness):.4f} ± {np.std(corner_brightness):.4f}")
    print(f"  Edge brightness:       {np.mean(edge_brightness):.4f} ± {np.std(edge_brightness):.4f}")
    print(f"  Center/Edge ratio:     {np.mean(center_to_edge_ratio):.4f} ± {np.std(center_to_edge_ratio):.4f}")

    return {
        'center': center_brightness,
        'corner': corner_brightness,
        'edge': edge_brightness,
        'total': total_brightness,
        'ratio': center_to_edge_ratio
    }

def approach2_temporal_sequence(data, name, num_classes=10):
    """
    Approach 2: Use temporal/sequential information
    Images 0-122 = class 0, 123-245 = class 1, etc.
    Assumes images are sequential measurements
    """
    print(f"\n{name.upper()} - Sequential Division:")
    print("-" * 70)

    n_images = data.shape[0]
    images_per_class = n_images // num_classes

    print(f"  Total images: {n_images}")
    print(f"  Target classes: {num_classes}")
    print(f"  Images per class: ~{images_per_class}")

    # Create labels based on sequence
    labels = np.zeros(n_images, dtype=int)
    for i in range(n_images):
        labels[i] = min(i // images_per_class, num_classes - 1)

    print(f"  Class distribution:")
    for c in range(num_classes):
        count = np.sum(labels == c)
        print(f"    Class {c}: {count} images")

    # Analyze if sequential images are more similar
    sequential_corr = []
    for i in range(0, n_images - 10, 50):  # Sample every 50th image
        img1 = data[i].flatten()
        img2 = data[i + 1].flatten()  # Next sequential image
        corr, _ = pearsonr(img1, img2)
        sequential_corr.append(corr)

    random_corr = []
    np.random.seed(42)
    for _ in range(len(sequential_corr)):
        i = np.random.randint(0, n_images)
        j = np.random.randint(0, n_images)
        if i != j:
            img1 = data[i].flatten()
            img2 = data[j].flatten()
            corr, _ = pearsonr(img1, img2)
            random_corr.append(corr)

    print(f"  Sequential pair correlation: {np.mean(sequential_corr):.4f}")
    print(f"  Random pair correlation:     {np.mean(random_corr):.4f}")

    if np.mean(sequential_corr) > np.mean(random_corr) + 0.01:
        print(f"  → Sequential images ARE more similar (good for temporal classes)")
    else:
        print(f"  → Sequential images NOT more similar (temporal division may be arbitrary)")

    return labels

def approach3_clustering_features(all_data, all_names, num_classes=10):
    """
    Approach 3: K-means clustering on image features
    """
    print(f"\nCLUSTERING-BASED LABELING:")
    print("-" * 70)

    # Combine all data
    combined_data = np.concatenate([all_data[name] for name in all_names], axis=0)
    print(f"  Total images: {combined_data.shape[0]}")

    # Flatten images
    flat_data = combined_data.reshape(combined_data.shape[0], -1)

    # Apply PCA for dimensionality reduction
    print(f"  Applying PCA...")
    pca = PCA(n_components=50)
    pca_features = pca.fit_transform(flat_data)
    print(f"  Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    # K-means clustering
    print(f"  Running K-means with {num_classes} clusters...")
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_features)

    print(f"  Cluster distribution:")
    for c in range(num_classes):
        count = np.sum(labels == c)
        print(f"    Cluster {c}: {count} images ({count/len(labels)*100:.1f}%)")

    # Analyze cluster quality (inertia)
    print(f"  Inertia: {kmeans.inertia_:.2f}")

    # Try to see if clusters correspond to original directories
    print(f"\n  Cluster composition by original directory:")
    offset = 0
    for name in all_names:
        n = all_data[name].shape[0]
        dir_labels = labels[offset:offset+n]

        print(f"  {name.upper()}:")
        for c in range(num_classes):
            count = np.sum(dir_labels == c)
            if count > 0:
                print(f"    Cluster {c}: {count} images ({count/n*100:.1f}%)")

        offset += n

    return labels, pca_features

def visualize_clustering(pca_features, labels, all_names, all_data):
    """Visualize clustering results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: PCA visualization colored by cluster
    ax = axes[0]
    scatter = ax.scatter(pca_features[:, 0], pca_features[:, 1],
                        c=labels, cmap='tab10', alpha=0.6, s=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('K-means Clustering (colored by cluster)')
    plt.colorbar(scatter, ax=ax)

    # Plot 2: PCA visualization colored by original directory
    ax = axes[1]
    original_labels = []
    offset = 0
    for idx, name in enumerate(all_names):
        n = all_data[name].shape[0]
        original_labels.extend([idx] * n)
    original_labels = np.array(original_labels)

    scatter = ax.scatter(pca_features[:, 0], pca_features[:, 1],
                        c=original_labels, cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Original Directory (colored by directory)')
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['cameraman', 'cell', 'mandrill'])

    plt.tight_layout()
    plt.savefig('scripts/clustering_visualization.png', dpi=150, bbox_inches='tight')
    print("\n  Clustering visualization saved to: scripts/clustering_visualization.png")

def main():
    print("\n" + "="*80)
    print("ALTERNATIVE LABELING STRATEGIES FOR X-RAY DATASET")
    print("="*80)

    xray_base = Path("x-ray dataset")

    # Load datasets
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

    # Approach 1: Spatial features
    print("\n" + "="*80)
    print("APPROACH 1: SPATIAL REGION FEATURES")
    print("="*80)

    spatial_features = {}
    for name in data.keys():
        spatial_features[name] = approach1_spatial_regions(data[name], name)

    # Approach 2: Sequential/temporal division
    print("\n" + "="*80)
    print("APPROACH 2: SEQUENTIAL/TEMPORAL DIVISION")
    print("="*80)

    for name in data.keys():
        labels = approach2_temporal_sequence(data[name], name, num_classes=10)

    # Approach 3: Clustering
    print("\n" + "="*80)
    print("APPROACH 3: K-MEANS CLUSTERING")
    print("="*80)

    cluster_labels, pca_features = approach3_clustering_features(
        data, list(data.keys()), num_classes=10
    )

    visualize_clustering(pca_features, cluster_labels, list(data.keys()), data)

    # Final recommendation
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("""
Based on the analysis:

1. ✗ SIMPLE 3-CLASS APPROACH (original directories as classes)
   - NOT RECOMMENDED: Directories are too similar (correlation diff = 0.0007)
   - Images across directories are as similar as within directories
   - Would likely result in poor classification accuracy

2. ✓ K-MEANS CLUSTERING (Approach 3)
   - RECOMMENDED: Creates meaningful clusters from data
   - Configurable number of classes (5, 10, 15, etc.)
   - Clusters based on actual image features, not arbitrary divisions
   - Can visualize cluster quality

3. ? SEQUENTIAL DIVISION (Approach 2)
   - DEPENDS: Only if sequential images show different properties
   - Check the "Sequential pair correlation" vs "Random pair correlation"
   - If sequential correlation is significantly higher, this could work

4. ✓ AUTOENCODER + CLUSTERING
   - BEST OPTION: Most principled approach
   - Learn compressed representation with autoencoder
   - Cluster in learned feature space
   - Should give better clusters than raw pixel clustering

NEXT STEPS:
  1. Implement simple autoencoder for feature learning
  2. Use encoder features for clustering into N classes
  3. Validate clusters are separable
  4. Use clustered labels for CNN+HDC training
    """)

if __name__ == "__main__":
    main()
