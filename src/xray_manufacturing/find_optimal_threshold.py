import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import glob
import argparse

def find_optimal_threshold(input_dir):
    print(f"Scanning {input_dir} for .h5 files...")
    
    if not os.path.exists(input_dir):
        print("Input directory not found.")
        return

    h5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    if not h5_files:
        print("No .h5 files found.")
        return
        
    print(f"Found {len(h5_files)} files: {[os.path.basename(f) for f in h5_files]}")

    # --- Step 1: Extract Max Values ---
    npy_path = "xray_manufacturing/all_max_values.npy"
    if os.path.exists(npy_path):
        print(f"Loading existing max values from {npy_path}...")
        max_values = np.load(npy_path)
        print(f"Loaded {len(max_values)} values.")
    else:
        # We will force re-extraction to ensure we cover all files
        print("Extracting max values from all files...")
        max_values = []
        
        for filepath in h5_files:
            print(f"Processing {filepath}...")
            try:
                with h5py.File(filepath, 'r') as f:
                    dataset_path = 'INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/data'
                    if dataset_path not in f:
                        print(f"Dataset {dataset_path} not found in {filepath}.")
                        continue

                    dset = f[dataset_path]
                    num_images = dset.shape[0]
                    print(f"  Images: {num_images}")
                    
                    batch_size = 2000
                    for i in tqdm(range(0, num_images, batch_size), desc="  Extracting"):
                        end_idx = min(i + batch_size, num_images)
                        # Read batch
                        images = dset[i:end_idx]
                        # Get max value per image (axis 1 and 2 are height and width)
                        batch_maxs = np.max(images, axis=(1, 2))
                        max_values.extend(batch_maxs)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                return
                
        max_values = np.array(max_values)
        np.save(npy_path, max_values)
        print(f"Saved {len(max_values)} max values to {npy_path}")

    # --- Step 2: Analyze Distribution ---
    print("\n--- Distribution Analysis ---")
    print(f"Global Min Max-Brightness: {np.min(max_values):.2f}")
    print(f"Global Max Max-Brightness: {np.max(max_values):.2f}")
    
    # 1. K-Means Clustering (k=2)
    # Reshape for sklearn (n_samples, n_features)
    X = max_values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    centers = kmeans.cluster_centers_.flatten()
    
    # Identify which cluster is "background" (lower center) and "signal" (higher center)
    sorted_idx = np.argsort(centers)
    background_center = centers[sorted_idx[0]]
    signal_center = centers[sorted_idx[1]]
    
    # The threshold is the decision boundary between the two centers
    kmeans_threshold = (background_center + signal_center) / 2
    
    print(f"\n[K-Means Results]")
    print(f"  Background Cluster Center: {background_center:.2f}")
    print(f"  Signal Cluster Center:     {signal_center:.2f}")
    print(f"  Calculated Threshold:      {kmeans_threshold:.2f}")
    
    # 2. Otsu's Method (Manual implementation for 1D array)
    hist, bin_edges = np.histogram(max_values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    total_weight = len(max_values)
    current_weight = 0
    sum_bg = 0
    sum_total = np.sum(bin_centers * hist)
    
    best_otsu_thresh = 0
    max_variance = 0
    
    for i in range(len(hist)):
        weight_bg = current_weight + hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total_weight - weight_bg
        if weight_fg == 0:
            break
            
        sum_bg += bin_centers[i] * hist[i]
        
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        
        # Inter-class variance
        variance = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
        
        if variance > max_variance:
            max_variance = variance
            best_otsu_thresh = bin_edges[i+1] # approximate
            
        current_weight = weight_bg

    print(f"\n[Otsu's Method Results]")
    print(f"  Calculated Threshold:      {best_otsu_thresh:.2f}")

    # --- Step 3: Visualization ---
    plt.figure(figsize=(10, 6))
    plt.hist(max_values, bins=100, color='skyblue', edgecolor='black', log=True)
    plt.axvline(kmeans_threshold, color='red', linestyle='dashed', linewidth=2, label=f'K-Means Threshold ({kmeans_threshold:.1f})')
    plt.axvline(best_otsu_thresh, color='green', linestyle='dashed', linewidth=2, label=f'Otsu Threshold ({best_otsu_thresh:.1f})')
    plt.title('Distribution of Maximum Pixel Brightness per Image')
    plt.xlabel('Max Pixel Value')
    plt.ylabel('Count (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('xray_manufacturing/brightness_histogram.png')
    print("\nSaved histogram to xray_manufacturing/brightness_histogram.png")

    # --- Step 4: Final Selection & Label Generation ---
    # Otsu is often more precise for determining the exact split in the valley
    final_threshold = best_otsu_thresh
    print(f"\nUsing Otsu Threshold: {final_threshold:.2f}")
    
    final_labels = (max_values > final_threshold).astype(int)
    hit_count = np.sum(final_labels)
    hit_rate = (hit_count / len(final_labels)) * 100.0
    
    print(f"\n--- Final Statistics ---")
    print(f"Total Images: {len(final_labels)}")
    print(f"Hits Detected: {hit_count}")
    print(f"Misses: {len(final_labels) - hit_count}")
    print(f"Hit Rate: {hit_rate:.2f}%")
    
    np.savetxt("xray_manufacturing/labels.txt", final_labels, fmt='%d')
    np.save("xray_manufacturing/labels.npy", final_labels)
    print("Labels saved to xray_manufacturing/labels.txt and xray_manufacturing/labels.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal threshold for X-ray images")
    parser.add_argument("--input_dir", default="../../manufacturing_xray", help="Directory containing .h5 files")
    args = parser.parse_args()
    
    find_optimal_threshold(args.input_dir)