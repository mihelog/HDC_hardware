import h5py
import numpy as np
import os
from tqdm import tqdm

def generate_labels(filepath, threshold=1200.0, output_txt="labels.txt", output_npy="labels.npy"):
    print(f"Generating labels for {filepath}")
    print(f"Threshold for hit (bright spot): max_pixel_value > {threshold}")
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    try:
        with h5py.File(filepath, 'r') as f:
            dataset_path = 'INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/data'
            if dataset_path not in f:
                print(f"Dataset {dataset_path} not found.")
                return

            dset = f[dataset_path]
            num_images = dset.shape[0]
            print(f"Total images: {num_images}")
            
            labels = []
            batch_size = 1000
            
            # Process in batches for efficiency
            for i in tqdm(range(0, num_images, batch_size)):
                end_idx = min(i + batch_size, num_images)
                batch = dset[i:end_idx]
                
                # Calculate max per image in batch
                # axis=(1,2) reduces (B, H, W) -> (B,)
                batch_maxs = np.max(batch, axis=(1, 2))
                
                # Generate labels
                batch_labels = (batch_maxs > threshold).astype(int)
                labels.extend(batch_labels)
            
            labels = np.array(labels)
            hit_count = np.sum(labels)
            hit_rate = (hit_count / num_images) * 100.0
            
            print(f"\nProcessing complete.")
            print(f"Total Hits: {hit_count}")
            print(f"Total Misses: {num_images - hit_count}")
            print(f"Hit Rate: {hit_rate:.2f}%")
            
            # Save to text file
            np.savetxt(output_txt, labels, fmt='%d')
            print(f"Labels saved to {output_txt}")
            
            # Save to numpy file
            np.save(output_npy, labels)
            print(f"Labels saved to {output_npy}")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    generate_labels("../../manufacturing_xray/CORR-R0079-AGIPD00-S00000.h5")
