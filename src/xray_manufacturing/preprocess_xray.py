import h5py
import numpy as np
from PIL import Image
import argparse
import os
import glob
from tqdm import tqdm

def preprocess(args):
    print(f"Preprocessing X-ray dataset from directory...")
    print(f"Input Directory: {args.input_dir}")
    print(f"Input Labels: {args.input_labels}")
    print(f"Target Size: {args.width}x{args.height}")
    print(f"Bit Width: {args.bits}")
    
    if not os.path.exists(args.input_dir):
        print("Error: Input directory not found.")
        return
    if not os.path.exists(args.input_labels):
        print("Error: Input labels file not found.")
        return

    h5_files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))
    if not h5_files:
        print("Error: No .h5 files found in input directory.")
        return
        
    print(f"Found {len(h5_files)} files: {[os.path.basename(f) for f in h5_files]}")

    # Load labels
    labels = np.load(args.input_labels)
    total_labels = len(labels)
    print(f"Loaded {total_labels} labels.")

    # First pass: Determine global min/max for normalization
    global_min = float('inf')
    global_max = float('-inf')
    total_images_found = 0
    
    print("Scanning all files for global min/max...")
    batch_size = 2000
    
    for filepath in h5_files:
        print(f"  Scanning {os.path.basename(filepath)}...")
        try:
            with h5py.File(filepath, 'r') as f:
                dataset_path = 'INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/data'
                if dataset_path not in f:
                    print(f"    Warning: Dataset not found in {filepath}")
                    continue
                    
                dset = f[dataset_path]
                num_in_file = dset.shape[0]
                total_images_found += num_in_file
                
                for i in range(0, num_in_file, batch_size):
                    end_idx = min(i + batch_size, num_in_file)
                    batch = dset[i:end_idx]
                    global_min = min(global_min, np.min(batch))
                    global_max = max(global_max, np.max(batch))
        except Exception as e:
            print(f"    Error reading {filepath}: {e}")
            return

    print(f"Total images found: {total_images_found}")
    print(f"Global Min: {global_min}")
    print(f"Global Max: {global_max}")
    
    if total_images_found != total_labels:
        print(f"Warning: Mismatch between images ({total_images_found}) and labels ({total_labels}).")
        # We will truncate to the minimum later
    
    # Prepare output dataset
    # We'll use the minimum count to be safe
    num_output_images = min(total_images_found, total_labels)
    processed_images = np.zeros((num_output_images, args.height, args.width), dtype=np.uint8)
    
    # Second pass: Process and resize
    print("Resizing and quantizing...")
    current_idx = 0
    
    for filepath in h5_files:
        if current_idx >= num_output_images:
            break
            
        print(f"  Processing {os.path.basename(filepath)}...")
        with h5py.File(filepath, 'r') as f:
            dataset_path = 'INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/data'
            dset = f[dataset_path]
            num_in_file = dset.shape[0]
            
            for i in tqdm(range(0, num_in_file, batch_size), desc=os.path.basename(filepath)):
                if current_idx >= num_output_images:
                    break
                    
                end_idx = min(i + batch_size, num_in_file)
                batch = dset[i:end_idx]
                
                # Check if we need to truncate this batch to fit remaining slots
                if current_idx + len(batch) > num_output_images:
                    batch = batch[:num_output_images - current_idx]
                
                # Normalize
                batch_norm = (batch - global_min) / (global_max - global_min)
                
                # Scale
                max_int_val = (1 << args.bits) - 1
                batch_scaled = batch_norm * max_int_val
                batch_uint = np.clip(batch_scaled, 0, max_int_val).astype(np.uint8)
                
                # Resize
                for j in range(len(batch_uint)):
                    img = Image.fromarray(batch_uint[j])
                    img_resized = img.resize((args.width, args.height), Image.BILINEAR)
                    processed_images[current_idx] = np.array(img_resized)
                    current_idx += 1

    # Truncate labels if necessary
    final_labels = labels[:num_output_images]

    # Save to new HDF5
    output_path = os.path.join(os.path.dirname(args.output), "manufacturing.h5")
    print(f"Saving {num_output_images} images to {output_path}...")
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('images', data=processed_images)
        f_out.create_dataset('labels', data=final_labels)
        
    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess X-ray manufacturing dataset")
    parser.add_argument("--input_dir", default="../../manufacturing_xray", help="Directory containing raw HDF5 files")
    parser.add_argument("--input_labels", default="xray_manufacturing/labels.npy", help="Path to labels .npy")
    parser.add_argument("--output", default="xray_manufacturing/manufacturing.h5", help="Output path")
    parser.add_argument("--width", type=int, default=32, help="Target width")
    parser.add_argument("--height", type=int, default=32, help="Target height")
    parser.add_argument("--bits", type=int, default=8, help="Bit width per pixel")
    
    args = parser.parse_args()
    preprocess(args)
