# X-Ray Manufacturing Dataset Creation Instructions

## 1. Download Raw Data
The raw data is part of the AGIPD dataset from European XFEL (CXIDB ID 185).
Download the HDF5 files (e.g., `CORR-R0079-AGIPD00-S00000.h5`, `S00001.h5`, etc.) from:
https://cxidb.org/id-185.html

Place the downloaded files in a directory accessible to the scripts (default expected location is `../../manufacturing_xray/`).

## 2. Automated Label Generation
Since the raw data is unlabeled, we generate labels based on pixel intensity (identifying "hits" vs "background").

Run the following script:
```bash
python3 xray_manufacturing/find_optimal_threshold.py --input_dir ../../manufacturing_xray
```
This script:
- Scans all `.h5` files in the input directory.
- Extracts max brightness per image across the entire dataset.
- Uses Otsu's Method to find a global optimal threshold.
- Saves binary labels to `xray_manufacturing/labels.npy`.

## 3. Preprocessing and Bundling
The raw detector data needs to be normalized, resized, and bundled for training.

Run the following script:
```bash
python3 xray_manufacturing/preprocess_xray.py \
    --input_dir ../../manufacturing_xray \
    --input_labels xray_manufacturing/labels.npy \
    --width 32 --height 32 --bits 8
```
This script:
- Scans all files to find global min/max intensity values for consistent normalization.
- Normalizes floating-point intensities to 8-bit integers [0, 255].
- Resizes images to 32x32 pixels using bilinear interpolation.
- Bundles processed images and labels into the final training file: `xray_manufacturing/manufacturing.h5`.

## 4. Verification
The `manufacturing.h5` file is now ready for use with `train_hdc.py` using the `--dataset manufacturing` flag.
Use `--samples_per_class` to limit the number of training samples if the dataset is too large.
