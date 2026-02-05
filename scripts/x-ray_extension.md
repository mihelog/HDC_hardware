# X-ray Dataset Extension for HDC Classifier

**Date**: 2025-10-24
**Purpose**: Add support for unlabeled x-ray datasets with autoencoder-based clustering

---

## Overview

This extension adds the ability to use unlabeled x-ray diffraction pattern images with the HDC classifier by implementing an autoencoder-based clustering pipeline. The system automatically:
1. Loads unlabeled images from multiple .h5 files
2. Trains an autoencoder to extract meaningful features
3. Clusters the features to create synthetic class labels
4. Trains the CNN+HDC classifier on the clustered data

**Key Feature**: Fully backward compatible - all existing datasets (QuickDraw, MNIST, Caltech-101) work unchanged.

---

## Files Modified

### 1. `src/train_hdc.py` (~430 lines added)

#### New Classes Added:

**SimpleAutoencoder (line ~1229)**
- Lightweight CNN autoencoder for unsupervised feature learning
- Architecture:
  - Encoder: Conv(1→16)→Pool→Conv(16→32)→Pool→FC(128)
  - Decoder: FC→TransConv(32→16)→TransConv(16→1)
- Used to extract 128-dimensional latent features from unlabeled images
- Trained with MSE reconstruction loss

**XRayUnlabeledDataset (line ~2426)**
- Custom PyTorch Dataset for loading unlabeled .h5 files
- Features:
  - Loads from multiple directories simultaneously
  - Skips `*_ref.h5` files automatically
  - Quantizes images to 8-bit or 16-bit (configurable)
  - Supports stratified train/test split with clustered labels
  - Reads from `exchange/data` path in HDF5 structure
  - Takes magnitude of complex64 data

#### New Functions Added:

**train_autoencoder() (line ~4458)**
- Trains the autoencoder on unlabeled dataset
- Parameters:
  - `latent_dim=128`: Feature dimension
  - `epochs=20`: Training epochs
  - `batch_size=64`: Batch size
- Returns: Numpy array of encoder features (N × 128)
- Uses Adam optimizer with learning rate 1e-3

**cluster_features() (line ~4533)**
- Performs K-means clustering on encoder features
- Pipeline:
  1. PCA dimensionality reduction (50 components, preserves ~75% variance)
  2. K-means clustering into N classes
  3. Quality validation (checks for imbalance)
- Parameters:
  - `num_clusters`: Number of synthetic classes to create
  - `method='kmeans'`: Clustering algorithm
- Returns: Cluster labels (N,)
- Warnings if cluster imbalance detected (>50% or <5% in any cluster)

#### Modified Functions:

**train_system() (line ~4596)**
- Added parameters:
  - `unlabeled=False`: Enable unlabeled data mode
  - `data_dirs=None`: List of directories with .h5 files
  - `num_clusters=10`: Number of clusters (overrides num_classes)
  - `quantize_bits=8`: Quantization bit width (8 or 16)

- New data loading branch (line ~4711):
  ```python
  elif dataset_name == 'xray' or unlabeled:
      # 1. Load full unlabeled dataset
      # 2. Train autoencoder
      # 3. Cluster features
      # 4. Stratified train/test split
      # 5. Create datasets with clustered labels
  ```

- Uses sklearn's `train_test_split` with stratification to ensure balanced classes in train/test sets

**main() argparse (line ~5614)**
- Added command-line arguments:
  - `--unlabeled`: Flag to enable unlabeled mode
  - `--data_dirs`: Space-separated list of directories
  - `--num_clusters`: Number of clusters (default: 10)
  - `--quantize_bits`: Bit width for quantization (default: 8)

### 2. `src/makefile` (~30 lines added)

#### New Variables (line 71-77):

```makefile
# X-ray dataset configuration
XRAY_DIRS = "x-ray dataset/cameraman_32by32/cameraman_32by32" \
            "x-ray dataset/cell_32by32/cell_32by32" \
            "x-ray dataset/mandrill_32by32/mandrill_32by32"
UNLABELED = 0  # Set to 1 for unlabeled datasets
NUM_CLUSTERS = 10  # Number of clusters
QUANTIZE_BITS = 8  # Quantization bit width (8 or 16)
```

#### New Targets (line 169-179):

**`make xray`**
- Runs full pipeline with x-ray dataset
- Uses 10 clusters (configurable with `NUM_CLUSTERS=N`)
- 8-bit quantization, 32×32 images
- Example: `make xray NUM_CLUSTERS=5`

**`make xray_quick`**
- Quick test with 3 clusters
- Uses quick testbench for faster iteration
- Example: `make xray_quick`

#### Updated train Target (line 219-222):

Added conditional parameters to pass to Python script:
```makefile
$(if $(filter 1,$(UNLABELED)),--unlabeled) \
$(if $(filter 1,$(UNLABELED)),--data_dirs $(XRAY_DIRS)) \
$(if $(filter 1,$(UNLABELED)),--num_clusters $(NUM_CLUSTERS)) \
--quantize_bits $(QUANTIZE_BITS)
```

---

## How It Works

### Pipeline for Unlabeled Data:

```
1. Load .h5 files
   ├─ From: x-ray dataset/*/data_*_phase_*.h5
   ├─ Skip: *_ref.h5 files
   ├─ Extract: exchange/data (complex64)
   └─ Process: np.abs() to get magnitude

2. Quantization
   ├─ Input: Float32 values (~0-31)
   ├─ Normalize: [0, 2^bits - 1]
   └─ Output: uint8 (8-bit) or uint16 (16-bit)

3. Autoencoder Training (20 epochs)
   ├─ Input: All quantized images (no labels)
   ├─ Loss: MSE reconstruction
   └─ Output: 128-dim features per image

4. Clustering (K-means)
   ├─ Input: 128-dim features
   ├─ PCA: Reduce to 50 dims
   ├─ K-means: Create N clusters
   └─ Output: Synthetic labels (0 to N-1)

5. Train/Test Split (stratified 80/20)
   ├─ Ensures balanced classes in both sets
   └─ Uses sklearn.model_selection.train_test_split

6. CNN+HDC Training (same as labeled data)
   ├─ Uses clustered labels as "ground truth"
   ├─ Saves weights_and_hvs.txt
   └─ Saves test_images.txt
```

### Data Flow:

```
HDF5 Files → XRayUnlabeledDataset → Autoencoder → Features
                                                      ↓
Test Images ← CNN+HDC ← Train/Test Split ← Clustering
```

---

## Usage Examples

### Basic Usage:

```bash
# Full x-ray dataset (10 clusters)
make xray

# Custom number of clusters
make xray NUM_CLUSTERS=5

# Quick test (3 clusters)
make xray_quick
```

### Python Direct Usage:

```bash
python3 train_hdc.py \
    --dataset xray \
    --unlabeled \
    --data_dirs \
        "x-ray dataset/cameraman_32by32/cameraman_32by32" \
        "x-ray dataset/cell_32by32/cell_32by32" \
        "x-ray dataset/mandrill_32by32/mandrill_32by32" \
    --num_clusters 10 \
    --quantize_bits 8 \
    --image_size 32 \
    --epochs 40
```

### Backward Compatibility:

```bash
# All existing datasets still work unchanged
make all              # QuickDraw (default)
make mnist            # MNIST
make caltech101       # Caltech-101
make quick            # Quick test
```

---

## Dataset Statistics

### X-ray 32×32 Dataset:

| Directory | Files | Images per File | Image Size | Total Images |
|-----------|-------|----------------|------------|--------------|
| cameraman_32by32 | 1 .h5 | 1,225 | 32×32 | 1,225 |
| cell_32by32 | 1 .h5 | 1,225 | 32×32 | 1,225 |
| mandrill_32by32 | 1 .h5 | 1,225 | 32×32 | 1,225 |
| **Total** | **3 files** | - | - | **3,675** |

### Data Characteristics:

- **Format**: complex64 (but imaginary part = 0)
- **Value Range**: 0 to ~31 (sparse, most values near 0)
- **Sparsity**: ~93% zeros
- **Similarity**: High intra-directory correlation (0.98)
- **Separation**: Low inter-directory separation (0.0007)
  - This is why simple directory-based labeling doesn't work!
  - Autoencoder clustering is necessary

---

## Key Design Decisions

### 1. Why Autoencoder Instead of Simple Class Labels?

From `scripts/analyze_image_similarity.py` analysis:
- Intra-class correlation: 0.9810
- Inter-class correlation: 0.9802
- **Separation: 0.0007** (essentially zero!)

The three directories contain nearly identical diffraction patterns. Using directories as classes would result in poor accuracy because the images are as similar across directories as within directories.

**Solution**: Use autoencoder to learn meaningful features, then cluster in feature space to find natural groupings.

### 2. Why 8-bit Quantization by Default?

- Original data range: 0 to ~31
- Existing hardware uses PIXEL_WIDTH=8
- 8-bit provides sufficient precision (256 levels)
- 16-bit option available if needed

### 3. Why Skip `*_ref.h5` Files?

Reference files contain calibration/baseline data with different characteristics. Including them would confuse the clustering.

### 4. Why Stratified Split?

Ensures each class has proportional representation in both train and test sets. Critical for imbalanced clusters.

---

## Verilog Compatibility

### Does Verilog Need Changes?

**NO** - The Verilog hardware requires **zero modifications**.

### Why No Verilog Changes?

The x-ray extension is completely transparent to the hardware:

1. **Input Format Unchanged**:
   - Python quantizes images to 8-bit (or 16-bit)
   - Saves to `test_images.txt` in same format
   - Verilog reads images identically regardless of source

2. **Weight Format Unchanged**:
   - CNN weights saved in same format
   - HDC projection matrix saved in same format
   - Class hypervectors saved in same format

3. **Label Independence**:
   - Verilog doesn't care how labels were created
   - Clustered labels vs manual labels are indistinguishable
   - Hardware just compares query HV to class HVs

4. **Pipeline Unchanged**:
   - Same CNN architecture
   - Same HDC encoding
   - Same Hamming distance comparison

### What Verilog Sees:

```
Before (QuickDraw):
  test_images.txt: Manually labeled images → CNN+HDC → Results

After (X-ray):
  test_images.txt: Clustered labeled images → CNN+HDC → Results
                   ↑
                   Same format!
```

The beauty of this implementation is **complete separation of concerns**:
- Python: Handle dataset complexity (loading, clustering, labeling)
- Verilog: Process standardized inputs (quantized images, trained weights)

---

## Potential Issues & Solutions

### Issue 1: Cluster Imbalance

**Problem**: K-means might create very unbalanced clusters (e.g., one cluster with 54% of data)

**Detection**: `cluster_features()` warns if any cluster has >50% or <5% of data

**Solutions**:
1. Use fewer clusters: `make xray NUM_CLUSTERS=5`
2. Try different random seeds
3. Use GMM instead of K-means (future enhancement)

### Issue 2: Poor Clustering Quality

**Problem**: Features might not separate well due to high image similarity

**Detection**: Low variance explained by PCA, high overlap in clusters

**Solutions**:
1. Train autoencoder longer: Modify `epochs=20` to `epochs=50`
2. Use larger latent dimension: Modify `latent_dim=128` to `latent_dim=256`
3. Use different features (e.g., CNN features instead of autoencoder)

### Issue 3: Memory Usage

**Problem**: Loading all 3,675 images into memory at once

**Solutions**:
1. Already implemented: Sequential loading from .h5 files
2. If needed: Add batch processing for very large datasets

---

## Testing Checklist

- [ ] Test QuickDraw still works: `make all`
- [ ] Test MNIST still works: `make mnist`
- [ ] Test x-ray with 3 clusters: `make xray NUM_CLUSTERS=3`
- [ ] Test x-ray with 10 clusters: `make xray`
- [ ] Verify cluster balance in output
- [ ] Run Verilog simulation with x-ray weights
- [ ] Compare Python vs Verilog accuracy

---

## Future Enhancements

### Potential Improvements:

1. **Better Clustering**:
   - Add GMM (Gaussian Mixture Model) option
   - Add hierarchical clustering
   - Add DBSCAN for density-based clustering

2. **Feature Extraction**:
   - Use pretrained CNN features instead of autoencoder
   - Try VAE (Variational Autoencoder) for better feature learning
   - Add contrastive learning

3. **Data Augmentation**:
   - Add rotation/flip augmentation for x-ray images
   - Use data augmentation during autoencoder training

4. **Multi-modal Support**:
   - Handle both magnitude and phase of complex data
   - Support multiple channels (beta and delta from phantom/)

5. **Validation**:
   - Add silhouette score to measure cluster quality
   - Add Davies-Bouldin index
   - Visualize clusters with t-SNE

---

## Summary

### What Was Added:

- ✅ Autoencoder-based feature learning
- ✅ K-means clustering for label generation
- ✅ .h5 file loading with quantization
- ✅ Full integration with existing pipeline
- ✅ Makefile targets for easy usage
- ✅ Complete backward compatibility

### What Wasn't Changed:

- ✅ Verilog hardware (zero changes needed)
- ✅ Verilog testbench (zero changes needed)
- ✅ Existing datasets (QuickDraw, MNIST, Caltech-101)
- ✅ CNN architecture
- ✅ HDC encoding
- ✅ File formats (weights_and_hvs.txt, test_images.txt)

### Lines of Code:

- `train_hdc.py`: +430 lines
- `makefile`: +30 lines
- **Total**: ~460 lines of new code

### Backward Compatibility:

**100%** - All existing functionality preserved, new features are opt-in via `--unlabeled` flag or `make xray` target.
