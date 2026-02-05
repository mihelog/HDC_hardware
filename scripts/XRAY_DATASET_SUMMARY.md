# X-ray Dataset Analysis Summary

## Question 1: Does the new dataset contain labels?

**NO** - The X-ray dataset does NOT contain explicit class labels.

### What the dataset contains:
- **32x32 datasets**: Each .h5 file contains 1,225 diffraction pattern images of size 32x32
- **Three sample types**: cameraman, cell, mandrill
- **Total images**: 3,675 (1,225 per sample type)
- **Data format**: Complex64, but imaginary part is zero (effectively real-valued)
- **Value range**: 0 to ~31 (suitable for 8-bit normalization to 0-255)

### File structure:
```
x-ray dataset/
├── cameraman_32by32/
│   └── data_camera_man_phase_n5.5e6.h5    (1, 1225, 32, 32)
├── cell_32by32/
│   └── data_cell_phase_n3e6.h5            (1, 1225, 32, 32)
└── mandrill_32by32/
    └── data_mandrill_phase_n5.5e6.h5      (1, 1225, 32, 32)
```

## Question 2: Can we use this dataset instead of QuickDraw?

**YES** - With simple modifications to create labels.

### Recommended Approach (No autoencoder needed):

**Simple 3-Class Classification:**
- Treat each sample type as a separate class
- Class 0: cameraman (1,225 images)
- Class 1: cell (1,225 images)
- Class 2: mandrill (1,225 images)

**Advantages:**
- ✅ Simple and direct implementation
- ✅ No need for autoencoders or complex preprocessing
- ✅ Same image size (32x32) as current system
- ✅ Balanced dataset (equal samples per class)
- ✅ Uses actual diffraction data (what hardware would see)
- ✅ Classes show different statistical properties (separable)

**Train/Test Split:**
- Training: 980 images per class (2,940 total)
- Testing: 245 images per class (735 total)
- Ratio: 80% train / 20% test

## Question 3: If no labels, can we use an autoencoder?

**YES** - If you want more than 3 classes, you can use an autoencoder.

### Autoencoder Approach (for configurable N classes):

**How it would work:**
1. Train autoencoder on all 3,675 diffraction images (unsupervised)
2. Extract encoder features for each image
3. Apply clustering (K-means, GMM, etc.) on encoder features
4. Create N synthetic classes (configurable: 5, 10, 15, etc.)
5. Use clustered labels for supervised CNN+HDC training

**Advantages:**
- ✅ Configurable number of classes
- ✅ Learns meaningful features from data
- ✅ Could discover sub-patterns within each sample type

**Disadvantages:**
- ❌ More complex pipeline
- ❌ Requires additional training step
- ❌ Labels are synthetic (no ground truth)
- ❌ May not be necessary if 3 classes work well

## Recommended Path Forward

### Phase 1: Start Simple (3 Classes)
1. Create `XRayDataset` class in `train_hdc.py`
2. Load 1,225 images from each sample type
3. Assign labels: 0=cameraman, 1=cell, 2=mandrill
4. Normalize to [0, 255] range
5. Use existing CNN+HDC pipeline
6. Evaluate accuracy with 3 classes

### Phase 2: If More Classes Needed
Only if Phase 1 results show need for more diversity:
1. Implement autoencoder (simple CNN-based)
2. Train on all 3,675 images
3. Cluster encoder features into N classes
4. Retrain CNN+HDC with synthetic labels

## Implementation Requirements

### Minimal changes to `train_hdc.py`:

```python
class XRayDataset(Dataset):
    def __init__(self, split='train', transform=None):
        # Load data from 3 h5 files
        # Assign labels: 0, 1, 2
        # Split train/test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # 32x32
        label = self.labels[idx]  # 0, 1, or 2

        # Normalize to [0, 255]
        image = (image / image.max() * 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        return image, label
```

### In `train_system()` function:
```python
if dataset_name == 'xray':
    train_dataset = XRayDataset(split='train', transform=transform)
    test_dataset = XRayDataset(split='test', transform=transform)
    num_classes = 3
```

## Statistical Separability

The three classes show different characteristics:

| Metric              | Cameraman | Cell    | Mandrill |
|---------------------|-----------|---------|----------|
| Mean intensity      | 0.333     | 0.328   | 0.330    |
| Std intensity       | 2.068     | 2.092   | 2.068    |
| Sparsity (% zeros)  | ~87%      | ~96%    | ~92%     |

These differences suggest the classes should be distinguishable by the CNN+HDC classifier.

## Next Steps

1. ✅ Dataset analysis complete
2. ⬜ Implement `XRayDataset` class
3. ⬜ Test with 3 classes
4. ⬜ Evaluate accuracy
5. ⬜ (Optional) Implement autoencoder if more classes needed

---

**CONCLUSION:** You can use the X-ray dataset with your HDC classifier. Start with the simple 3-class approach (no autoencoder needed). Only implement autoencoder clustering if you need more than 3 classes.
