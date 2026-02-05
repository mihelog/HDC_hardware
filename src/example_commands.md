# HDC Classifier — Command Reference

Quick-lookup guide for every way to invoke the build system.
All commands are run from this directory (`src/Claude/`).

---

## At a Glance

| What you want | Command |
|---------------|---------|
| Full manufacturing run (train + simulate) | `make manufacturing` |
| Same, but LFSR projection (77% less memory, 96.5% accuracy) | `make manufacturing_lfsr` |
| Train only (no Verilog sim) | `make python_only DATASET=manufacturing` |
| Simulate only (reuse existing weights) | `make manufacturing_verilog_only` |
| Quick smoke-test (tiny data, ~1 min) | `make manufacturing_quick` |
| Full run for any other dataset | `make <dataset>` (see per-dataset sections below) |
| LFSR on any dataset | `make <dataset> USE_LFSR_PROJECTION=1` |

---

## Makefile Parameter Reference

All parameters use `?=` defaults; override on the command line with `KEY=VALUE`.

### Dataset & Image

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `DATASET` | _(none — must be set)_ | Which dataset to load: `manufacturing`, `quickdraw`, `mnist`, `caltech101`, `xray` |
| `NUM_CLASSES` | `2` | Number of output classes. Override to e.g. `10` for full quickdraw/mnist |
| `IMAGE_SIZE` | `32` | Square image side in pixels (resized if needed) |
| `PIXEL_WIDTH` | `8` | Bits per pixel presented to the hardware pipeline |

### Training

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `EPOCHS` | `75` | Total CNN training epochs |
| `BATCH_SIZE` | `64` | Mini-batch size for CNN training |
| `TEST_SPLIT` | `0.2` | Fraction of data held out for testing |
| `SAMPLES_PER_CLASS` | `5000` | Training samples loaded per class (increased from 4000, 2026-02-02) |
| `NUM_TEST_IMAGES` | `200` | Images saved to `test_images.txt` for Verilog (increased from 100, 2026-02-02) |
| `SEED` | `42` | Global random seed (CNN, HDC, LFSR all derive from this) |
| `QAT_EPOCHS` | `0` | Epochs of quantisation-aware training. `0` = auto (starts at epoch 26) |
| `QAT_FUSE_BN` | `0` | `1` = fuse batch-norm into conv weights at QAT start |

### HDC Classifier

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `HV_DIM` | `10000` | Hypervector dimension (projection output size). Increased from 5000 (2026-02-01) for better accuracy |
| `ENCODING_LEVELS` | `4` | Binary features per FC output. `4` = three thresholds (384 features); `3` = two thresholds (256 features); `2` = one threshold (128 features). Increased from 3 (2026-02-01) |
| `PROJ_WEIGHT_WIDTH` | `4` | Bits per stored projection weight. Only matters when `USE_LFSR_PROJECTION=0` |
| `USE_LFSR_PROJECTION` | `0` | `1` = generate projection on-the-fly via 256 LFSRs; projection matrix is not stored at all (saves ~480 KB) |
| `USE_LEARNED_PROJECTION` | `0` | `1` = train the projection matrix instead of using random |
| `USE_RANDOM_PROJECTION` | `0` | `1` = use Gaussian-initialised random projection (alternative to default binary random) |
| `USE_PER_FEATURE_THRESHOLDS` | `1` | `1` = per-feature encoding thresholds; `0` = single global threshold |
| `ONLINE_LEARNING` | `1` | `1` = update class hypervectors during test evaluation (changed from 0 to 1, 2026-02-01) |

### Unsupervised / X-ray Only

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `UNLABELED` | `0` | `1` = run autoencoder clustering instead of supervised training |
| `NUM_CLUSTERS` | `10` | Number of clusters when `UNLABELED=1` |
| `QUANTIZE_BITS` | `8` | Pixel quantisation width for x-ray images |

### Simulation & Debug

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `VERILOG_SIM` | `iverilog` | Simulator backend: `iverilog`, `vcs`, or `xcelium` |
| `TESTBENCH` | `full` | `full` = 100-image testbench; `quick` = 5-image tiny testbench |
| `DEBUG` | `0` | `1` = enable CNN, HDC, loading, and latency debug prints |
| `DETAILED` | `0` | `1` = verbose per-bit loading trace (very noisy) |
| `DEBUG_ONLINE` | `0` | `1` = print online-learning update details |
| `WAVES` | `0` | `1` = dump VCD waveform file (`hdc_classifier.vcd`) |
| `FAST_LOAD` | `1` | `1` = backdoor-load weights directly into the DUT's internal memory (simulation only); `0` = use serial bit-stream load. Either this or `SKIP_LOADING` being `1` activates backdoor mode. |
| `SKIP_LOADING` | `1` | Same effect as `FAST_LOAD` — either flag being `1` defines `BACKDOOR_LOAD` for the testbench. To force serial loading, set **both** to `0`. |

> **How configuration loading works:**
> By default (`FAST_LOAD=1` / `SKIP_LOADING=1`) the testbench bypasses the serial
> config interface entirely.  It writes every weight, threshold, HV bit, and LUT
> entry straight into the DUT's `loaded_data_mem[]` array via hierarchical
> reference, then force-sets `loading_complete`.  This is instantaneous and is the
> normal way to run simulations.
>
> Set both flags to `0` to exercise the real serial path: each configuration bit is
> clocked through the `data_in` / `conf_valid_in` port one bit per cycle, exactly
> as real hardware would receive them.  A full manufacturing config is ~1 M bits, so
> serial loading adds millions of clock cycles to the simulation.
>
> ```bash
> # Default — backdoor loading (fast, normal usage)
> make manufacturing
>
> # Force serial bit-stream loading (exercises the real config interface)
> make manufacturing FAST_LOAD=0 SKIP_LOADING=0
> ```

### Utility Targets

| Target | What it does |
|--------|--------------|
| `make clean` | Remove all generated files (weights, sim outputs, cached data) |
| `make report` | Re-extract accuracy summary from the last `sim.log` |
| `make sweep` | Automatic sweep over quickdraw/mnist × 2/5/10 classes |
| `make help` | Print makefile help text |

---

## Manufacturing

Primary production target. 2-class binary classification on X-ray ptychography data.
8000 training images, 32×32 grayscale.

### Full pipeline (train + simulate)
```bash
make manufacturing
```
Uses stored projection matrix with default parameters:
- HV_DIM=10000
- ENCODING_LEVELS=4
- PROJ_WEIGHT_WIDTH=3
- ONLINE_LEARNING=1
- Memory: ~617 KB

### LFSR projection (77% less memory, 96.5% accuracy)
```bash
make manufacturing_lfsr
```
Eliminates the 480 KB stored projection matrix. 256 parallel 32-bit LFSRs
regenerate identical ±1 weights on-the-fly. Uses improved configuration:
- HV_DIM=10000 (increased from 5000)
- ENCODING_LEVELS=4 (increased from 3)
- ONLINE_LEARNING=1 (enabled)
- Adaptive class balancing (automatic)

Results: 96.5% accuracy (Class 0: 97%, Class 1: 96%), ~136 KB memory,
2694 cycles/image, 100% Python/Verilog agreement.

`PROJ_WEIGHT_WIDTH` is irrelevant here — the LFSR path bypasses the stored
matrix entirely.

### Python training only
```bash
make python_only DATASET=manufacturing
```
Produces `weights_and_hvs.txt` and `test_images.txt`. No Verilog runs.
Uses default parameters (HV_DIM=10000, ENCODING_LEVELS=4, ONLINE_LEARNING=1).

### Python training only — LFSR variant
```bash
make python_only DATASET=manufacturing USE_LFSR_PROJECTION=1
```
Same as above but generates LFSR-mode configuration (no projection matrix stored).

### Verilog simulation only (reuse existing weights)
```bash
make manufacturing_verilog_only
```
Requires a prior training run. Useful when you change Verilog and want to
re-simulate without retraining.

### Verilog only — LFSR variant
```bash
make verilog_only DATASET=manufacturing USE_LFSR_PROJECTION=1
```
Weights must have been produced by a matching LFSR training run (no projection
matrix in the file). Configuration parameters must match the training run.

### Quick smoke-test
```bash
make manufacturing_quick
```
4×4 images, 10-D hypervectors, 5 test images, 2 epochs. Runs in ~1 minute.

### Reduced-memory variant (smaller HV_DIM)
```bash
make manufacturing_small                  # full pipeline, HV_DIM=4000
make manufacturing_small_verilog_only     # simulate only
```

### With debug / waveforms
```bash
make manufacturing DEBUG=1                   # CNN + HDC debug prints
make manufacturing DEBUG=1 WAVES=1           # also dump VCD
make manufacturing DEBUG=1 DETAILED=1        # per-bit loading trace (very verbose)
make manufacturing_verilog_only DEBUG=1      # debug on re-simulation only
```

---

## QuickDraw

Sketch recognition. Auto-downloads data (~2000 samples per class).
Default `NUM_CLASSES=2`; append `NUM_CLASSES=10` for the full 10-class problem.

### Full pipeline
```bash
make quickdraw                              # 2 classes (default)
make quickdraw NUM_CLASSES=10               # full 10-class
```

### LFSR projection
```bash
make quickdraw USE_LFSR_PROJECTION=1
make quickdraw NUM_CLASSES=10 USE_LFSR_PROJECTION=1
```

### Python only
```bash
make python_only DATASET=quickdraw
make python_only DATASET=quickdraw NUM_CLASSES=10 USE_LFSR_PROJECTION=1
```

### Verilog only
```bash
make verilog_only DATASET=quickdraw
make verilog_only DATASET=quickdraw NUM_CLASSES=10 USE_LFSR_PROJECTION=1
```

### Quick smoke-test
```bash
make quickdraw_quick
```

---

## MNIST

Handwritten digit recognition. Auto-downloads via PyTorch (28×28 → resized to 32×32).
Default `NUM_CLASSES=2`; append `NUM_CLASSES=10` for the full digit set.

### Full pipeline
```bash
make mnist                                  # 2 classes (default)
make mnist NUM_CLASSES=10                   # full 10-class
```

### LFSR projection
```bash
make mnist USE_LFSR_PROJECTION=1
make mnist NUM_CLASSES=10 USE_LFSR_PROJECTION=1
```

### Python only
```bash
make python_only DATASET=mnist
make python_only DATASET=mnist NUM_CLASSES=10 USE_LFSR_PROJECTION=1
```

### Verilog only
```bash
make verilog_only DATASET=mnist
make verilog_only DATASET=mnist NUM_CLASSES=10 USE_LFSR_PROJECTION=1
```

### Quick smoke-test
```bash
make mnist_quick
```

---

## Caltech-101

Object recognition. Requires manual download of the Caltech-101 dataset before running.
Default `NUM_CLASSES=2`.

### Full pipeline
```bash
make caltech101
make caltech101 NUM_CLASSES=10              # more classes
```

### LFSR projection
```bash
make caltech101 USE_LFSR_PROJECTION=1
```

### Python only
```bash
make python_only DATASET=caltech101
make python_only DATASET=caltech101 USE_LFSR_PROJECTION=1
```

### Verilog only
```bash
make verilog_only DATASET=caltech101
make verilog_only DATASET=caltech101 USE_LFSR_PROJECTION=1
```

### Quick smoke-test
```bash
make caltech101_quick                       # 2 classes, 8×8 images
```

---

## X-ray (Unsupervised)

Unsupervised clustering via autoencoder. Uses local X-ray data directories.
Labels are inferred by k-means clustering, not ground truth.

### Full pipeline
```bash
make xray                                   # 10 clusters (default)
make xray NUM_CLUSTERS=5                    # fewer clusters
```

### LFSR projection
```bash
make xray USE_LFSR_PROJECTION=1
make xray NUM_CLUSTERS=5 USE_LFSR_PROJECTION=1
```

### Python only
```bash
make python_only DATASET=xray UNLABELED=1 NUM_CLUSTERS=10 QUANTIZE_BITS=8 IMAGE_SIZE=32
```

### Verilog only
```bash
make verilog_only DATASET=xray UNLABELED=1 NUM_CLUSTERS=10 QUANTIZE_BITS=8 IMAGE_SIZE=32
```

### Quick smoke-test
```bash
make xray_quick                             # 3 clusters, quick testbench
```
