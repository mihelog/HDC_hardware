#=======================================================================================
# Makefile for HDC Image Classification System
#=======================================================================================
#
# DESCRIPTION:
#   Complete build system for training, simulating, and verifying the hybrid CNN-HDC
#   image classification system. Orchestrates Python training and Verilog simulation.
#
# MAIN TARGETS:
#   make manufacturing        - Train and simulate on manufacturing dataset (2-class, 98% target)
#   make manufacturing_lfsr   - Manufacturing with LFSR projection (no stored matrix)
#   make quickdraw            - Train and simulate on QuickDraw dataset (10-class sketches)
#   make mnist                - Train and simulate on MNIST dataset (10-class digits)
#   make caltech101           - Train and simulate on Caltech-101 dataset (101-class objects)
#   make xray                 - Train and simulate on X-ray dataset (unsupervised clustering)
#   make help                 - Show detailed usage information
#
# QUICK TEST TARGETS (Fast, small-scale verification):
#   make manufacturing_quick  - Quick test with 5 images, 2 classes, 2 epochs
#   make quickdraw_quick      - Quick test for QuickDraw dataset
#   make mnist_quick          - Quick test for MNIST dataset
#   make caltech101_quick     - Quick test for Caltech-101 dataset
#   make xray_quick           - Quick test for X-ray dataset
#
# COMPONENT TARGETS:
#   make train                - Run Python training only (generates weights and test data)
#   make compile              - Compile Verilog only
#   make simulate             - Run Verilog simulation only (requires prior training)
#   make report               - Extract accuracy summary from simulation output
#   make clean                - Remove generated files
#
# DEBUG OPTIONS:
#   make <target> DEBUG=1     - Enable debug output (CNN, HDC, loading)
#   make <target> WAVES=1     - Enable waveform dumping (creates .vcd file)
#   make <target> DETAILED=1  - Enable detailed loading debug (very verbose)
#
# EXAMPLE USAGE:
#   make manufacturing                    - Full manufacturing pipeline
#   make manufacturing DEBUG=1 WAVES=1    - With debug output and waveforms
#   make train DATASET=quickdraw          - Train only (no simulation)
#   make manufacturing_verilog_only       - Simulate only (skip training)
#
# GENERATED FILES:
#   weights_and_hvs.txt       - Binary configuration for Verilog
#   test_images.txt           - Test images (8-bit or 16-bit per pixel)
#   test_labels.txt           - Ground truth labels
#   cnn_model.pth             - PyTorch checkpoint
#   verilog_params/*.vh       - Auto-generated Verilog parameters
#   sim.log                   - Verilog simulation log (always created by simulate target)
#   output                    - Combined Python + Verilog output log (only if you redirect/tee stdout)
#   hdc_classifier.vcd        - Waveform dump (if WAVES=1)
#
# CHANGE LOG:
#   2025-07-31: Added USE_LEARNED_PROJECTION=1 for better HDC accuracy
#   2025-08-01: Reverted QAT_EPOCHS to 0 (auto mode) for better convergence
#   2025-08-22: Set QAT_EPOCHS=20 for hardware-aware training from epoch 1
#
#=======================================================================================

#=======================================================================================
# CONFIGURATION - Build Tools and Simulation Settings
#=======================================================================================
PYTHON = python3                          # Python interpreter (use python3 for compatibility)
# Verilog simulator: iverilog, vcs, or xcelium
VERILOG_SIM = iverilog
SIM_TIME = 10ms                           # Maximum simulation time (timeout)

# Testbench selection: full or quick
TESTBENCH = full

# Debug flag support (disabled by default)
# To enable: make <target> DEBUG=1
# To enable detailed dumps: make <target> DEBUG=1 DETAILED=1
DEBUG ?= 0
DETAILED ?= 0

# Waveform dumping (disabled by default for speed)
# To enable: make <target> WAVES=1
WAVES ?= 0

ifeq ($(WAVES),1)
WAVE_FLAGS = -DDUMP_WAVEFORMS
WAVE_FLAGS_VCS = +define+DUMP_WAVEFORMS
else
WAVE_FLAGS =
WAVE_FLAGS_VCS =
endif

# Configuration loading control
# SKIP_LOADING=1 -> backdoor load (fast, skips serial stream)
# SKIP_LOADING=0 -> serial bit-by-bit load (slow, hardware-accurate)
SKIP_LOADING ?= 1

# Legacy alias (deprecated): FAST_LOAD
# If set explicitly, map to SKIP_LOADING unless SKIP_LOADING is already set by user.
ifneq ($(origin FAST_LOAD), undefined)
ifneq ($(origin SKIP_LOADING), command line)
ifneq ($(origin SKIP_LOADING), environment)
SKIP_LOADING := $(FAST_LOAD)
endif
endif
endif

ifeq ($(DEBUG),1)
DEBUG_FLAGS = -DDEBUG_CNN -DDEBUG_LOADING -DDEBUG_HDC -DDEBUG_RECIPROCAL_LUT -DDEBUG_LATENCY
DEBUG_FLAGS_VCS = +define+DEBUG_CNN +define+DEBUG_LOADING +define+DEBUG_RECIPROCAL_LUT +define+DEBUG_LATENCY
else
DEBUG_FLAGS =
DEBUG_FLAGS_VCS =
endif

ifeq ($(DETAILED),1)
DEBUG_FLAGS += -DDEBUG_LOADING_DETAILED
DEBUG_FLAGS_VCS += +define+DEBUG_LOADING_DETAILED
endif

# Optional: Enable online learning debug messages (independent of ONLINE_LEARNING functionality)
# Use DEBUG_ONLINE=1 to enable debug output for online learning updates
DEBUG_ONLINE ?= 0
ifeq ($(DEBUG_ONLINE),1)
DEBUG_FLAGS += -DDEBUG_ONLINE_LEARNING
DEBUG_FLAGS_VCS += +define+DEBUG_ONLINE_LEARNING
endif

# Include Python-generated shift parameters if available
SHIFT_PARAMS_FILE = verilog_params/shift_params.vh
ifneq ($(wildcard $(SHIFT_PARAMS_FILE)),)
    SHIFT_FLAGS = -DINCLUDE_SHIFT_PARAMS
    SHIFT_FLAGS_VCS = +define+INCLUDE_SHIFT_PARAMS
else
    SHIFT_FLAGS = 
    SHIFT_FLAGS_VCS = 
endif

# Python script parameters
# DATASET has no default - must be specified via target (quickdraw, mnist, caltech101, xray)
DATASET ?=
NUM_CLASSES = 2
IMAGE_SIZE = 32
HV_DIM=5000  # Default hypervector dimension (memory-optimized)
NUM_FEATURES=64  # Default FC layer outputs (memory-optimized)
FC_WEIGHT_WIDTH ?= 6  # FC weight width in bits (default: 6)
TEST_SPLIT = 0.2
EPOCHS = 75  # Default training epochs
BATCH_SIZE = 64
NUM_TEST_IMAGES = 200
SAMPLES_PER_CLASS = 5000  # Increased from 2000 to 4000 for more training data
ENCODING_LEVELS = 4  # Increased from 3 to 4 for finer feature granularity (2026-02-01)
PIXEL_WIDTH = 8   # Pixel width in bits for hardware implementation (auto-detected from data)
ARITHMETIC_MODE = integer  # Options: float, integer (default: integer for hardware accuracy)
USE_LEARNED_PROJECTION = 0  # Use learned projection matrix for better accuracy (default: disabled)
USE_RANDOM_PROJECTION = 0  # Use random projection (Johnson-Lindenstrauss) - fallback if learned fails
HDC_EPOCHS = 20  # Epochs for learned projection training
QAT_EPOCHS = 0  # Number of epochs for QAT (0 = auto mode: float training first 20 epochs, then QAT)
QAT_FUSE_BN ?= 0  # Fuse batch norm into conv weights when enabling QAT (0=disabled, 1=enabled)
ONLINE_LEARNING ?= 1  # Enable online learning during test evaluation (default: enabled)
USE_PER_FEATURE_THRESHOLDS ?= 1  # Use per-feature HDC thresholds in Python (0=global thresholds)
SEED ?= 42  # Random seed for reproducibility (can be changed for variety)
DEBUG_PIPELINE ?= 0  # Enable extra pipeline diagnostics in Python
DEBUG_SAMPLES ?= 2   # Number of images to dump when DEBUG_PIPELINE=1

# X-ray dataset configuration
XRAY_DIRS = "../../x-ray dataset/cameraman_32by32/cameraman_32by32" \
            "../../x-ray dataset/cell_32by32/cell_32by32" \
            "../../x-ray dataset/mandrill_32by32/mandrill_32by32"
UNLABELED = 0  # Set to 1 for unlabeled datasets with autoencoder clustering
NUM_CLUSTERS = 10  # Number of clusters for unlabeled data
QUANTIZE_BITS = 8  # Quantization bit width for x-ray images (8 or 16)

# Quick test parameters (override for quick testing)
ifeq ($(TESTBENCH),quick)
NUM_CLASSES = 2
IMAGE_SIZE = 4
HV_DIM = 10
NUM_TEST_IMAGES = 5
SAMPLES_PER_CLASS = 100
EPOCHS = 2
TB_SRCS = hdc_classifier_tb_quick.v
else
TB_SRCS = hdc_classifier_tb.v
endif

# Projection weight width (default 4 bits, use 1 for reduced memory)
PROJ_WEIGHT_WIDTH ?= 4

# LFSR on-the-fly projection (eliminates 480 KB projection matrix)
USE_LFSR_PROJECTION ?= 0

ifeq ($(SKIP_LOADING),1)
# Force BACKDOOR_LOAD when SKIP_LOADING is enabled
SKIP_FLAGS = -DBACKDOOR_LOAD
SKIP_FLAGS_VCS = +define+BACKDOOR_LOAD
else
SKIP_FLAGS =
SKIP_FLAGS_VCS =
endif

# Dataset specific settings
ifeq ($(DATASET),mnist)
IMAGE_SIZE = 28  # MNIST default size
endif

ifeq ($(DATASET),caltech101)
# Caltech-101 works better with larger images
IMAGE_SIZE = 64
BATCH_SIZE = 32  # Smaller batch due to larger images
endif

# Verilog files
RTL_SRCS = hdc_classifier.v hdc_top.v
TB_SRCS = hdc_classifier_tb.v
TOP_MODULE = hdc_classifier_tb

# DPI support (set DPI=1 to enable DPI file I/O)
DPI ?= 0
ifeq ($(DPI),1)
DPI_FLAGS = -DUSE_DPI
DPI_FLAGS_VCS = +define+USE_DPI
DPI_LIB = dpi_fileio.so
else
DPI_FLAGS =
DPI_FLAGS_VCS =
DPI_LIB =
endif

# Output files
WEIGHTS_FILE = weights_and_hvs.txt
TEST_IMAGES_FILE = test_images.txt
VCD_FILE = hdc_classifier.vcd

# Simulator-specific settings
ifeq ($(VERILOG_SIM),iverilog)
COMPILE_CMD = iverilog -g2012 -o sim.vvp
RUN_CMD = vvp sim.vvp
SIM_EXEC = sim.vvp
else ifeq ($(VERILOG_SIM),vcs)
COMPILE_CMD = vcs -sverilog -debug_access+all -o simv
RUN_CMD = ./simv
SIM_EXEC = simv simv.daidir
else ifeq ($(VERILOG_SIM),xcelium)
COMPILE_CMD = xrun -compile -access +rwc -sv
RUN_CMD = xrun -R -access +rwc -sv
SIM_EXEC = xcelium.d
endif

# Default target - require dataset selection
.PHONY: all
all:
ifndef DATASET
	@echo "======================================"
	@echo "ERROR: No dataset specified!"
	@echo "======================================"
	@echo ""
	@echo "Please specify a dataset to use:"
	@echo "  make quickdraw    - QuickDraw dataset (10 classes, sketches)"
	@echo "  make mnist        - MNIST dataset (handwritten digits)"
	@echo "  make caltech101   - Caltech-101 dataset (objects)"
	@echo "  make xray         - X-ray dataset (unlabeled, autoencoder clustering)"
	@echo ""
	@echo "Quick test options:"
	@echo "  make quickdraw_quick"
	@echo "  make mnist_quick"
	@echo "  make xray_quick"
	@echo ""
	@echo "Or run with explicit DATASET parameter:"
	@echo "  make all DATASET=quickdraw"
	@echo ""
	@exit 1
else
	$(MAKE) train simulate report
endif

#=======================================================================================
# QUICKDRAW DATASET TARGETS - Sketch Recognition (10-class)
#=======================================================================================
.PHONY: quickdraw
quickdraw: ## QuickDraw dataset (10-class sketch recognition)
          ## - Auto-downloads QuickDraw data (2000 samples/class)
          ## - 32×32 grayscale sketches, 7000-D hypervectors, 3-level encoding
          ## - Typical accuracy: 85-90%
	$(MAKE) all DATASET=quickdraw

.PHONY: quickdraw_quick
quickdraw_quick: ## Quick QuickDraw test (2 classes, small testbench)
	$(MAKE) all DATASET=quickdraw TESTBENCH=quick NUM_CLASSES=2

#=======================================================================================
# MNIST DATASET TARGETS - Handwritten Digit Recognition (10-class)
#=======================================================================================
.PHONY: mnist
mnist: ## MNIST dataset (10-class handwritten digits)
      ## - Auto-downloads MNIST via PyTorch (28×28 → 32×32)
      ## - Classic benchmark dataset
      ## - Typical accuracy: 95-98%
	$(MAKE) all DATASET=mnist

.PHONY: mnist_quick
mnist_quick: ## Quick MNIST test (2 classes, small testbench)
	$(MAKE) all DATASET=mnist TESTBENCH=quick NUM_CLASSES=2

#=======================================================================================
# CALTECH-101 DATASET TARGETS - Object Recognition (101-class)
#=======================================================================================
.PHONY: caltech101
caltech101: ## Caltech-101 dataset (101-class object recognition)
           ## - Requires manual download of Caltech-101
           ## - Large-scale multi-class classification
           ## - Uses 64×64 images for better accuracy
	$(MAKE) all DATASET=caltech101

.PHONY: caltech101_quick
caltech101_quick: ## Quick Caltech-101 test (2 classes, 8×8 images)
	$(MAKE) all DATASET=caltech101 TESTBENCH=quick NUM_CLASSES=2 IMAGE_SIZE=8

#=======================================================================================
# X-RAY DATASET TARGETS - Unsupervised Clustering
#=======================================================================================
.PHONY: xray
xray: ## X-ray dataset (unsupervised clustering with autoencoder)
	$(MAKE) all DATASET=xray UNLABELED=1 NUM_CLUSTERS=$(NUM_CLUSTERS) \
		QUANTIZE_BITS=8 IMAGE_SIZE=32

# X-ray quick test (fewer clusters for fast testing)
.PHONY: xray_quick
xray_quick:
	$(MAKE) all DATASET=xray UNLABELED=1 NUM_CLUSTERS=3 \
		QUANTIZE_BITS=8 IMAGE_SIZE=32 TESTBENCH=quick

# Manufacturing test target (High Accuracy: ~3.0 Mbits with 3-bit projection)
#=======================================================================================
# MANUFACTURING DATASET TARGETS - X-ray Diffraction (2-class, 98% accuracy target)
#=======================================================================================
# Primary production target for manufacturing inspection using X-ray ptychography data
# Verified configuration achieving 98% Verilog / 96.3% Python HDC accuracy
# Dataset: 8000 training samples (4000 per class), 100 test images, 32×32 grayscale
#=======================================================================================

.PHONY: manufacturing
manufacturing: ## Full manufacturing pipeline (Python training + Verilog simulation)
              ## - 2-class binary classification ("hit" vs "background")
              ## - 50 CNN epochs with QAT, 5000-D hypervectors, 2-level binary encoding
              ## - 3-bit projection weights for memory efficiency (3.01 Mbits total)
              ## - Achieves 98% Verilog accuracy, 96.3% Python accuracy
              ## - Takes ~30 min training + ~10 min simulation
	$(MAKE) all DATASET=manufacturing EPOCHS=50 ENCODING_LEVELS=3 USE_LEARNED_PROJECTION=0 \
		HV_DIM=5000 PROJ_WEIGHT_WIDTH=3

.PHONY: manufacturing_quick
manufacturing_quick: ## Quick manufacturing test (5 images, 2 epochs, ~1 min total)
                    ## - Fast verification for development/debugging
                    ## - Uses small testbench: 2 classes, 4×4 images, 10-D HVs
                    ## - Useful for quick syntax checks and pipeline verification
	$(MAKE) all DATASET=manufacturing TESTBENCH=quick USE_LEARNED_PROJECTION=0

.PHONY: manufacturing_verilog_only
manufacturing_verilog_only: ## Verilog simulation only (skip Python training)
                           ## - Requires prior training (weights_and_hvs.txt must exist)
                           ## - Useful for re-running Verilog with different debug settings
                           ## - Example: make manufacturing_verilog_only DEBUG=1 WAVES=1
	$(MAKE) verilog_only DATASET=manufacturing

.PHONY: manufacturing_lfsr
manufacturing_lfsr: ## LFSR projection — eliminates 480 KB stored projection matrix
                   ## - Projection generated on-the-fly by 256 parallel 32-bit LFSRs
                   ## - Memory: ~141 KB total (77% reduction vs stored matrix)
                   ## - Improved accuracy with HV_DIM=10000, ENCODING_LEVELS=4, online learning
	$(MAKE) all DATASET=manufacturing EPOCHS=50 USE_LEARNED_PROJECTION=0 \
		USE_LFSR_PROJECTION=1

.PHONY: manufacturing_small
manufacturing_small: ## Low-memory manufacturing configuration (~5.5 Mbits vs 3.01 Mbits)
                    ## - Uses 4000-D hypervectors (vs 5000-D) and 3-level encoding
                    ## - Reduced memory footprint for resource-constrained FPGAs
                    ## - May have slightly lower accuracy (~95-96%)
	$(MAKE) all DATASET=manufacturing HV_DIM=4000 ENCODING_LEVELS=3

.PHONY: manufacturing_small_verilog_only
manufacturing_small_verilog_only: ## Verilog-only version of manufacturing_small target
	$(MAKE) verilog_only DATASET=manufacturing HV_DIM=4000 ENCODING_LEVELS=3

# Train the HDC system using Python
.PHONY: train
train:
	@echo "======================================"
	@echo "Training HDC System"
	@echo "======================================"
	@echo "Dataset: $(DATASET)"
	@echo "Mode: $(TESTBENCH)"
	@echo "Parameters: CLASSES=$(NUM_CLASSES), SIZE=$(IMAGE_SIZE), HV_DIM=$(HV_DIM), NUM_FEATURES=$(NUM_FEATURES)"
	@echo "Epochs: $(EPOCHS), Batch size: $(BATCH_SIZE)"
ifeq ($(USE_RANDOM_PROJECTION),1)
	@echo "HDC Mode: RANDOM PROJECTION (Johnson-Lindenstrauss)"
else ifeq ($(USE_LEARNED_PROJECTION),1)
	@echo "HDC Mode: LEARNED PROJECTION ($(HDC_EPOCHS) epochs)"
else
	@echo "HDC Mode: Random binary projection (default)"
endif
ifeq ($(DATASET),quickdraw)
	@echo "Samples per class: $(SAMPLES_PER_CLASS)"
endif
	@echo "======================================"
	$(PYTHON) train_hdc.py \
		--dataset $(DATASET) \
		--num_classes $(NUM_CLASSES) \
		--image_size $(IMAGE_SIZE) \
		--hv_dim $(HV_DIM) \
		--test_split $(TEST_SPLIT) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--num_test_images $(NUM_TEST_IMAGES) \
		--samples_per_class $(SAMPLES_PER_CLASS) \
		--encoding_levels $(ENCODING_LEVELS) \
		--fc_weight_width $(FC_WEIGHT_WIDTH) \
		--pixel_width $(PIXEL_WIDTH) \
		--arithmetic_mode $(ARITHMETIC_MODE) \
		--qat_epochs $(QAT_EPOCHS) \
		$(if $(filter 1,$(QAT_FUSE_BN)),--qat_fuse_bn) \
		$(if $(filter 1,$(USE_RANDOM_PROJECTION)),--use_random_projection) \
		$(if $(filter 1,$(USE_LFSR_PROJECTION)),--use_lfsr_projection) \
		$(if $(filter 1,$(USE_LEARNED_PROJECTION)),--use_learned_projection) \
		$(if $(filter 1,$(USE_LEARNED_PROJECTION)),--hdc_epochs $(HDC_EPOCHS)) \
		$(if $(filter 1,$(ONLINE_LEARNING)),--enable_online_learning) \
		$(if $(filter 1,$(USE_PER_FEATURE_THRESHOLDS)),--use_per_feature_thresholds) \
		$(if $(filter 1,$(UNLABELED)),--unlabeled) \
		$(if $(filter 1,$(UNLABELED)),--data_dirs $(XRAY_DIRS)) \
		$(if $(filter 1,$(UNLABELED)),--num_clusters $(NUM_CLUSTERS)) \
		$(if $(filter 1,$(DEBUG_PIPELINE)),--debug_pipeline) \
		$(if $(filter 1,$(DEBUG_PIPELINE)),--debug_samples $(DEBUG_SAMPLES)) \
		--quantize_bits $(QUANTIZE_BITS) \
		--proj_weight_width $(PROJ_WEIGHT_WIDTH) \
		--seed $(SEED)
	@echo "Training complete. Weights saved to $(WEIGHTS_FILE)"
	@echo "Test images saved to $(TEST_IMAGES_FILE)"

# Build DPI library if needed
.PHONY: dpi_lib
dpi_lib:
ifeq ($(DPI),1)
	@echo "Building DPI library..."
	gcc -shared -fPIC -o dpi_fileio.so dpi_fileio.c
	@echo "DPI library built: dpi_fileio.so"
endif

# Compile Verilog with dynamic parameters
.PHONY: compile
compile: $(RTL_SRCS) $(TB_SRCS) dpi_lib
	@echo "======================================"
	@echo "Compiling Verilog"
	@echo "======================================"
	@echo "Testbench: $(TB_SRCS)"
	@echo "Parameters: CLASSES=$(NUM_CLASSES), SIZE=$(IMAGE_SIZE), HV_DIM=$(HV_DIM), NUM_FEATURES=$(NUM_FEATURES)"
	@if [ -f $(SHIFT_PARAMS_FILE) ]; then \
		echo "Using Python-generated shift parameters from $(SHIFT_PARAMS_FILE)"; \
	else \
		echo "Using default shift parameters (no $(SHIFT_PARAMS_FILE) found)"; \
	fi
	@echo "======================================"
ifeq ($(VERILOG_SIM),iverilog)
	@if [ -f $(SHIFT_PARAMS_FILE) ]; then \
		$(COMPILE_CMD) -DIMG_WIDTH_ARG=$(IMAGE_SIZE) -DIMG_HEIGHT_ARG=$(IMAGE_SIZE) \
			-DNUM_CLASSES_ARG=$(NUM_CLASSES) -DHV_DIM_ARG=$(HV_DIM) -DPIXEL_WIDTH_ARG=$(PIXEL_WIDTH) \
			-DNUM_FEATURES_ARG=$(NUM_FEATURES) \
			-DPROJ_WEIGHT_WIDTH_ARG=$(PROJ_WEIGHT_WIDTH) -DENCODING_LEVELS_ARG=$(ENCODING_LEVELS) \
			-DENABLE_ONLINE_LEARNING_ARG=$(ONLINE_LEARNING) -DUSE_PER_FEATURE_THRESHOLDS_ARG=$(USE_PER_FEATURE_THRESHOLDS) \
			-DUSE_LFSR_PROJECTION_ARG=$(USE_LFSR_PROJECTION) \
			-DDATASET_NAME=\"$(DATASET)\" \
			$(DEBUG_FLAGS) -DINCLUDE_SHIFT_PARAMS $(DPI_FLAGS) $(WAVE_FLAGS) $(SKIP_FLAGS) \
			$(TB_SRCS) $(RTL_SRCS); \
	else \
		$(COMPILE_CMD) -DIMG_WIDTH_ARG=$(IMAGE_SIZE) -DIMG_HEIGHT_ARG=$(IMAGE_SIZE) \
			-DNUM_CLASSES_ARG=$(NUM_CLASSES) -DHV_DIM_ARG=$(HV_DIM) -DPIXEL_WIDTH_ARG=$(PIXEL_WIDTH) \
			-DNUM_FEATURES_ARG=$(NUM_FEATURES) \
			-DPROJ_WEIGHT_WIDTH_ARG=$(PROJ_WEIGHT_WIDTH) -DENCODING_LEVELS_ARG=$(ENCODING_LEVELS) \
			-DENABLE_ONLINE_LEARNING_ARG=$(ONLINE_LEARNING) -DUSE_PER_FEATURE_THRESHOLDS_ARG=$(USE_PER_FEATURE_THRESHOLDS) \
			-DUSE_LFSR_PROJECTION_ARG=$(USE_LFSR_PROJECTION) \
			-DDATASET_NAME=\"$(DATASET)\" \
			$(DEBUG_FLAGS) $(DPI_FLAGS) $(WAVE_FLAGS) $(SKIP_FLAGS) \
			$(TB_SRCS) $(RTL_SRCS); \
	fi
else ifeq ($(VERILOG_SIM),vcs)
	$(COMPILE_CMD) +define+IMG_WIDTH_ARG=$(IMAGE_SIZE) +define+IMG_HEIGHT_ARG=$(IMAGE_SIZE) \
		+define+NUM_CLASSES_ARG=$(NUM_CLASSES) +define+HV_DIM_ARG=$(HV_DIM) +define+PIXEL_WIDTH_ARG=$(PIXEL_WIDTH) \
		+define+NUM_FEATURES_ARG=$(NUM_FEATURES) \
		+define+PROJ_WEIGHT_WIDTH_ARG=$(PROJ_WEIGHT_WIDTH) \
		+define+USE_PER_FEATURE_THRESHOLDS_ARG=$(USE_PER_FEATURE_THRESHOLDS) \
		+define+USE_LFSR_PROJECTION_ARG=$(USE_LFSR_PROJECTION) \
		+define+DATASET_NAME=\"$(DATASET)\" \
		$(DEBUG_FLAGS_VCS) $(SHIFT_FLAGS_VCS) $(WAVE_FLAGS_VCS) $(SKIP_FLAGS_VCS) \
		$(TB_SRCS) $(RTL_SRCS) -o simv
else ifeq ($(VERILOG_SIM),xcelium)
	$(COMPILE_CMD) -define IMG_WIDTH_ARG=$(IMAGE_SIZE) -define IMG_HEIGHT_ARG=$(IMAGE_SIZE) \
		-define NUM_CLASSES_ARG=$(NUM_CLASSES) -define HV_DIM_ARG=$(HV_DIM) -define PIXEL_WIDTH_ARG=$(PIXEL_WIDTH) \
		-define NUM_FEATURES_ARG=$(NUM_FEATURES) \
		-define PROJ_WEIGHT_WIDTH_ARG=$(PROJ_WEIGHT_WIDTH) \
		-define USE_PER_FEATURE_THRESHOLDS_ARG=$(USE_PER_FEATURE_THRESHOLDS) \
		-define USE_LFSR_PROJECTION_ARG=$(USE_LFSR_PROJECTION) \
		-define DATASET_NAME=\"$(DATASET)\" \
		$(SHIFT_FLAGS) $(WAVE_FLAGS) $(SKIP_FLAGS) \
		$(TB_SRCS) $(RTL_SRCS) -top $(TOP_MODULE)
endif

# Run simulation
.PHONY: simulate
simulate: compile $(WEIGHTS_FILE) $(TEST_IMAGES_FILE)
	@echo "======================================"
	@echo "Running Verilog Simulation"
	@echo "======================================"
	@echo "Mode: $(TESTBENCH)"
	@echo "Dataset: $(DATASET)"
	@echo "======================================"
	stdbuf -oL $(RUN_CMD) | tee sim.log
	@echo "Simulation complete. Waveform saved to $(VCD_FILE)"

# View waveforms (for iverilog with gtkwave)
.PHONY: wave
wave: $(VCD_FILE)
	gtkwave $(VCD_FILE) &

# Generate synthetic test data (for quick testing without training)
.PHONY: gen_test_data
gen_test_data:
	@echo "Generating synthetic test data..."
	@echo "Parameters: CLASSES=$(NUM_CLASSES), SIZE=$(IMAGE_SIZE), HV_DIM=$(HV_DIM), NUM_FEATURES=$(NUM_FEATURES)"
	@echo "import numpy as np" > gen_test_data.py
	@echo "with open('$(WEIGHTS_FILE)', 'w') as f:" >> gen_test_data.py
	@echo "    f.write('IMG_SIZE $(IMAGE_SIZE)\\n')" >> gen_test_data.py
	@echo "    f.write('NUM_CLASSES $(NUM_CLASSES)\\n')" >> gen_test_data.py
	@echo "    f.write('HV_DIM $(HV_DIM)\\n')" >> gen_test_data.py
	@echo "    # Calculate weights needed for image size $(IMAGE_SIZE)" >> gen_test_data.py
	@echo "    # Conv1 weights and bias" >> gen_test_data.py
	@echo "    for i in range(8*1*3*3 + 8):" >> gen_test_data.py
	@echo "        f.write(f'{np.random.randint(0, 256)}\\n')" >> gen_test_data.py
	@echo "    # Conv2 weights and bias" >> gen_test_data.py
	@echo "    for i in range(16*8*3*3 + 16):" >> gen_test_data.py
	@echo "        f.write(f'{np.random.randint(0, 256)}\\n')" >> gen_test_data.py
	@echo "    # FC weights and bias - size depends on pooled dimensions" >> gen_test_data.py
	@echo "    pool_size = $(IMAGE_SIZE) // 4  # After 2 pooling layers" >> gen_test_data.py
	@echo "    fc_input_size = 16 * pool_size * pool_size" >> gen_test_data.py
	@echo "    for i in range($(NUM_FEATURES) * fc_input_size + $(NUM_FEATURES)):" >> gen_test_data.py
	@echo "        f.write(f'{np.random.randint(0, 256)}\\n')" >> gen_test_data.py
	@echo "    # Projection matrix" >> gen_test_data.py
	@echo "    for i in range($(NUM_FEATURES) * ($(ENCODING_LEVELS) - 1) * $(HV_DIM)):" >> gen_test_data.py
	@echo "        f.write(f'{np.random.randint(0, 2)}\\n')" >> gen_test_data.py
	@echo "    # Hypervectors" >> gen_test_data.py
	@echo "    for i in range($(NUM_CLASSES) * $(HV_DIM)):" >> gen_test_data.py
	@echo "        f.write(f'{np.random.randint(0, 2)}\\n')" >> gen_test_data.py
	@echo "with open('$(TEST_IMAGES_FILE)', 'w') as f:" >> gen_test_data.py
	@echo "    for i in range($(NUM_TEST_IMAGES)):" >> gen_test_data.py
	@echo "        f.write(f'{i % $(NUM_CLASSES)}\\n')" >> gen_test_data.py
	@echo "        for j in range($(IMAGE_SIZE) * $(IMAGE_SIZE)):" >> gen_test_data.py
	@echo "            f.write(f'{np.random.randint(0, 65536)}\\n')" >> gen_test_data.py
	@$(PYTHON) gen_test_data.py
	@rm -f gen_test_data.py

# Quick test with synthetic data
.PHONY: quick_test
quick_test: gen_test_data simulate

# Run only the Python training
.PHONY: python_only
python_only: train

# Run only the Verilog simulation (assumes weights exist)
.PHONY: verilog_only
verilog_only: simulate

# Train with random binary projection (for comparison)
.PHONY: train_random
train_random:
	$(MAKE) train USE_LEARNED_PROJECTION=0

# Generate report from simulation log
.PHONY: report
report:
	@echo "======================================"
	@echo "Extracting Results"
	@echo "======================================"
	@grep -E "(Accuracy|SUCCESS|ERROR|correct|Total)" sim.log 2>/dev/null || echo "No simulation log found"

# Parameter sweep for different configurations
.PHONY: sweep
sweep:
	@for dataset in quickdraw mnist; do \
		for classes in 2 5 10; do \
			echo "Testing $(dataset) with NUM_CLASSES=$$classes"; \
			$(MAKE) all DATASET=$$dataset NUM_CLASSES=$$classes IMAGE_SIZE=32 EPOCHS=5; \
			mv sim.log sim_$${dataset}_c$$classes.log; \
		done; \
	done

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -f $(WEIGHTS_FILE) $(TEST_IMAGES_FILE)
	rm -f $(VCD_FILE) *.vvp *.log
	rm -rf $(SIM_EXEC)
	rm -rf quickdraw_data/
	rm -rf mnist_data/
	rm -rf caltech101_data/
	rm -rf __pycache__/
	rm -f *.pyc
	rm -f gen_test_data.py
	rm -f cnn_model.pth cnn_weights.pkl.bin hdc_vectors.pkl.bin
	rm -f cnn_weights_hw.bin hdc_vectors_hw.bin
	rm -rf verilog_params/

# Clean everything including downloaded data
.PHONY: cleanall
cleanall: clean
	rm -rf quickdraw_data/
	rm -rf mnist_data/
	rm -rf caltech101_data/

# Help target
.PHONY: help
help:
	@echo "HDC Image Classification System Makefile"
	@echo "========================================"
	@echo "Dataset Targets:"
	@echo "  quickdraw     - Run with QuickDraw dataset (10 classes, sketches)"
	@echo "  mnist         - Run with MNIST dataset (handwritten digits)"
	@echo "  caltech101    - Run with Caltech-101 dataset (objects)"
	@echo "  xray          - Run with X-ray dataset (unlabeled, 10 clusters)"
	@echo "  manufacturing - Run with Manufacturing X-ray dataset (2 classes)"
	@echo ""
	@echo "Quick Test Targets:"
	@echo "  quickdraw_quick - Quick test with QuickDraw (2 classes)"
	@echo "  mnist_quick     - Quick test with MNIST (2 classes)"
	@echo "  caltech101_quick - Quick test with Caltech-101 (2 classes)"
	@echo "  xray_quick      - Quick test with X-ray (3 clusters)"
	@echo "  manufacturing_quick - Quick test with Manufacturing X-ray (2 classes)"
	@echo ""
	@echo "Individual Steps:"
	@echo "  train         - Train HDC system using Python"
	@echo "  simulate      - Run Verilog simulation"
	@echo "  compile       - Compile Verilog only"
	@echo "  wave          - View waveforms (iverilog/gtkwave)"
	@echo "  python_only   - Run only Python training"
	@echo "  verilog_only  - Run only Verilog simulation"
	@echo "  verify_loading - Verify data loads correctly from Python to hardware"
	@echo "  report        - Extract results from simulation"
	@echo "  sweep         - Parameter sweep test"
	@echo "  clean         - Clean generated files"
	@echo "  cleanall      - Clean everything including data"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Configuration Variables:"
	@echo "  DATASET         - Dataset to use (quickdraw/mnist/caltech101/xray)"
	@echo "  TESTBENCH       - Testbench mode (full/quick, default: full)"
	@echo "  VERILOG_SIM     - Simulator (iverilog/vcs/xcelium, default: iverilog)"
	@echo "  DEBUG           - Enable debug messages (DEBUG=1, default: 0)"
	@echo "  DETAILED        - Enable detailed debug dumps (DETAILED=1, default: 0)"
	@echo "  NUM_CLASSES     - Number of classes (default: 2)"
	@echo "  NUM_CLUSTERS    - Number of clusters for xray (default: 10)"
	@echo "  IMAGE_SIZE      - Image size (default: 32, MNIST: 28, Caltech: 64)"
	@echo "  HV_DIM          - Hypervector dimension (default: 5000)"
	@echo "  EPOCHS          - Training epochs (default: 75)"
	@echo "  BATCH_SIZE      - Batch size (default: 64)"
	@echo "  ONLINE_LEARNING - Enable online learning (0=disabled, 1=enabled, default: 1)"
	@echo "  USE_PER_FEATURE_THRESHOLDS - Use per-feature HDC thresholds in Python/Verilog (0=global, 1=per-feature)"
	@echo "  QAT_FUSE_BN     - Fuse batch norm when enabling QAT (0=disabled, 1=enabled, default: 0)"
	@echo ""
	@echo "Examples:"
	@echo "  make quickdraw              # QuickDraw dataset (10 classes)"
	@echo "  make mnist                  # MNIST dataset"
	@echo "  make xray                   # X-ray dataset (10 clusters)"
	@echo "  make xray NUM_CLUSTERS=5    # X-ray with 5 clusters"
	@echo "  make quickdraw DEBUG=1      # Enable debug messages"
	@echo "  make mnist DEBUG=1 DETAILED=1  # Full debug with detailed dumps"
	@echo "  make quickdraw NUM_CLASSES=5 EPOCHS=20"
	@echo "  make quickdraw ONLINE_LEARNING=0  # Disable online learning"
	@echo "  make manufacturing QAT_FUSE_BN=1  # Fuse BN at QAT start"

# Dependencies
$(WEIGHTS_FILE):
	@echo "Error: $(WEIGHTS_FILE) not found. Run 'make train' first."
	@exit 1

$(TEST_IMAGES_FILE):
	@echo "Error: $(TEST_IMAGES_FILE) not found. Run 'make train' first."
	@exit 1

# For debugging - check file sizes
.PHONY: check_files
check_files:
	@echo "Checking generated files:"
	@if [ -f $(WEIGHTS_FILE) ]; then \
		fc_size=$$((16 * ($(IMAGE_SIZE) / 4) * ($(IMAGE_SIZE) / 4))); \
		total_cnn_weights=$$((8*1*3*3 + 8 + 16*8*3*3 + 16 + $(NUM_FEATURES)*$$fc_size + $(NUM_FEATURES))); \
		expected_lines=$$((3 + $$total_cnn_weights + $(NUM_FEATURES)*($(ENCODING_LEVELS)-1)*$(HV_DIM) + $(NUM_CLASSES)*$(HV_DIM))); \
		echo "  $(WEIGHTS_FILE): $$(wc -l < $(WEIGHTS_FILE)) lines"; \
		echo "  Expected lines: $$expected_lines"; \
		echo "  FC input size: $$fc_size"; \
		echo "  Total CNN weights: $$total_cnn_weights"; \
		actual_lines=$$(wc -l < $(WEIGHTS_FILE)); \
		if [ $$actual_lines -eq $$expected_lines ]; then \
			echo "  File size: CORRECT"; \
		else \
			echo "  File size: MISMATCH!"; \
		fi \
	else \
		echo "  $(WEIGHTS_FILE): NOT FOUND"; \
	fi
	@if [ -f $(TEST_IMAGES_FILE) ]; then \
		echo "  $(TEST_IMAGES_FILE): $$(wc -l < $(TEST_IMAGES_FILE)) lines"; \
		expected_lines=$$(($(NUM_TEST_IMAGES) * (1 + $(IMAGE_SIZE)*$(IMAGE_SIZE)))); \
		echo "  Expected lines: $$expected_lines"; \
	else \
		echo "  $(TEST_IMAGES_FILE): NOT FOUND"; \
	fi

# Lint Verilog code
.PHONY: lint
lint:
	@echo "Running Verilog lint..."
	verilator --lint-only $(RTL_SRCS)

# For continuous integration
.PHONY: ci
ci: quick mnist_quick
	@echo "CI checks passed"

# Verify loading - checks that all data loads correctly from Python to hardware
.PHONY: verify_loading
verify_loading:
	@echo "Verifying loading from weights_and_hvs.txt to hardware..."
	@$(PYTHON) verify_loading.py
	@if [ $$? -eq 0 ]; then \
		echo "✓ Loading verification PASSED"; \
	else \
		echo "✗ Loading verification FAILED"; \
		exit 1; \
	fi

# Debug compilation
.PHONY: debug_compile
debug_compile:
	@echo "VERILOG_SIM = $(VERILOG_SIM)"
	@echo "COMPILE_CMD = $(COMPILE_CMD)"
	@echo "RTL_SRCS = $(RTL_SRCS)"
	@echo "TB_SRCS = $(TB_SRCS)"
	@echo "TESTBENCH = $(TESTBENCH)"
	@echo "DATASET = $(DATASET)"
	@echo "NUM_CLASSES = $(NUM_CLASSES)"
	@echo "IMAGE_SIZE = $(IMAGE_SIZE)"
	@echo "HV_DIM = $(HV_DIM)"
	@echo "EPOCHS = $(EPOCHS)"
	@echo "SAMPLES_PER_CLASS = $(SAMPLES_PER_CLASS)"
	@echo "ENCODING_LEVELS = $(ENCODING_LEVELS)"
	@echo "Attempting to compile..."
	$(COMPILE_CMD) $(TB_SRCS) $(RTL_SRCS)

.PHONY: debug_vars
debug_vars:
	@echo "VERILOG_SIM='$(VERILOG_SIM)'"
	@echo "RUN_CMD='$(RUN_CMD)'"
	@echo "COMPILE_CMD='$(COMPILE_CMD)'"
