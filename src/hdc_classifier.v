//======================================================================================
// hdc_classifier.v - HDC Image Classification System
//======================================================================================
//
// DESCRIPTION:
//   Complete synthesizable hardware implementation of a hybrid CNN-HDC image classifier
//   achieving 98% accuracy on manufacturing datasets. The design implements an 11-stage
//   pipeline combining Convolutional Neural Network (CNN) feature extraction with
//   Hyperdimensional Computing (HDC) classification.
//
// PIPELINE STAGES:
//   1. Input Stage      - Receives 32x32x8-bit grayscale images
//   2. Conv1 Stage      - First convolution (1→8 channels, 3x3 kernel, ReLU)
//   3. Pool1 Stage      - Max pooling 2x2, stride 2
//   4. Conv2 Stage      - Second convolution (8→16 channels, 3x3 kernel, ReLU)
//   5. Pool2 Stage      - Max pooling 2x2, stride 2
//   6. FC Stage         - Fully connected layer (1024→128 features, no activation)
//   7. HDC Encoding     - Binary encoding using adaptive thresholds
//   8. Projection       - Random projection to 5000-D hypervector space (3-bit weights)
//   9. Query Generation - Binarize projection to create query hypervector
//   10. Hamming Distance - Compute distances to all class hypervectors
//   11. Classification   - Argmin distance to determine predicted class
//
// ARCHITECTURE:
//   - Fully flattened design (~6000 lines) - all stages integrated at top level
//   - Reason: Avoids Icarus Verilog issues with large unpacked arrays through ports
//   - Configuration data loaded into unified 1D memory array (loaded_data_mem)
//   - Pipeline synchronization via valid/done handshaking
//   - Latency: 2,444 cycles/image (for 32x32 manufacturing dataset)
//
// SYNTHESIZABILITY:
//   - FPGA/ASIC ready - uses only synthesizable Verilog constructs
//   - No floating-point arithmetic - all integer operations
//   - Debug printouts wrapped in `ifdef DEBUG directives
//   - Tested with Vivado (Xilinx) and Quartus (Intel/Altera)
//   - Target: Xilinx Ultrascale+, Intel Stratix/Arria
//   - Resources: ~50K LUTs, ~3 Mbits BRAM (manufacturing config)
//
// VERIFIED PERFORMANCE:
//   - Manufacturing dataset: 98% accuracy (Verilog), 96.3% (Python)
//   - Memory: 3.01 Mbits on-chip SRAM
//   - Throughput: ~200-400 images/sec @ 500 MHz
//
// CONFIGURATION LOADING:
//   - Serial bitstream loading via write_enable and data_in
//   - Configuration includes: CNN weights, biases, HDC thresholds,
//     projection matrix, class hypervectors, confidence LUT, online learning enable bit
//   - Total size: ~3.01 Mbits + 1 bit for NUM_CLASSES=2, HV_DIM=5000
//   - Online learning control: Last bit in configuration stream (0=disabled, 1=enabled)
//
// USAGE:
//   1. Assert reset_b (active low)
//   2. Load configuration bitstream serially (write_enable=1, data_in)
//   3. Wait for loading_complete
//   4. Present image_data and assert valid
//   5. Wait for valid_out (classification complete)
//   6. Read predicted_class and confidence
//
// AUTHORS:
//   Developed with assistance from AI tools (Claude and Gemini)
//   Principal Investigator: George Michelogiannakis (mihelog@lbl.gov)
//   Organization: Lawrence Berkeley National Laboratory
//
//======================================================================================

// Auto-generated CNN weight widths from Python training
`include "verilog_params/weight_widths.vh"

module hdc_classifier #(
    //==================================================================================
    // IMAGE PARAMETERS
    //==================================================================================
    parameter IMG_WIDTH = 32,              // Image width in pixels (default: 32)
    parameter IMG_HEIGHT = 32,             // Image height in pixels (default: 32)
    parameter PIXEL_WIDTH = 8,             // Bits per pixel (default: 8-bit grayscale)

    //==================================================================================
    // CLASSIFICATION PARAMETERS
    //==================================================================================
    parameter NUM_CLASSES = 2,             // Number of output classes (default: 2 for manufacturing)
    parameter HDC_HV_DIM = 5000,          // Hypervector dimension (reduced from 10000 to 5000 for memory optimization, 2026-02-02)
    parameter HDC_CONF_WIDTH = 4,          // Confidence output bit width (0-15 range)
    parameter CONFIDENCE_LUT_SIZE = 5000, // Confidence lookup table entries (match HV_DIM=5000)

    //==================================================================================
    // WEIGHT BIT WIDTHS (Quantization precision)
    //==================================================================================
    parameter CONV1_WEIGHT_WIDTH = `CONV1_WEIGHT_WIDTH_VH, // Conv1 weight/bias precision (from training)
    parameter CONV2_WEIGHT_WIDTH = `CONV2_WEIGHT_WIDTH_VH, // Conv2 weight/bias precision (from training)
    parameter FC_WEIGHT_WIDTH = `FC_WEIGHT_WIDTH_VH,       // FC weight precision (from training)
    parameter FC_BIAS_WIDTH = `FC_BIAS_WIDTH_VH,           // FC bias precision (from training)

    //==================================================================================
    // CNN ARCHITECTURE PARAMETERS
    //==================================================================================
    parameter FC_OUT_SIZE = 64,            // FC output dimension (num_features: 64 for memory-opt, 128 for high-acc, 2026-02-02)

    //==================================================================================
    // HDC PARAMETERS
    //==================================================================================
    parameter HDC_PROJ_WEIGHT_WIDTH = 4,   // Projection weight bit width (4-bit signed)
    parameter ENABLE_ONLINE_LEARNING = 1,  // Enable online learning (changed from 0 to 1, 2026-02-01)
    parameter ENCODING_LEVELS = 4,         // HDC encoding levels (increased from 3 to 4, 2026-02-01)
    parameter USE_ADAPTIVE_THRESHOLDS = 0, // 1=per-image min/max thresholds, 0=use thresholds from file (matches Python training)
    parameter USE_PER_FEATURE_THRESHOLDS = 1, // Use per-feature thresholds (changed from 0 to 1, 2026-02-01)
    parameter USE_LFSR_PROJECTION = 0,     // 1=generate projection on-the-fly via 256 LFSRs, 0=load from memory
    parameter LFSR_MASTER_SEED = 32'd42,   // Master seed for LFSR projection (seed[i] = MASTER_SEED + i + 1)

    //==================================================================================
    // PARALLELISM PARAMETERS (Trade-off: speed vs area)
    //==================================================================================
    parameter PARALLEL_PROJ = 20,          // Projection dimensions computed per cycle (default: 20)
    parameter PARALLEL_CONV1 = 8,          // Conv1 output channels computed in parallel
    parameter PARALLEL_CONV2 = 4           // Conv2 output channels computed in parallel
)(
    //==================================================================================
    // CONTROL SIGNALS
    //==================================================================================
    input clk,                             // System clock (all operations synchronous to this)
    input reset_b,                         // Active-low asynchronous reset

    //==================================================================================
    // CLASSIFICATION INTERFACE
    //==================================================================================
    input valid,                           // Asserted when image_data is valid (synchronous)
                                           // Starts classification pipeline
    input [IMG_WIDTH*IMG_HEIGHT*PIXEL_WIDTH-1:0] image_data,
                                           // Flattened input image (row-major order)
                                           // For 32x32x8: [8191:0] containing 1024 pixels

    //==================================================================================
    // CONFIGURATION LOADING INTERFACE (Serial bitstream)
    //==================================================================================
    input write_enable,                    // Asserted during configuration loading (synchronous)
    input data_in,                         // Serial configuration data bit (MSB first)
                                           // Loads: weights, biases, thresholds, proj matrix,
                                           //        class HVs, confidence LUT, online learning enable bit

    //==================================================================================
    // OUTPUTS (All synchronous, registered)
    //==================================================================================
    output reg [CLASS_WIDTH-1:0] predicted_class,
                                           // Predicted class ID (0 to NUM_CLASSES-1)
                                           // Valid when valid_out is asserted
    output reg [HDC_CONF_WIDTH-1:0] confidence,
                                           // Classification confidence (0-15, higher=more confident)
                                           // Derived from Hamming distance via lookup table
    output reg valid_out,                  // Asserted for 1 cycle when classification complete
                                           // Indicates predicted_class and confidence are valid
    output reg loading_complete,           // Asserted when configuration loading finished
                                           // Must be high before classification can begin
    output reg ready                       // Asserted when ready to accept new image
                                           // Low during classification pipeline execution
);

//======================================================================================
// DERIVED PARAMETERS (Automatically calculated from user parameters)
//======================================================================================
localparam CLASS_WIDTH = $clog2(NUM_CLASSES);  // Bits needed to represent class ID

//======================================================================================
// CNN ARCHITECTURE PARAMETERS (Fixed architecture)
//======================================================================================

// Conv1 Stage (First convolutional layer)
localparam CONV1_OUT_CH = 8;                   // Number of output channels (feature maps)
localparam CONV1_IN_CH = 1;                    // Number of input channels (grayscale=1)
localparam CONV1_KERNEL = 3;                   // Kernel size (3x3)
localparam CONV1_OUT_SIZE = IMG_WIDTH;         // Output spatial size (same as input with padding=1)
localparam POOL1_SIZE = 2;                     // Pooling window size (2x2)
localparam POOL1_OUT_SIZE = CONV1_OUT_SIZE / POOL1_SIZE;  // Output size after pooling (16 for 32x32 input)

// Conv2 Stage (Second convolutional layer)
localparam CONV2_OUT_CH = 16;                  // Number of output channels (feature maps)
localparam CONV2_IN_CH = CONV1_OUT_CH;         // Input channels = Conv1 output channels (8)
localparam CONV2_KERNEL = 3;                   // Kernel size (3x3)
localparam CONV2_OUT_SIZE = POOL1_OUT_SIZE;    // Output spatial size (same as input with padding=1)
localparam POOL2_SIZE = 2;                     // Pooling window size (2x2)
localparam POOL2_OUT_SIZE = (CONV2_OUT_SIZE / POOL2_SIZE) > 0 ? (CONV2_OUT_SIZE / POOL2_SIZE) : 1;
                                                // Output size after pooling (8 for 32x32 input)
                                                // Guarded against division resulting in 0

// Fully Connected Stage
localparam FC_INPUT_SIZE = CONV2_OUT_CH * POOL2_OUT_SIZE * POOL2_OUT_SIZE;
                                                // FC input dimension (flattened Conv2 output)
                                                // For 32x32 input: 16 * 8 * 8 = 1024
// FC_OUT_SIZE is now a parameter (defined in parameter list above)

//======================================================================================
// HDC ENCODING PARAMETERS
//======================================================================================
localparam EXPANDED_FEATURES = FC_OUT_SIZE * (ENCODING_LEVELS - 1);
                                                // Number of binary features after encoding
                                                // For ENCODING_LEVELS=2 (binary): 128 * 1 = 128
                                                // For thermometer encoding: 128 * (L-1)
localparam PROJ_MATRIX_ROWS = EXPANDED_FEATURES;  // Projection matrix row dimension (256 for binary)
localparam PROJ_MATRIX_COLS = HDC_HV_DIM;      // Projection matrix column dimension (5000)

//======================================================================================
// CONFIGURATION MEMORY SIZE CALCULATIONS (in bits)
// These determine the size of the loaded_data_mem array
//======================================================================================

// Conv1 weights and biases
localparam CONV1_WEIGHT_BITS = CONV1_OUT_CH * CONV1_IN_CH * CONV1_KERNEL * CONV1_KERNEL * CONV1_WEIGHT_WIDTH;
                                                // Conv1 weights: 8 * 1 * 3 * 3 * 12 = 864 bits
localparam CONV1_BIAS_BITS = CONV1_OUT_CH * CONV1_WEIGHT_WIDTH;
                                                // Conv1 biases: 8 * 12 = 96 bits

// Conv2 weights and biases
localparam CONV2_WEIGHT_BITS = CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL * CONV2_KERNEL * CONV2_WEIGHT_WIDTH;
                                                // Conv2 weights: 16 * 8 * 3 * 3 * 10 = 11,520 bits
localparam CONV2_BIAS_BITS = CONV2_OUT_CH * CONV2_WEIGHT_WIDTH;
                                                // Conv2 biases: 16 * 10 = 160 bits

// FC weights and biases
localparam FC_WEIGHT_BITS = FC_OUT_SIZE * FC_INPUT_SIZE * FC_WEIGHT_WIDTH;
                                                // FC weights: 128 * 1024 * 16 = 2,097,152 bits (2.1 Mbits)
                                                // This is the largest memory component
localparam FC_BIAS_BITS = FC_OUT_SIZE * FC_BIAS_WIDTH;
                                                // FC biases: 128 * 8 = 1,024 bits (8-bit precision)

// HDC encoding thresholds
localparam FEATURE_THRESH_COUNT = FC_OUT_SIZE * (ENCODING_LEVELS - 1);
localparam THRESHOLD_COUNT = USE_PER_FEATURE_THRESHOLDS ? (FEATURE_THRESH_COUNT + 1) : ENCODING_LEVELS;
localparam PROJ_THRESH_ID = USE_PER_FEATURE_THRESHOLDS ? FEATURE_THRESH_COUNT : (ENCODING_LEVELS - 1);
localparam THRESHOLD_BITS = 32 * THRESHOLD_COUNT;
                                                // Thresholds for HDC encoding levels
                                                // Global: ENCODING_LEVELS * 32 (feature thresholds + projection threshold)
                                                // Per-feature: (FC_OUT_SIZE*(ENCODING_LEVELS-1) + 1) * 32

// Projection matrix
localparam PROJ_MATRIX_BITS = USE_LFSR_PROJECTION ? 0 :
                                                (PROJ_MATRIX_ROWS * PROJ_MATRIX_COLS * HDC_PROJ_WEIGHT_WIDTH);
                                                // 0 when LFSR mode (generated on-the-fly)
                                                // Otherwise: 256 * 5000 * 3 = 3,840,000 bits (3.84 Mbits)

// Class hypervectors
localparam HV_BITS = NUM_CLASSES * HDC_HV_DIM;
                                                // Class hypervectors: 2 * 5000 = 10,000 bits (10 Kbits)

// Confidence lookup table
localparam CONFIDENCE_LUT_BITS = CONFIDENCE_LUT_SIZE * 4;
                                                // Confidence LUT: 500 * 4 = 2,000 bits (2 Kbits)

//======================================================================================
// SHIFT PARAMETERS (Fixed-point right-shift values for overflow prevention)
// These are dynamically set by Python training script based on profiling
// Shift values normalize activation ranges to prevent overflow in subsequent stages
//======================================================================================

`ifdef INCLUDE_SHIFT_PARAMS
    `include "verilog_params/shift_params.vh"  // Include auto-generated shift values from Python
`endif

// Pixel input shift (typically 0 - no shift needed for 8-bit inputs)
`ifdef PIXEL_SHIFT_OVERRIDE
    localparam PIXEL_SHIFT = `PIXEL_SHIFT_OVERRIDE;  // Override from command line if specified
`else
    `ifndef PIXEL_SHIFT
        `define PIXEL_SHIFT 0                    // Default: no shift for pixel values
    `endif
    localparam PIXEL_SHIFT = `PIXEL_SHIFT;
`endif

// Conv1 output shift (typically 7-8 for 12-bit weights)
`ifdef CONV1_SHIFT_OVERRIDE
    localparam CONV1_SHIFT = `CONV1_SHIFT_OVERRIDE;
`else
    `ifndef CONV1_SHIFT
        `define CONV1_SHIFT 7                    // Default: shift by 7 bits
    `endif
    localparam CONV1_SHIFT = `CONV1_SHIFT;       // Normalizes Conv1 output dynamic range
`endif

// Conv2 output shift (typically 6 for 10-bit weights)
`ifdef CONV2_SHIFT_OVERRIDE
    localparam CONV2_SHIFT = `CONV2_SHIFT_OVERRIDE;
`else
    `ifndef CONV2_SHIFT
        `define CONV2_SHIFT 6                    // Default: shift by 6 bits
    `endif
    localparam CONV2_SHIFT = `CONV2_SHIFT;       // Normalizes Conv2 output dynamic range
`endif

// FC output shift (typically 10 for 16-bit weights)
`ifdef FC_SHIFT_OVERRIDE
    localparam FC_SHIFT = `FC_SHIFT_OVERRIDE;
`else
    `ifndef FC_SHIFT
        `define FC_SHIFT 10                      // Default: shift by 10 bits
    `endif
    localparam FC_SHIFT = `FC_SHIFT;             // Normalizes FC output dynamic range
`endif

//======================================================================================
// CONFIGURATION MEMORY LAYOUT
// All configuration data packed into a single 1D unpacked array
// Layout (in order): Conv1 weights → Conv1 biases → Conv2 weights → Conv2 biases →
//                    FC weights → FC biases → Thresholds → Projection matrix →
//                    Class hypervectors → Confidence LUT → Online learning enable bit
//======================================================================================

// Start position (bit offset) for each configuration block
localparam CONV1_WEIGHT_START = 0;                                    // Conv1 weights start at bit 0
localparam CONV1_BIAS_START = CONV1_WEIGHT_START + CONV1_WEIGHT_BITS;  // Conv1 biases follow weights
localparam CONV2_WEIGHT_START = CONV1_BIAS_START + CONV1_BIAS_BITS;    // Conv2 weights follow Conv1 biases
localparam CONV2_BIAS_START = CONV2_WEIGHT_START + CONV2_WEIGHT_BITS;  // Conv2 biases follow weights
localparam FC_WEIGHT_START = CONV2_BIAS_START + CONV2_BIAS_BITS;       // FC weights (largest block)
localparam FC_BIAS_START = FC_WEIGHT_START + FC_WEIGHT_BITS;           // FC biases follow weights
localparam THRESHOLD_START = FC_BIAS_START + FC_BIAS_BITS;             // HDC encoding thresholds
localparam PROJ_MATRIX_START = THRESHOLD_START + THRESHOLD_BITS;       // Projection matrix (second largest)
localparam HV_START = PROJ_MATRIX_START + PROJ_MATRIX_BITS;            // Class hypervectors
localparam CONFIDENCE_LUT_START = HV_START + HV_BITS;                  // Confidence lookup table
localparam ONLINE_LEARNING_ENABLE_START = CONFIDENCE_LUT_START + CONFIDENCE_LUT_BITS;  // OL enable bit (last bit)
localparam TOTAL_BITS = ONLINE_LEARNING_ENABLE_START + 1;              // Total configuration size (+1 for OL bit)
                                                                        // Stored-matrix mode: ~3.01 Mbits; LFSR mode: ~138 KB

//======================================================================================
// MAIN CONFIGURATION STORAGE
// Unpacked 1D array holds all configuration data (weights, HVs, etc.)
// Loaded serially via write_enable and data_in during initialization
//======================================================================================
reg loaded_data_mem [0:TOTAL_BITS-1];   // Unpacked 1D array - each element is 1 bit
                                         // Easier to simulate than packed array
                                         // Maps to BRAM in synthesis
reg [$clog2(TOTAL_BITS)-1:0] load_counter;  // Tracks position during serial loading

//======================================================================================
// ONLINE LEARNING ENABLE BIT
// Loaded from configuration memory (last bit in bitstream)
// Extracted once loading is complete
//======================================================================================
reg online_learning_enable_reg;  // Internal storage for OL enable bit

//======================================================================================
// ACCESSOR FUNCTIONS (Extract configuration data from loaded_data_mem)
//======================================================================================
// These functions provide structured access to the packed configuration memory.
// All functions:
//   - Take coordinates/indices as inputs
//   - Calculate bit position in loaded_data_mem
//   - Extract multi-bit values bit-by-bit (for-loop over bit width)
//   - Return the extracted value
// This approach allows the flattened 1D memory to be accessed as if it were a
// structured array of weights, biases, and hypervectors.
//======================================================================================

    //==================================================================================
    // CONV1 WEIGHT ACCESSOR
    // Extracts a single Conv1 weight from memory
    // Inputs:
    //   och - Output channel index [0 to CONV1_OUT_CH-1] (default: 0-7)
    //   ich - Input channel index [0 to CONV1_IN_CH-1] (default: 0 for grayscale)
    //   ky  - Kernel Y position [0 to CONV1_KERNEL-1] (default: 0-2 for 3x3)
    //   kx  - Kernel X position [0 to CONV1_KERNEL-1] (default: 0-2 for 3x3)
    // Returns: CONV1_WEIGHT_WIDTH-bit signed weight value
    // Memory layout: weights stored in row-major order (och, ich, ky, kx)
    //==================================================================================
    function [CONV1_WEIGHT_WIDTH-1:0] get_conv1_weight;
        input [4:0] och, ich; input [2:0] ky, kx;
        integer idx, bit_pos, i; reg [CONV1_WEIGHT_WIDTH-1:0] res;
        begin
            // Calculate linear index into weight array
            idx = ((32'(och) * CONV1_IN_CH + 32'(ich)) * CONV1_KERNEL + 32'(ky)) * CONV1_KERNEL + 32'(kx);
            // Calculate bit position in memory
            bit_pos = CONV1_WEIGHT_START + idx * CONV1_WEIGHT_WIDTH;
            // Extract bits one by one from unpacked array
            for (i=0; i<CONV1_WEIGHT_WIDTH; i=i+1) res[i] = loaded_data_mem[bit_pos + i];
            get_conv1_weight = res;
        end
    endfunction

    //==================================================================================
    // CONV1 BIAS ACCESSOR
    // Extracts a single Conv1 bias from memory
    // Inputs:
    //   och - Output channel index [0 to CONV1_OUT_CH-1] (default: 0-7)
    // Returns: CONV1_WEIGHT_WIDTH-bit signed bias value (one per output channel)
    //==================================================================================
    function [CONV1_WEIGHT_WIDTH-1:0] get_conv1_bias;
        input [4:0] och;
        integer bit_pos, i; reg [CONV1_WEIGHT_WIDTH-1:0] res;
        begin
            // Calculate bit position (one bias per output channel)
            bit_pos = CONV1_BIAS_START + 32'(och) * CONV1_WEIGHT_WIDTH;
            // Extract bits from memory
            for (i=0; i<CONV1_WEIGHT_WIDTH; i=i+1) res[i] = loaded_data_mem[bit_pos + i];
            get_conv1_bias = res;
        end
    endfunction

    //==================================================================================
    // CONV2 WEIGHT ACCESSOR
    // Extracts a single Conv2 weight from memory
    // Inputs:
    //   och - Output channel index [0 to CONV2_OUT_CH-1] (default: 0-15)
    //   ich - Input channel index [0 to CONV2_IN_CH-1] (default: 0-7 from Conv1)
    //   ky  - Kernel Y position [0 to CONV2_KERNEL-1] (default: 0-2 for 3x3)
    //   kx  - Kernel X position [0 to CONV2_KERNEL-1] (default: 0-2 for 3x3)
    // Returns: CONV2_WEIGHT_WIDTH-bit signed weight value
    //==================================================================================
    function [CONV2_WEIGHT_WIDTH-1:0] get_conv2_weight;
        input [5:0] och, ich; input [1:0] ky, kx;
        integer idx, bit_pos, i; reg [CONV2_WEIGHT_WIDTH-1:0] res;
        begin
            // Calculate linear index (same layout as Conv1)
            idx = ((32'(och) * CONV2_IN_CH + 32'(ich)) * CONV2_KERNEL + 32'(ky)) * CONV2_KERNEL + 32'(kx);
            bit_pos = CONV2_WEIGHT_START + idx * CONV2_WEIGHT_WIDTH;
            for (i=0; i<CONV2_WEIGHT_WIDTH; i=i+1) res[i] = loaded_data_mem[bit_pos + i];
            get_conv2_weight = res;
        end
    endfunction

    //==================================================================================
    // CONV2 BIAS ACCESSOR
    // Extracts a single Conv2 bias from memory
    // Inputs:
    //   och - Output channel index [0 to CONV2_OUT_CH-1] (default: 0-15)
    // Returns: CONV2_WEIGHT_WIDTH-bit signed bias value
    //==================================================================================
    function [CONV2_WEIGHT_WIDTH-1:0] get_conv2_bias;
        input [5:0] och;
        integer bit_pos, i; reg [CONV2_WEIGHT_WIDTH-1:0] res;
        begin
            bit_pos = CONV2_BIAS_START + 32'(och) * CONV2_WEIGHT_WIDTH;
            for (i=0; i<CONV2_WEIGHT_WIDTH; i=i+1) res[i] = loaded_data_mem[bit_pos + i];
            get_conv2_bias = res;
        end
    endfunction

    //==================================================================================
    // FC WEIGHT ACCESSOR
    // Extracts a single fully-connected layer weight from memory
    // Inputs:
    //   och - Output feature index [0 to FC_OUT_SIZE-1] (default: 0-127)
    //   ich - Input feature index [0 to FC_INPUT_SIZE-1] (default: 0-1023)
    // Returns: FC_WEIGHT_WIDTH-bit signed weight value
    // Note: FC layer has 1024 inputs (flattened Pool2 output) and 128 outputs
    //==================================================================================
    function [FC_WEIGHT_WIDTH-1:0] get_fc_weight;
        input [6:0] och; input [9:0] ich;
        integer i, p; reg [FC_WEIGHT_WIDTH-1:0] r;
        begin
            // Calculate bit position (row-major: output x input)
            p = FC_WEIGHT_START + (32'(och) * FC_INPUT_SIZE + 32'(ich)) * FC_WEIGHT_WIDTH;
            for (i=0; i<FC_WEIGHT_WIDTH; i=i+1) r[i] = loaded_data_mem[p + i];
            get_fc_weight = r;
        end
    endfunction

    //==================================================================================
    // FC BIAS ACCESSOR
    // Extracts a single FC bias from memory
    // Inputs:
    //   och - Output feature index [0 to FC_OUT_SIZE-1] (default: 0-127)
    // Returns: FC_WEIGHT_WIDTH-bit signed bias value
    //==================================================================================
    function [FC_BIAS_WIDTH-1:0] get_fc_bias;
        input [6:0] och; integer i, p; reg [FC_BIAS_WIDTH-1:0] r;
        begin
            p = FC_BIAS_START + 32'(och) * FC_BIAS_WIDTH;
            for (i=0; i<FC_BIAS_WIDTH; i=i+1) r[i] = loaded_data_mem[p + i];
            get_fc_bias = r;
        end
    endfunction

    //==================================================================================
    // HDC ENCODING THRESHOLD ACCESSOR
    // Extracts one of the HDC encoding thresholds
    // Inputs:
    //   id - Threshold index [0 to NUM_THRESHOLDS-1]
    //        For 2-level encoding: 0 = level-1 threshold
    //        For 3-level encoding: 0 = level-1, 1 = level-2
    // Returns: 32-bit threshold value used to convert FC outputs to binary features
    // Note: Thresholds are typically percentile-based (e.g., median for 50th percentile)
    //==================================================================================
    function [31:0] get_thresh;
        input [15:0] id; integer i, p; reg [31:0] r;
        begin p = THRESHOLD_START + 32'(id)*32; for (i=0; i<32; i=i+1) r[i] = loaded_data_mem[p+i]; get_thresh=r; end
    endfunction

    //==================================================================================
    // PER-FEATURE THRESHOLD ACCESSOR
    // Extracts a threshold for a specific feature and encoding level
    // Inputs:
    //   feat - Feature index [0 to FC_OUT_SIZE-1]
    //   level - Encoding level index [1 to ENCODING_LEVELS-1]
    // Returns: 32-bit signed threshold
    // Layout: level-major, contiguous thresholds per level
    //==================================================================================
    function [31:0] get_feature_thresh;
        input [6:0] feat; input [3:0] level; integer id; integer i, p; reg [31:0] r;
        begin
            id = (32'(level - 1) * FC_OUT_SIZE) + 32'(feat);
            // Inline threshold extraction to avoid nested function calls (synthesis-friendly)
            p = THRESHOLD_START + 32'(id) * 32;
            for (i=0; i<32; i=i+1) r[i] = loaded_data_mem[p + i];
            get_feature_thresh = r;
        end
    endfunction

    //==================================================================================
    // PROJECTION THRESHOLD ACCESSOR
    // Extracts the projection threshold (same as first encoding threshold)
    // Returns: 32-bit threshold value (typically same as get_thresh(0))
    // Note: This is a convenience function for projection stage
    //==================================================================================
    function [31:0] get_proj_thresh;
        integer i, p; reg [31:0] r;
        begin p=THRESHOLD_START + 32'(PROJ_THRESH_ID)*32; for(i=0; i<32; i=i+1) r[i]=loaded_data_mem[p+i]; get_proj_thresh=r; end
    endfunction

    //==================================================================================
    // INTEGER SQUARE ROOT (floor) FOR 64-BIT VALUES
    // Used for per-image std deviation computation in normalization
    //==================================================================================
    function automatic [31:0] isqrt64;
        input [63:0] x;
        integer i;
        reg [63:0] rem;
        reg [31:0] root;
        reg [63:0] test_div;
        begin
            rem = 0;
            root = 0;
            for (i = 0; i < 32; i = i + 1) begin
                rem = {rem[61:0], x[63-2*i -: 2]};
                root = root << 1;
                test_div = {root, 1'b1};
                if (rem >= test_div) begin
                    rem = rem - test_div;
                    root = root + 1'b1;
                end
            end
            isqrt64 = root;
        end
    endfunction

    //==================================================================================
    // PROJECTION MATRIX ACCESSOR
    // Extracts a single 3-bit weight from the random projection matrix
    // Inputs:
    //   f - Feature index [0 to PROJ_MATRIX_ROWS-1] (default: 0-255 for 2x FC outputs)
    //   h - Hypervector dimension index [0 to PROJ_MATRIX_COLS-1] (default: 0-4999)
    // Returns: HDC_PROJ_WEIGHT_WIDTH-bit signed weight (3-bit: -4 to +3)
    // Purpose: Projects binary features from 256-D to 5000-D hypervector space
    //==================================================================================
    function [HDC_PROJ_WEIGHT_WIDTH-1:0] get_proj;
        input [15:0] f; input [13:0] h; integer i, p; reg [HDC_PROJ_WEIGHT_WIDTH-1:0] r;
        begin
            // Calculate position in projection matrix (row-major layout)
            p = PROJ_MATRIX_START + (32'(f)*PROJ_MATRIX_COLS+32'(h))*HDC_PROJ_WEIGHT_WIDTH;
            r = 0;
            for(i=0; i<HDC_PROJ_WEIGHT_WIDTH; i=i+1) r[i]=loaded_data_mem[p+i];
            get_proj=r;
        end
    endfunction

    //==================================================================================
    // CLASS HYPERVECTOR BIT ACCESSOR
    // Extracts a single bit from a class prototype hypervector
    // Inputs:
    //   c - Class index [0 to NUM_CLASSES-1] (default: 0-1 for manufacturing)
    //   b - Bit position [0 to HDC_HV_DIM-1] (default: 0-4999)
    // Returns: Single bit (0 or 1) from the class hypervector
    // Purpose: Class hypervectors are learned prototypes for each class
    //          Classification compares query HV to these via Hamming distance
    //==================================================================================
    function get_hv_bit; input [CLASS_WIDTH-1:0] c; input [13:0] b; integer p;
    begin p = HV_START + 32'(c)*HDC_HV_DIM + 32'(b); get_hv_bit = loaded_data_mem[p]; end endfunction

    //==================================================================================
    // CONFIDENCE LOOKUP TABLE ACCESSOR
    // Extracts confidence value for a given Hamming distance
    // Inputs:
    //   d - Hamming distance [0 to CONFIDENCE_LUT_SIZE-1] (default: 0-10000)
    // Returns: HDC_CONF_WIDTH-bit confidence score (default: 4-bit, 0-15)
    // Purpose: Maps raw Hamming distance to normalized confidence score
    //          Smaller distance = higher confidence
    //          Returns 0 if distance exceeds LUT size
    //==================================================================================
    function [HDC_CONF_WIDTH-1:0] get_conf; input [15:0] d; integer i, p; reg [HDC_CONF_WIDTH-1:0] r;
    begin if(d<CONFIDENCE_LUT_SIZE) begin p=CONFIDENCE_LUT_START+32'(d)*HDC_CONF_WIDTH; for(i=0;i<HDC_CONF_WIDTH;i=i+1) r[i]=loaded_data_mem[p+i]; get_conf=r; end else get_conf=0; end endfunction


// =====================================================================================
// INTERNAL SIGNALS (REGISTERS)
// =====================================================================================

// Online Learning Signals (Declared early for memory arbitration)
    reg ol_we;
    reg [$clog2(TOTAL_BITS)-1:0] ol_addr;
    reg ol_data;

// Conv1 Signals
    reg [PIXEL_WIDTH-1:0] image_pixels [0:IMG_HEIGHT-1][0:IMG_WIDTH-1];
    reg [2:0] conv1_state;
    reg [7:0] conv1_ch, conv1_y, conv1_x;
    reg conv1_valid_prev;
    reg signed [31:0] conv1_out [0:CONV1_OUT_CH-1][0:CONV1_OUT_SIZE-1][0:CONV1_OUT_SIZE-1];
    reg valid_s1, conv1_done;

// Pool1 Signals
    reg pool1_done, pool1_sent;
    reg signed [31:0] pool1_out [0:CONV1_OUT_CH-1][0:POOL1_OUT_SIZE-1][0:POOL1_OUT_SIZE-1];
    reg valid_s2;

// Conv2 Signals
    reg [2:0] conv2_state;
    reg [7:0] conv2_ch, conv2_y, conv2_x;
    reg signed [31:0] conv2_out [0:CONV2_OUT_CH-1][0:CONV2_OUT_SIZE-1][0:CONV2_OUT_SIZE-1];
    reg valid_s3, conv2_done;

// Pool2 Signals
    reg pool2_done, pool2_sent;
    reg signed [31:0] pool2_out [0:CONV2_OUT_CH-1][0:POOL2_OUT_SIZE-1][0:POOL2_OUT_SIZE-1];
    reg valid_s4;

// FC Signals
    reg [2:0] fc_state;
    reg [7:0] fc_idx;
    reg signed [31:0] fc_max;
    reg signed [31:0] fc_out [0:FC_OUT_SIZE-1];
    reg signed [31:0] fc_min, fc_range;
    reg valid_s5;

// Normalization Signals (per-image, based on FC outputs)
    reg signed [63:0] norm_sum;
    reg signed [63:0] norm_sumsq;
    reg [31:0] norm_count;
    reg signed [31:0] norm_scale_fp;   // Q8.8 scale factor
    reg signed [63:0] norm_shift_fp;   // Q8.8 shift factor
    reg signed [31:0] norm_test_mean_int;
    reg [31:0] norm_test_std_int;

//======================================================================================
// PER-IMAGE NORMALIZATION THRESHOLDS
// Dynamically computed from fc_min and fc_range for each image
//======================================================================================
    reg signed [31:0] adaptive_thresh1;  // fc_min + fc_range/3 (≈33rd percentile)
    reg signed [31:0] adaptive_thresh2;  // fc_min + 2*fc_range/3 (≈67th percentile)

// HDC Encoding Signals
    reg hdc_thresh_loaded;
    reg hdc_prev_valid;
    reg binary_features [0:EXPANDED_FEATURES-1];
    reg valid_s11;

// Projection Signals
    reg [2:0] proj_state;
    reg [15:0] proj_idx;
    reg signed [31:0] projection_sums [0:HDC_HV_DIM-1];
    reg valid_s14, proj_done;

// LFSR Projection Signals (active only when USE_LFSR_PROJECTION=1)
    reg  [31:0]              lfsr_state      [0:EXPANDED_FEATURES-1]; // Per-feature LFSR state
    reg  [PARALLEL_PROJ-1:0] lfsr_proj_wts   [0:EXPANDED_FEATURES-1]; // Weights for current batch
    reg  [31:0]              lfsr_next_state [0:EXPANDED_FEATURES-1]; // State after PARALLEL_PROJ steps

// Query Gen Signals
    reg signed [31:0] query_proj_thresh;
    reg query_thresh_loaded;
    reg query_hv [0:HDC_HV_DIM-1];
    reg valid_s15;

// Hamming Signals
    reg [2:0] ham_state;
    reg [7:0] ham_cls;
    reg [15:0] hamming_distances [0:NUM_CLASSES-1];
    reg valid_s16;

// Classification Signals
    reg [2:0] class_state;
    reg valid_s17;

// Online Learning Internal Signals
    reg [15:0] lfsr;
    reg [2:0] ol_state;
    reg [15:0] ol_idx;


// =====================================================================================
// FLATTENED LOGIC
// =====================================================================================

// --- Memory Management Logic & Global Reset ---
    always @(posedge clk) begin
        integer i;
        if (!reset_b) begin
            load_counter <= 0;
            loading_complete <= 0;
            hdc_thresh_loaded <= 0;
            online_learning_enable_reg <= 0;  // Initialize OL enable to 0
        end else begin
            if (write_enable && !loading_complete) begin
                loaded_data_mem[load_counter] <= data_in;
                load_counter <= load_counter + 1;
                if (load_counter == TOTAL_BITS - 1) begin
                    loading_complete <= 1;
                    online_learning_enable_reg <= data_in;  // Capture the last bit (OL enable)
                end
            end else if (ol_we && loading_complete) begin
                // Arbitrated write from Online Learning
                loaded_data_mem[ol_addr] <= ol_data;
            end
        end
    end

// --- Ready / Valid Out Logic ---
    always @(posedge clk or negedge reset_b) begin
        if (!reset_b) ready <= 0;
        else ready <= loading_complete && !valid_s1 && !valid_s2 && !valid_s3 && !valid_s4 && 
                     !valid_s5 && !valid_s11 && !valid_s14 && !valid_s15 && !valid_s16 && !valid_s17;
    end

    reg valid_s17_prev;
    always @(posedge clk or negedge reset_b) begin
        if (!reset_b) begin
            valid_out <= 0;
            valid_s17_prev <= 0;
        end else begin
            valid_s17_prev <= valid_s17;
            if (valid && loading_complete) valid_out <= 0;
            else if (valid_s17 && !valid_s17_prev) valid_out <= 1;
        end
    end

//======================================================================================
// STAGE 1: CONV1 - First Convolutional Layer
//======================================================================================
// Performs 2D convolution with the following configuration:
//   - Input channels: 1 (grayscale)
//   - Output channels: 8
//   - Kernel size: 3×3
//   - Padding: 1 (same padding, preserves spatial dimensions)
//   - Activation: ReLU (max(0, x))
//   - Output shift: CONV1_SHIFT (right-shift to prevent overflow)
//
// State Machine:
//   State 0: Wait for valid input, unpack flattened image into 2D array
//   State 1: Compute convolutions for all output channels and spatial positions
//   State 2: Done, signal next stage (Pool1)
//
// Parallelism: Processes PARALLEL_CONV1 output channels per cycle for performance
//======================================================================================
    always @(posedge clk or negedge reset_b) begin
        integer in_y, in_x, ky, kx, ch_off;
        reg signed [31:0] pix;
        reg signed [CONV1_WEIGHT_WIDTH-1:0] w, b;
        reg signed [63:0] sum, prod;

        if (!reset_b) begin
            // Reset all Conv1 state
            valid_s1 <= 0; conv1_state <= 0; conv1_valid_prev <= 0; conv1_done <= 0;
            conv1_ch <= 0; conv1_y <= 0; conv1_x <= 0;
        end else begin
            case (conv1_state)
                //==========================================================================
                // STATE 0: Wait for input and unpack image
                //==========================================================================
                0: begin
                    conv1_valid_prev <= valid;
                    // Detect rising edge of valid signal
                    if (valid && loading_complete && !conv1_valid_prev) begin
                        //======================================================================
                        // UNPACK IMAGE: Convert flattened image_data to 2D pixel array
                        // For-loops iterate over all pixels (32×32 = 1024 for default config)
                        // Each pixel is PIXEL_WIDTH bits (default: 8-bit grayscale)
                        // Unpacking enables easier spatial access during convolution
                        //======================================================================
                        for (in_y=0; in_y<IMG_HEIGHT; in_y=in_y+1)
                            for (in_x=0; in_x<IMG_WIDTH; in_x=in_x+1)
                                image_pixels[in_y][in_x] = image_data[(in_y*IMG_WIDTH+in_x)*PIXEL_WIDTH +: PIXEL_WIDTH];

                        `ifdef DEBUG_CNN
                        $display("DEBUG_TRACE: Image Loaded. Pixel[0][0] = %d", image_pixels[0][0]);
                        `endif

                        // Initialize counters and advance to convolution state
                        conv1_state <= 1; conv1_ch <= 0; conv1_y <= 0; conv1_x <= 0; valid_s1 <= 1; conv1_done <= 0;
                    end
                end

                //==========================================================================
                // STATE 1: Perform 2D convolution across all channels and spatial positions
                //==========================================================================
                1: begin
                    //======================================================================
                    // PARALLEL CHANNEL PROCESSING
                    // Process PARALLEL_CONV1 output channels simultaneously
                    // ch_off iterates from 0 to PARALLEL_CONV1-1
                    // Actual channel index is (conv1_ch + ch_off)
                    //======================================================================
                    for (ch_off=0; ch_off<PARALLEL_CONV1; ch_off=ch_off+1) begin
                        if (32'(conv1_ch) + ch_off < CONV1_OUT_CH) begin
                            // Initialize accumulator with bias for this output channel
                            b = get_conv1_bias(5'(conv1_ch + ch_off));
                            sum = $signed({{(64-CONV1_WEIGHT_WIDTH){b[CONV1_WEIGHT_WIDTH-1]}}, b});

                            //==================================================================
                            // KERNEL Y LOOP (3 iterations for 3×3 kernel)
                            // ky iterates over kernel rows [0, 1, 2]
                            //==================================================================
                            for (ky=0; ky<CONV1_KERNEL; ky=ky+1) begin
                                //==============================================================
                                // KERNEL X LOOP (3 iterations for 3×3 kernel)
                                // kx iterates over kernel columns [0, 1, 2]
                                // Total 9 iterations (3×3) for each spatial position
                                //==============================================================
                                for (kx=0; kx<CONV1_KERNEL; kx=kx+1) begin
                                    //==========================================================
                                    // PADDING CALCULATION
                                    // Padding=1 means add 1 pixel border around image
                                    // in_y/in_x are actual pixel coordinates (may be negative)
                                    // Subtract 1 to shift kernel center, then add ky/kx offset
                                    //==========================================================
                                    in_y = $signed({24'b0, conv1_y}) - 32'sd1 + ky;
                                    in_x = $signed({24'b0, conv1_x}) - 32'sd1 + kx;

                                    //==========================================================
                                    // BOUNDARY HANDLING
                                    // If in_y/in_x within image bounds: use actual pixel
                                    // If outside bounds (padding region): use 0
                                    //==========================================================
                                    if (in_y >= 0 && in_y < IMG_HEIGHT && in_x >= 0 && in_x < IMG_WIDTH) begin
                                        // Valid pixel: expand unsigned 8-bit to signed 32-bit
                                        // No mean subtraction (matches Python QAT model)
                                        pix = $signed({24'b0, image_pixels[in_y][in_x]});
                                    end else begin
                                        pix = 0; // Zero padding for out-of-bounds
                                    end

                                    if (1) begin // Scope block for local variable 'w'
                                        // Load weight for this (output_ch, input_ch=0, ky, kx)
                                        w = get_conv1_weight(5'(conv1_ch + ch_off), 5'd0, 3'(ky), 3'(kx));
                                        // Multiply pixel by weight and accumulate
                                        prod = pix * $signed(w);
                                        sum = sum + prod;
                                    end
                                end // End kx loop
                            end // End ky loop

                            //==================================================================
                            // POST-PROCESSING
                            // 1. Right-shift by CONV1_SHIFT (normalize dynamic range)
                            // 2. Apply ReLU activation (max(0, sum))
                            // 3. Store result in conv1_out array
                            //==================================================================
                            sum = sum >>> CONV1_SHIFT;
                            conv1_out[3'(conv1_ch+ch_off)][5'(conv1_y)][5'(conv1_x)] <= (sum > 0) ? sum[31:0] : 0;

                            // DEBUG: Print all channels for first pixel
                            `ifdef DEBUG_CNN
                            if (conv1_y == 0 && conv1_x == 0)
                                $display("DEBUG_TRACE: Conv1[CH=%d][0][0] = %d", conv1_ch+ch_off, (sum > 0) ? sum[31:0] : 0);
                            `endif
                        end
                    end

                    //==================================================================
                    // COUNTER INCREMENT LOGIC
                    // Iterate through all output spatial positions and channels
                    // Order: X (innermost) → Y → Channel (outermost)
                    // For 32×32 input with PARALLEL_CONV1=1: 32×32×8 = 8,192 cycles
                    //==================================================================
                    if (conv1_x == CONV1_OUT_SIZE-1) begin
                        conv1_x <= 0;                              // Reset X, increment Y
                        if (conv1_y == CONV1_OUT_SIZE-1) begin
                            conv1_y <= 0;                          // Reset Y, increment channel
                            if (conv1_ch + PARALLEL_CONV1 >= CONV1_OUT_CH) begin
                                conv1_state <= 2;                  // All channels done
                                conv1_done <= 1;                   // Signal completion
                            end else begin
                                conv1_ch <= conv1_ch + PARALLEL_CONV1; // Process next channel batch
                            end
                        end else begin
                            conv1_y <= conv1_y + 1;                // Increment Y
                        end
                    end else begin
                        conv1_x <= conv1_x + 1;                    // Increment X
                    end
                end

                //==========================================================================
                // STATE 2: Reset and return to idle
                //==========================================================================
                2: begin
                    conv1_done <= 0;
                    valid_s1 <= 0;
                    conv1_state <= 0;
                end
            endcase
        end
    end

//======================================================================================
// STAGE 2: POOL1 - First Max Pooling Layer
//======================================================================================
// Performs 2×2 max pooling on Conv1 output
//   - Input size: 32×32×8 (Conv1 output with padding=1)
//   - Output size: 16×16×8 (2× spatial reduction)
//   - Pool size: 2×2
//   - Stride: 2 (non-overlapping windows)
//   - Operation: max(4 values in 2×2 window)
//
// Implementation: Single-cycle operation using combinational for-loops
//   All channels and spatial positions computed in parallel
//======================================================================================
    always @(posedge clk or negedge reset_b) begin
        integer ch, y, x; reg signed [31:0] max_v;
        if (!reset_b) begin
            valid_s2 <= 0; pool1_done <= 0; pool1_sent <= 0;
        end else begin
            if (valid_s1 && !pool1_done && conv1_done) begin
                pool1_done <= 1;
                //==========================================================================
                // MAX POOLING COMPUTATION
                // Triple nested for-loops iterate over:
                //   ch: All channels [0 to CONV1_OUT_CH-1] (default: 0-7)
                //   y:  Output Y positions [0 to POOL1_OUT_SIZE-1] (default: 0-15)
                //   x:  Output X positions [0 to POOL1_OUT_SIZE-1] (default: 0-15)
                //
                // For each output position (ch, y, x):
                //   Input window spans conv1_out[ch][y*2:y*2+1][x*2:x*2+1]
                //   Compute max of 4 values: [y*2][x*2], [y*2][x*2+1], [y*2+1][x*2], [y*2+1][x*2+1]
                //==========================================================================
                for (ch=0; ch<CONV1_OUT_CH; ch=ch+1)
                    for (y=0; y<POOL1_OUT_SIZE; y=y+1)
                        for (x=0; x<POOL1_OUT_SIZE; x=x+1) begin
                            // Find maximum of 2×2 window
                            max_v = conv1_out[ch][y*2][x*2];
                            if (conv1_out[ch][y*2][x*2+1] > max_v) max_v = conv1_out[ch][y*2][x*2+1];
                            if (conv1_out[ch][y*2+1][x*2] > max_v) max_v = conv1_out[ch][y*2+1][x*2];
                            if (conv1_out[ch][y*2+1][x*2+1] > max_v) max_v = conv1_out[ch][y*2+1][x*2+1];
                            pool1_out[ch][y][x] <= max_v;
                        end
                `ifdef DEBUG_CNN
                $display("DEBUG_TRACE: Pool1 Done. Pool1[0][0][0] = %d", pool1_out[0][0][0]); // Warning: this prints previous value
                `endif
            end else if (pool1_done && !pool1_sent) begin
                // Signal next stage that Pool1 output is ready
                valid_s2 <= 1; pool1_sent <= 1;
                `ifdef DEBUG_CNN
                $display("DEBUG_TRACE: Pool1 Valid. Pool1[0][0][0] = %d", pool1_out[0][0][0]);
                $display("DEBUG_CNN: Pool1[0] first 8 values:");
                for (ch=0; ch<8; ch=ch+1)
                    $display("  Pool1[%0d][0][0] = %0d", ch, pool1_out[ch][0][0]);
                `endif
            end else if (!valid_s1) begin
                // Reset when Conv1 invalidates
                pool1_done <= 0; valid_s2 <= 0; pool1_sent <= 0;
            end else if (pool1_sent) valid_s2 <= 0;
        end
    end

//======================================================================================
// STAGE 3: CONV2 - Second Convolutional Layer
//======================================================================================
// Performs 2D convolution on Pool1 output
//   - Input channels: 8 (from Pool1)
//   - Output channels: 16
//   - Kernel size: 3×3
//   - Padding: 1 (same padding)
//   - Activation: ReLU
//   - Output shift: CONV2_SHIFT
//
// State Machine:
//   State 0: Wait for Pool1 output
//   State 1: Compute convolutions for all output channels and spatial positions
//   State 2: Done, signal next stage (Pool2)
//======================================================================================
    always @(posedge clk or negedge reset_b) begin
        integer ich, ky, kx, ch_off, in_y, in_x;
        reg signed [31:0] p1_val;
        reg signed [CONV2_WEIGHT_WIDTH-1:0] w, b;
        reg signed [63:0] sum, prod;

        if (!reset_b) begin
            valid_s3 <= 0; conv2_state <= 0; conv2_done <= 0;
            conv2_ch <= 0; conv2_y <= 0; conv2_x <= 0;
        end else begin
            case (conv2_state)
                //==========================================================================
                // STATE 0: Wait for Pool1 output
                //==========================================================================
                0: if (valid_s2) begin conv2_state <= 1; conv2_ch <= 0; conv2_y <= 0; conv2_x <= 0; valid_s3 <= 1; end

                //==========================================================================
                // STATE 1: Perform 2D convolution with multiple input channels
                //==========================================================================
                1: begin
                    //======================================================================
                    // PARALLEL CHANNEL PROCESSING
                    // Process PARALLEL_CONV2 output channels simultaneously
                    //======================================================================
                    for (ch_off=0; ch_off<PARALLEL_CONV2; ch_off=ch_off+1) begin
                        if (32'(conv2_ch) + ch_off < CONV2_OUT_CH) begin
                            // Initialize accumulator with bias
                            b = get_conv2_bias(6'(conv2_ch + ch_off));
                            sum = $signed({{(64-CONV2_WEIGHT_WIDTH){b[CONV2_WEIGHT_WIDTH-1]}}, b});

                            //==================================================================
                            // INPUT CHANNEL LOOP (8 iterations for 8 input channels from Pool1)
                            // ich iterates over all input channels [0 to CONV2_IN_CH-1]
                            // Each output feature map is computed from ALL input channels
                            //==================================================================
                            for (ich=0; ich<CONV2_IN_CH; ich=ich+1) begin
                                //==============================================================
                                // KERNEL Y LOOP (3 iterations for 3×3 kernel)
                                //==============================================================
                                for (ky=0; ky<CONV2_KERNEL; ky=ky+1) begin
                                    //==========================================================
                                    // KERNEL X LOOP (3 iterations for 3×3 kernel)
                                    // Total: 8 input_ch × 3×3 kernel = 72 multiply-accumulates
                                    //==========================================================
                                    for (kx=0; kx<CONV2_KERNEL; kx=kx+1) begin
                                        // Calculate input coordinates with padding
                                        in_y = $signed({24'b0, conv2_y}) - 32'sd1 + ky;
                                        in_x = $signed({24'b0, conv2_x}) - 32'sd1 + kx;

                                        // Boundary check (padding=1)
                                        if (in_y>=0 && in_y<POOL1_OUT_SIZE && in_x>=0 && in_x<POOL1_OUT_SIZE) begin
                                            // Load value from Pool1 output
                                            p1_val = pool1_out[ich][in_y][in_x];
                                            // Load weight for (out_ch, in_ch, ky, kx)
                                            w = get_conv2_weight(6'(conv2_ch+ch_off), 6'(ich), 2'(ky), 2'(kx));
                                            // Multiply and accumulate
                                            prod = p1_val * $signed(w);
                                            sum = sum + prod;
                                        end
                                        // Note: Zero padding implicit (no accumulation for out-of-bounds)
                                    end
                                end
                            end

                            // Apply right-shift and ReLU activation
                            sum = sum >>> CONV2_SHIFT;
                            conv2_out[4'(conv2_ch+ch_off)][4'(conv2_y)][4'(conv2_x)] <= (sum > 0) ? sum[31:0] : 0;

                            // DEBUG
                            `ifdef DEBUG_CNN
                            if (conv2_ch+ch_off == 0 && conv2_y == 0 && conv2_x == 0)
                                $display("DEBUG_TRACE: Conv2[0][0][0] = %d", (sum > 0) ? sum[31:0] : 0);
                            if (conv2_y == 0 && conv2_x == 0 && conv2_ch+ch_off < 4)
                                $display("  Conv2[%0d][0][0] = %0d", conv2_ch+ch_off, (sum > 0) ? sum[31:0] : 0);
                            `endif
                        end
                    end
                    if (conv2_x == CONV2_OUT_SIZE-1) begin
                        conv2_x <= 0;
                        if (conv2_y == CONV2_OUT_SIZE-1) begin
                            conv2_y <= 0;
                            if (conv2_ch + PARALLEL_CONV2 >= CONV2_OUT_CH) begin
                            conv2_state <= 2;
                            conv2_done <= 1; // Assert done early
                        end else begin
                            conv2_ch <= conv2_ch + PARALLEL_CONV2;
                        end
                    end else conv2_y <= conv2_y + 1;
                end else conv2_x <= conv2_x + 1;
            end
            2: begin valid_s3 <= 0; conv2_state <= 0; conv2_done <= 0; end
            endcase
        end
    end

// --- STAGE 4: POOL2 Logic ---
    always @(posedge clk or negedge reset_b) begin
        integer ch, y, x; reg signed [31:0] max_v;
        if (!reset_b) begin valid_s4<=0; pool2_done<=0; pool2_sent<=0; end
        else begin
            if (valid_s3 && !pool2_done && conv2_done) begin
                pool2_done <= 1;
                for (ch=0; ch<CONV2_OUT_CH; ch=ch+1)
                    for (y=0; y<POOL2_OUT_SIZE; y=y+1)
                        for (x=0; x<POOL2_OUT_SIZE; x=x+1) begin
                            max_v = conv2_out[ch][y*2][x*2];
                            if (conv2_out[ch][y*2][x*2+1] > max_v) max_v = conv2_out[ch][y*2][x*2+1];
                            if (conv2_out[ch][y*2+1][x*2] > max_v) max_v = conv2_out[ch][y*2+1][x*2];
                            if (conv2_out[ch][y*2+1][x*2+1] > max_v) max_v = conv2_out[ch][y*2+1][x*2+1];
                            pool2_out[ch][y][x] <= max_v;
                        end
            end else if (pool2_done && !pool2_sent) begin
                valid_s4<=1; pool2_sent<=1;
                `ifdef DEBUG_CNN
                $display("DEBUG_TRACE: Pool2 Valid. Pool2[0][0][0] = %d", pool2_out[0][0][0]);
                $display("DEBUG_CNN: Pool2[0] first 16 values:");
                for (ch=0; ch<16; ch=ch+1)
                    $display("  Pool2[%0d][0][0] = %0d", ch, pool2_out[ch][0][0]);
                $display("DEBUG_CNN: Pool2 samples from all positions:");
                $display("  Pool2[0][0][0:7] = %0d %0d %0d %0d %0d %0d %0d %0d",
                         pool2_out[0][0][0], pool2_out[0][0][1], pool2_out[0][0][2], pool2_out[0][0][3],
                         pool2_out[0][0][4], pool2_out[0][0][5], pool2_out[0][0][6], pool2_out[0][0][7]);
                `endif
            end
            else if (!valid_s3) begin pool2_done<=0; valid_s4<=0; pool2_sent<=0; end
            else if (pool2_sent) valid_s4<=0;
        end
    end

// --- STAGE 5: FC Logic ---
    always @(posedge clk or negedge reset_b) begin
        integer i, ch, y, x; reg signed [63:0] sum, prod;
        reg signed [FC_WEIGHT_WIDTH-1:0] w;
        reg signed [FC_BIAS_WIDTH-1:0] b;
        if (!reset_b) begin
            valid_s5<=0; fc_state<=0; fc_idx<=0; fc_max<=0; fc_min<=32'h7FFFFFFF;
        end else begin
            case (fc_state)
                0: if (valid_s4) begin fc_state<=1; fc_idx<=0; valid_s5<=1; fc_max<=0; fc_min<=32'h7FFFFFFF; end
                1: begin
                    b = get_fc_bias(7'(fc_idx));
                    sum = $signed({{(64-FC_BIAS_WIDTH){b[FC_BIAS_WIDTH-1]}}, b});

                    `ifdef DEBUG_CNN
                    if (fc_idx == 0) begin
                        $display("DEBUG_FC_STATE1: Computing FC[0]");
                        $display("  get_fc_bias(0) = %b (%d) [x=%b]", b, $signed(b), (b === 'x || b === 'X));
                        $display("  Initial sum from bias = %d", sum);
                    end
                    `endif

                    for (i=0; i<FC_INPUT_SIZE; i=i+1) begin
                        ch = i >> 6; y = (i & 63) >> 3; x = i & 7;
                        w = get_fc_weight(7'(fc_idx), 10'(i));
                        prod = $signed(pool2_out[ch][y][x]) * $signed(w);
                        sum = sum + prod;

                        `ifdef DEBUG_CNN
                        if (fc_idx == 0 && i < 3) begin
                            $display("  i=%0d: pool2[%0d][%0d][%0d]=%d, weight=%d [x=%b], prod=%d, sum=%d",
                                     i, ch, y, x, $signed(pool2_out[ch][y][x]), $signed(w),
                                     (w === 'x || w === 'X), $signed(prod), $signed(sum));
                        end
                        `endif
                    end
                    fc_out[7'(fc_idx)] <= 32'(sum >>> FC_SHIFT);
                    if (32'(sum>>>FC_SHIFT) > fc_max) fc_max <= 32'(sum>>>FC_SHIFT);
                    if (32'(sum>>>FC_SHIFT) < fc_min) fc_min <= 32'(sum>>>FC_SHIFT);

                    `ifdef DEBUG_CNN
                    if (fc_idx == 0) $display("DEBUG_TRACE: FC[0] = %d (from sum=%d >> %0d)", sum >>> FC_SHIFT, sum, FC_SHIFT);
                    `endif

                    if (fc_idx == FC_OUT_SIZE-1) begin
                        fc_state<=2;
                        fc_range<=fc_max-fc_min;

                        // Compute per-image adaptive thresholds for 3-level encoding
                        // Threshold 1: min + range/3 (≈33rd percentile)
                        // Threshold 2: min + 2*range/3 (≈67th percentile)
                        // This matches Python's _encode_per_image() logic
                        if ((fc_max - fc_min) == 0) begin
                            // Edge case: All FC outputs are identical (fc_range = 0)
                            // Use sign-based encoding as fallback
                            adaptive_thresh1 <= 0;
                            adaptive_thresh2 <= 32'h7FFFFFFF;  // Large value ensures all zeros for level 2
                        end else begin
                            // Compute thresholds based on ENCODING_LEVELS parameter
                            // For level k: threshold = fc_min + (range * k) / ENCODING_LEVELS
                            // ENCODING_LEVELS=2, k=1: fc_min + range/2 (50th percentile)
                            // ENCODING_LEVELS=3, k=1: fc_min + range/3 (33rd), k=2: fc_min + 2*range/3 (67th)
                            adaptive_thresh1 <= fc_min + ((fc_max - fc_min) * 32'd1) / 32'(ENCODING_LEVELS);
                            adaptive_thresh2 <= fc_min + ((fc_max - fc_min) * 32'd2) / 32'(ENCODING_LEVELS);
                        end
                    end
                    else fc_idx<=fc_idx+1;
                end
                2: begin valid_s5<=0; if (!valid_s4) fc_state<=0; end
            endcase
        end
    end

// --- STAGE 6: HDC Encoding Logic ---
    always @(posedge clk) if (loading_complete && !hdc_thresh_loaded) begin
        hdc_thresh_loaded <= 1;
        `ifdef DEBUG_HDC
        $display("DEBUG_TRACE: Static Thresholds Enabled (Loaded from file).");
        `endif
    end

    always @(posedge clk or negedge reset_b) begin
        integer i, k;
        reg signed [31:0] range, dyn_t;
        reg signed [31:0] threshold;  // Threshold selector for adaptive encoding
        reg signed [31:0] fc_cmp;
        reg signed [63:0] norm_fp;
        reg signed [63:0] mean_fp;
        reg signed [63:0] sumsq_fp;
        reg signed [63:0] mean_sq_fp;
        reg signed [63:0] variance_fp;
        reg signed [31:0] std_fp;
        if (!reset_b) begin 
            valid_s11<=0; 
            hdc_prev_valid<=0;
            for (i=0; i<EXPANDED_FEATURES; i=i+1) binary_features[i] <= 0;
        end
        else begin
            hdc_prev_valid <= valid_s5;
            // Detect falling edge of valid_s5 from FC (state 2 done)
            if (!valid_s5 && hdc_prev_valid) begin 
                valid_s11 <= 1;

                // Per-image normalization stats (positive FC outputs only)
                norm_sum = 0;
                norm_sumsq = 0;
                norm_count = 0;
                for (i=0; i<FC_OUT_SIZE; i=i+1) begin
                    if (fc_out[i] > 0) begin
                        norm_sum = norm_sum + fc_out[i];
                        norm_sumsq = norm_sumsq + ($signed(fc_out[i]) * $signed(fc_out[i]));
                        norm_count = norm_count + 1;
                    end
                end
                if (norm_count > 0) begin
                    // Compute mean and std in fixed-point to preserve small variance
                    mean_fp = ($signed(norm_sum) <<< 16) / $signed(norm_count);   // Q16.16
                    sumsq_fp = ($signed(norm_sumsq) <<< 16) / $signed(norm_count); // Q16.16
                    mean_sq_fp = (mean_fp * mean_fp) >>> 16;                     // Q16.16
                    variance_fp = sumsq_fp - mean_sq_fp;                          // Q16.16
                    if (variance_fp < 0) variance_fp = 0;
                    std_fp = isqrt64(variance_fp[63:0]);                           // Q8.8
                end else begin
                    mean_fp = 0;
                    std_fp = 0;
                end

                norm_test_mean_int = mean_fp >>> 16;
                norm_test_std_int = std_fp >>> 8;

                `ifdef NORM_TRAIN_MEAN
                    if (`NORM_ENABLED && std_fp != 0) begin
                        norm_scale_fp = ($signed(`NORM_TRAIN_STD) <<< 8) / $signed(std_fp); // Q8.8
                        norm_shift_fp = $signed(`NORM_TRAIN_MEAN) - ((mean_fp * $signed(norm_scale_fp)) >>> 16);
                    end else begin
                        norm_scale_fp = 32'sd256;  // 1.0 in Q8.8
                        norm_shift_fp = 64'sd0;
                    end
                `else
                    norm_scale_fp = 32'sd256;
                    norm_shift_fp = 64'sd0;
                `endif
                
                `ifdef DEBUG_HDC
                $display("DEBUG_FC: FC min=%d, max=%d, range=%d", $signed(fc_min), $signed(fc_max), $signed(fc_range));
                $display("DEBUG_ADAPTIVE: Thresh1 (range*1/%0d) = %d", ENCODING_LEVELS, $signed(adaptive_thresh1));
                $display("DEBUG_ADAPTIVE: Thresh2 (range*2/%0d) = %d", ENCODING_LEVELS, $signed(adaptive_thresh2));
                $display("DEBUG_FC: First 10 FC outputs:");
                for (i=0; i<10; i=i+1) begin
                    $display("  fc_out[%0d] = %d", i, $signed(fc_out[i]));
                end
                $display("DEBUG_NORM: count=%0d sum=%0d sumsq=%0d mean=%0d std=%0d", 
                         norm_count, $signed(norm_sum), $signed(norm_sumsq), $signed(norm_test_mean_int), $signed(norm_test_std_int));
                $display("DEBUG_NORM: scale_fp=%0d shift_fp=%0d", $signed(norm_scale_fp), $signed(norm_shift_fp));
                for (i=0; i<8; i=i+1) begin
                    reg signed [63:0] dbg_norm_fp;
                    reg signed [31:0] dbg_fc_cmp;
                    dbg_fc_cmp = fc_out[i];
                    if (`NORM_ENABLED) begin
                        if (fc_out[i] > 0) begin
                            dbg_norm_fp = ($signed(fc_out[i]) * $signed(norm_scale_fp)) + $signed(norm_shift_fp);
                            dbg_fc_cmp = $signed(dbg_norm_fp >>> 8);
                            if (dbg_fc_cmp < 0) dbg_fc_cmp = 0;
                        end else begin
                            dbg_fc_cmp = 0;
                        end
                    end
                    $display("DEBUG_NORM: fc_out[%0d]=%0d norm_fc=%0d", i, $signed(fc_out[i]), $signed(dbg_fc_cmp));
                end
                `endif

                for (i=0; i<FC_OUT_SIZE; i=i+1) begin
                    // Normalize per feature if enabled (matches Python: normalize positives, clamp negatives to 0)
                    fc_cmp = fc_out[i];
                    `ifdef NORM_TRAIN_MEAN
                        if (`NORM_ENABLED) begin
                            if (fc_out[i] > 0) begin
                                norm_fp = ($signed(fc_out[i]) * $signed(norm_scale_fp)) + $signed(norm_shift_fp);
                                fc_cmp = $signed(norm_fp >>> 8);  // back to integer domain
                                if (fc_cmp < 0) fc_cmp = 0;
                            end else begin
                                fc_cmp = 0;
                            end
                        end
                    `endif
                    for (k=1; k<ENCODING_LEVELS; k=k+1) begin
                        // Per-image adaptive thresholds (replaces static thresholds)
                        // Threshold formula: fc_min + (range * k) / ENCODING_LEVELS
                        // ENCODING_LEVELS=2: k=1 uses range/2 (50th percentile)
                        // ENCODING_LEVELS=3: k=1 uses range/3 (33rd), k=2 uses 2*range/3 (67th)
                        // This matches Python's normalization: feat_norm > (k/encoding_levels)

                        if (USE_ADAPTIVE_THRESHOLDS) begin
                            // Select per-image adaptive threshold based on level
                            case (k)
                                1: threshold = adaptive_thresh1;
                                2: threshold = adaptive_thresh2;
                                default: threshold = 0;  // Fallback
                            endcase
                        end else begin
                            // Use static thresholds loaded from file (training-derived)
                            if (USE_PER_FEATURE_THRESHOLDS) begin
                                threshold = $signed(get_feature_thresh(7'(i), 4'(k)));
                            end else begin
                                threshold = $signed(get_thresh(16'(k - 1)));
                            end
                        end

                        // Apply threshold (greater-than to match Python's behavior)
                        if (fc_cmp > threshold) begin
                            binary_features[i + (k-1)*FC_OUT_SIZE] <= 1;
                        end else begin
                            binary_features[i + (k-1)*FC_OUT_SIZE] <= 0;
                        end
                    end
                end
            end else if (valid_s11) begin
                // Print features after they settle
                integer k_cnt, cnt;
                cnt = 0;
                for (k_cnt=0; k_cnt<EXPANDED_FEATURES; k_cnt=k_cnt+1) if (binary_features[k_cnt]) cnt=cnt+1;
                `ifdef DEBUG_HDC
                $display("DEBUG_TRACE: Active Features: %d / %d", cnt, EXPANDED_FEATURES);
                $display("DEBUG_TRACE: First 8 features: %b %b %b %b %b %b %b %b", 
                    binary_features[0], binary_features[1], binary_features[2], binary_features[3],
                    binary_features[4], binary_features[5], binary_features[6], binary_features[7]);
                `endif
                valid_s11 <= 0; 
            end else valid_s11 <= 0;
        end
    end

//======================================================================================
// LFSR PROJECTION: Combinational unroll of PARALLEL_PROJ steps per feature
// When USE_LFSR_PROJECTION=1, each of the 256 features has a 32-bit Fibonacci LFSR.
// Polynomial: x^32 + x^22 + x^2 + x + 1  (taps at bits 31, 21, 1, 0)
// Each step: feedback = state[31]^state[21]^state[1]^state[0]; output = feedback
//            next_state = {state[30:0], feedback}
// Weight mapping: output=1 → +1, output=0 → −1  (matches Python LFSR32 class)
// This block runs every cycle; outputs are only used when proj_state==1.
//======================================================================================
    always @(*) begin
        integer i, j;
        reg [31:0] cs;   // current state (advances through unrolled steps)
        reg fb;           // feedback bit
        for (i = 0; i < EXPANDED_FEATURES; i = i + 1) begin
            cs = lfsr_state[i];
            for (j = 0; j < PARALLEL_PROJ; j = j + 1) begin
                fb = cs[31] ^ cs[21] ^ cs[1] ^ cs[0];
                lfsr_proj_wts[i][j] = fb;       // output = feedback (weight bit)
                cs = {cs[30:0], fb};             // shift left, insert feedback
            end
            lfsr_next_state[i] = cs;             // state after PARALLEL_PROJ steps
        end
    end

//======================================================================================
// STAGE 7: PROJECTION - Random Projection to High-Dimensional Space
//======================================================================================
// Projects binary features from 256-D to 5000-D hypervector space
//   - Input: binary_features[256] (from HDC encoding stage)
//   - Output: projection_sums[5000] (32-bit signed sums)
//   - Projection matrix: 256×5000 with 3-bit weights {-4, -3, -2, -1, 0, 1, 2, 3}
//     OR: on-the-fly ±1 weights via LFSR when USE_LFSR_PROJECTION=1
//   - Operation: For each HV dimension, compute weighted sum of active binary features
//
// State Machine:
//   State 0: Wait for binary features (also resets LFSRs when entering state 1)
//   State 1: Compute projections in parallel (PARALLEL_PROJ dimensions per cycle)
//   State 2: Done, signal Query Gen stage
//
// Key Operation: projection_sums[h] = Σ(binary_features[i] * projection_matrix[i][h])
//                where i ranges over all 256 features
//======================================================================================
    always @(posedge clk or negedge reset_b) begin
        integer j, i; reg signed [31:0] sum; reg [HDC_PROJ_WEIGHT_WIDTH-1:0] proj_val;
        if (!reset_b) begin
            valid_s14<=0; proj_state<=0; proj_done<=0;
            // Initialize all LFSR states to their seeds on reset
            if (USE_LFSR_PROJECTION) begin
                for (i=0; i<EXPANDED_FEATURES; i=i+1)
                    lfsr_state[i] <= LFSR_MASTER_SEED + 32'(i) + 32'd1;
            end
        end
        else begin
            case (proj_state)
                0: if (valid_s11) begin
                    proj_state<=1; proj_idx<=0; valid_s14<=1;
                    // Reset all LFSRs to seeds for this new image
                    if (USE_LFSR_PROJECTION) begin
                        for (i=0; i<EXPANDED_FEATURES; i=i+1)
                            lfsr_state[i] <= LFSR_MASTER_SEED + 32'(i) + 32'd1;
                    end
                end
                1: begin
                    //======================================================================
                    // PARALLEL PROJECTION COMPUTATION
                    // Process PARALLEL_PROJ hypervector dimensions simultaneously
                    // j iterates from 0 to PARALLEL_PROJ-1
                    //======================================================================
                    for (j=0; j<PARALLEL_PROJ; j=j+1) if (32'(proj_idx)+j < HDC_HV_DIM) begin
                        sum = 0;
                        //==================================================================
                        // FEATURE LOOP - Accumulate contributions from active features
                        // i iterates over all binary features [0 to EXPANDED_FEATURES-1]
                        // Only active features (binary_features[i] == 1) contribute
                        // This is a sparse dot product optimization
                        //==================================================================
                        for (i=0; i<EXPANDED_FEATURES; i=i+1) if (binary_features[i]) begin
                            if (USE_LFSR_PROJECTION) begin
                                // LFSR mode: weight from combinational unroll
                                // lfsr_proj_wts[i][j] is 1 for +1, 0 for -1
                                sum = sum + (lfsr_proj_wts[i][j] ? 32'd1 : -32'd1);
                            end else begin
                                // Memory mode: load weight from stored projection matrix
                                proj_val = get_proj(16'(i), 14'(32'(proj_idx)+j));
                                if (HDC_PROJ_WEIGHT_WIDTH == 1) begin
                                    sum = sum + (proj_val[0] ? 32'd1 : -32'd1);
                                end else begin
                                    sum = sum + 32'($signed(proj_val));
                                end
                            end
                        end
                        // Store projection sum for this HV dimension
                        projection_sums[32'(proj_idx)+j] <= sum;
                        `ifdef DEBUG_HDC
                        if (32'(proj_idx)+j == 0) $display("DEBUG_TRACE: ProjSum[0] = %d", sum);
                        `endif
                    end
                    // Advance LFSRs to next batch (state updated after 20 steps)
                    if (USE_LFSR_PROJECTION) begin
                        for (i=0; i<EXPANDED_FEATURES; i=i+1)
                            lfsr_state[i] <= lfsr_next_state[i];
                    end
                    // Advance to next batch of HV dimensions
                    if (32'(proj_idx) + PARALLEL_PROJ >= HDC_HV_DIM) proj_state<=2;
                    else proj_idx <= proj_idx + 16'(PARALLEL_PROJ);
                end
                2: begin
                    if (!proj_done) begin valid_s14<=1; proj_done<=1; end
                    else begin valid_s14<=0; if (!valid_s11) begin proj_state<=0; proj_done<=0; end end
                end
            endcase
        end
    end

//======================================================================================
// STAGE 8: QUERY GENERATION - Binarize Projection Sums to Create Query Hypervector
//======================================================================================
// Converts continuous projection_sums to binary query_hv for classification
//   - Input: projection_sums[5000] (32-bit signed)
//   - Threshold: query_proj_thresh (percentile-based, typically median)
//   - Output: query_hv[5000] (binary: 0 or 1)
//   - Operation: query_hv[i] = (projection_sums[i] > threshold) ? 1 : 0
//
// This creates the final query hypervector that represents the input image
// in the high-dimensional space and will be compared to class prototypes.
//======================================================================================
    always @(posedge clk or negedge reset_b) begin
        if (!reset_b) begin
            query_thresh_loaded <= 0;
            query_proj_thresh <= 0;
        end else if(loading_complete && !query_thresh_loaded) begin
            // Load projection threshold (typically median of projection sums)
            // Projection Threshold is always the last threshold loaded
            query_proj_thresh <= get_thresh(16'(PROJ_THRESH_ID));
            query_thresh_loaded<=1;
            `ifdef DEBUG_HDC
            $display("DEBUG_TRACE: Query Proj Thresh Loaded = %d", get_thresh(16'(PROJ_THRESH_ID)));
            `endif
        end
    end

    always @(posedge clk or negedge reset_b) begin
        integer i;
        if (!reset_b) valid_s15 <= 0;
        else if (valid_s14 && proj_done) begin
            valid_s15 <= 1;
            //==============================================================================
            // BINARIZATION LOOP
            // Iterate over all HV dimensions [0 to HDC_HV_DIM-1] (default: 0-4999)
            // Compare each projection sum to threshold to create binary hypervector
            // Right-shift threshold by 16 to match fixed-point scaling
            //==============================================================================
            for (i=0; i<HDC_HV_DIM; i=i+1) query_hv[i] <= (projection_sums[i] > (query_proj_thresh>>>16)) ? 1'b1 : 1'b0;
        end else valid_s15 <= 0;
    end

//======================================================================================
// STAGE 9: HAMMING DISTANCE - Compute Similarity to Class Prototypes
//======================================================================================
// Calculates Hamming distance between query_hv and each class hypervector
//   - Input: query_hv[5000] (binary)
//   - Class HVs: Stored in memory, accessed via get_hv_bit()
//   - Output: hamming_distances[NUM_CLASSES] (16-bit unsigned)
//   - Operation: hamming_dist[c] = count(query_hv[i] XOR class_hv[c][i])
//
// Hamming distance measures dissimilarity - lower distance = higher similarity
// This is the core HDC classification metric (unlike Euclidean distance in NNs)
//
// State Machine:
//   State 0: Wait for query hypervector
//   State 1: Compute distance to each class sequentially
//   State 2: Done, signal Classification stage
//======================================================================================
    always @(posedge clk or negedge reset_b) begin
        integer i; reg [31:0] sum;
        if (!reset_b) begin valid_s16<=0; ham_state<=0; end
        else case(ham_state)
            0: if (valid_s15) begin ham_state<=1; ham_cls<=0; valid_s16<=1; end
            1: begin
                sum = 0;
                //==============================================================================
                // HAMMING DISTANCE COMPUTATION
                // Iterate over all HV dimensions [0 to HDC_HV_DIM-1] (default: 0-4999)
                // Count number of bit positions where query_hv differs from class_hv
                // XOR operation: (query_hv[i] != get_hv_bit(class, i))
                // This is equivalent to popcount(query_hv XOR class_hv)
                //==============================================================================
                for (i=0; i<HDC_HV_DIM; i=i+1) if (query_hv[i] != get_hv_bit(CLASS_WIDTH'(ham_cls), 14'(i))) sum=sum+1;
                hamming_distances[ham_cls] <= sum[15:0];
                `ifdef DEBUG_HDC
                $display("DEBUG_HAM: Class %d, Sum = %d", ham_cls, sum);
                `endif
                // Advance to next class or finish
                if (ham_cls == NUM_CLASSES-1) ham_state<=2; else ham_cls<=ham_cls+1;
            end
            2: begin valid_s16<=0; if (!valid_s15) ham_state<=0; end
        endcase
    end

//======================================================================================
// STAGE 10: CLASSIFICATION - Select Class with Minimum Hamming Distance
//======================================================================================
// Determines predicted class by finding minimum Hamming distance
//   - Input: hamming_distances[NUM_CLASSES]
//   - Output: predicted_class (argmin of distances)
//   - Output: confidence (from lookup table based on minimum distance)
//
// Operation: predicted_class = argmin_c(hamming_distances[c])
// Lower Hamming distance indicates higher similarity to class prototype
//======================================================================================
    always @(posedge clk or negedge reset_b) begin
        integer i; reg [15:0] min_d; reg [CLASS_WIDTH-1:0] p_cls;
        if (!reset_b) begin valid_s17<=0; class_state<=0; end
        else begin
            if (!valid_s16 && class_state==1) begin // Falling edge of valid_s16
                 //==========================================================================
                 // ARGMIN SEARCH
                 // Initialize with class 0, then iterate to find minimum distance
                 //==========================================================================
                 min_d = hamming_distances[0]; p_cls = 0;
                 `ifdef DEBUG_HDC
                 $display("DEBUG_CLASS: Init min_d=%d, p_cls=0", min_d);
                 `endif
                 //==========================================================================
                 // CLASS COMPARISON LOOP
                 // Iterate over classes [1 to NUM_CLASSES-1]
                 // Update minimum distance and predicted class when lower distance found
                 //==========================================================================
                 for (i=1; i<NUM_CLASSES; i=i+1) begin
                     `ifdef DEBUG_HDC
                     $display("DEBUG_CLASS: Checking Class %d, Dist=%d", i, hamming_distances[i]);
                     `endif
                     if (hamming_distances[i] < min_d) begin
                         min_d = hamming_distances[i]; p_cls = CLASS_WIDTH'(i);
                         `ifdef DEBUG_HDC
                         $display("DEBUG_CLASS: New min_d=%d, p_cls=%d", min_d, p_cls);
                         `endif
                     end
                 end
                 predicted_class <= p_cls;
                 confidence <= get_conf(min_d);
                 valid_s17 <= 1;
                 class_state <= 0; 
            end else if (valid_s16) class_state <= 1;
            else valid_s17 <= 0;
        end
    end

// --- STAGE 11: Online Learning Logic ---
    always @(posedge clk or negedge reset_b) begin
        reg [23:0] thresh_val;
        integer pos_int;
        reg [15:0] lfsr_next;
        reg lfsr_step;
        if (!reset_b) begin ol_state<=0; ol_we<=0; ol_addr<=0; ol_data<=0; ol_idx<=0; lfsr<=16'hACE1; end
        else if (ENABLE_ONLINE_LEARNING) begin
            case(ol_state)
                0: if (valid_s17 && online_learning_enable_reg && loading_complete && confidence >= 8) begin
                     ol_state<=1; ol_idx<=0; ol_we<=0;
                   end
                1: begin
                   thresh_val = 24'((32'(confidence) * 64) << 8);
                   pos_int = HV_START + 32'(predicted_class)*HDC_HV_DIM + 32'(ol_idx);
                   // Match Python LFSR: right shift, taps at bits 0,2,3,5
                   lfsr_next = {lfsr[0]^lfsr[2]^lfsr[3]^lfsr[5], lfsr[15:1]};
                   lfsr_step = 0;
                   if (query_hv[($clog2(HDC_HV_DIM))'(ol_idx)] != loaded_data_mem[pos_int]) begin
                       lfsr_step = 1;
                       if (lfsr_next < thresh_val[15:0]) begin
                           ol_we <= 1; ol_addr <= ($clog2(TOTAL_BITS))'(pos_int); ol_data <= query_hv[($clog2(HDC_HV_DIM))'(ol_idx)];
                       end else ol_we <= 0;
                   end else ol_we <= 0;
                   if (lfsr_step) lfsr <= lfsr_next;
                   
                   if (ol_idx == HDC_HV_DIM-1) ol_state<=2; else ol_idx<=ol_idx+1;
                end
                2: begin ol_we<=0; if(!valid_s17) ol_state<=0; end
            endcase
        end else begin
            ol_we<=0;
        end
    end

endmodule
