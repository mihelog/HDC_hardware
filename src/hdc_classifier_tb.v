//======================================================================================
// hdc_classifier_tb.v - Testbench for HDC Image Classification System
//======================================================================================
//
// DESCRIPTION:
//   Comprehensive self-verifying testbench for the HDC classifier. Loads configuration
//   data (CNN weights and class hypervectors), streams test images, captures predictions,
//   and computes accuracy statistics.
//
// KEY FEATURES:
//   - Serial configuration loading from text files
//   - Cycle-accurate performance measurement
//   - Per-class and overall accuracy statistics
//   - Prediction distribution analysis
//   - Confidence score analysis
//   - Python vs Verilog comparison (agreement percentage)
//   - Timeout detection for pipeline stalls
//
// USAGE:
//   1. Python training script generates: weights_and_hvs.txt, test_images.txt, test_labels.txt
//   2. Testbench loads configuration via serial data_in/write_enable interface
//   3. Testbench processes NUM_TEST_IMAGES images one at a time
//   4. Results printed at end: accuracy, latency, confidence distribution
//
// STATISTICS REPORTED:
//   - Overall Verilog accuracy
//   - Per-class accuracy
//   - Prediction distribution (class balance)
//   - Latency per image (min, max, average cycles)
//   - Confidence distribution
//   - Python vs Verilog agreement (if Python predictions available)
//   - Detailed per-image output (image#, label, prediction, confidence, cycles)
//
// AUTHORS:
//   Developed with assistance from AI tools (Claude and Gemini)
//   Principal Investigator: George Michelogiannakis (mihelog@lbl.gov)
//   Lawrence Berkeley National Laboratory
//
//======================================================================================

`timescale 1ns / 1ps

//======================================================================================
// INCLUDE FILES - Auto-generated parameters from Python training
//======================================================================================
`include "verilog_params/scales.vh"           // Quantization scales for weights
`include "verilog_params/weight_widths.vh"    // Bit widths for CNN weights
`include "verilog_params/shift_params.vh"     // Shift values for overflow prevention

//======================================================================================
// COMPILER DIRECTIVES - Enable hardware optimizations
//======================================================================================
`define INCLUDE_SHIFT_PARAMS              // Use shift parameters for fixed-point
`define INCLUDE_CONFIDENCE_LUT            // Include confidence lookup table

//======================================================================================
// CONFIDENCE LUT SIZE - Must match Python-generated LUT size
//======================================================================================
`define TB_CONFIDENCE_LUT_SIZE 5000       // Confidence LUT entries (distance → confidence)

module hdc_classifier_tb;

//======================================================================================
// DPI FUNCTION IMPORTS (Optional - for advanced file I/O)
// Direct Programming Interface - allows C functions to be called from Verilog
// Only used if USE_DPI is defined during compilation
//======================================================================================
`ifdef USE_DPI
import "DPI-C" function int dpi_open_weights_file(string filename);
import "DPI-C" function int dpi_get_threshold(int thresh_id);
import "DPI-C" function int dpi_read_weight(output int value);
import "DPI-C" function void dpi_get_bit_string(input int value, input int bit_width, output string str);
import "DPI-C" function void dpi_close_file();
import "DPI-C" function void dpi_print_stats();
`endif

//======================================================================================
// TESTBENCH PARAMETERS (Overridable from command line)
// These can be set via +define+PARAM=value or -D PARAM=value during compilation
//======================================================================================
parameter IMG_WIDTH = `ifdef IMG_WIDTH_ARG `IMG_WIDTH_ARG `else 32 `endif;
                                                  // Image width in pixels (default: 32)
parameter IMG_HEIGHT = `ifdef IMG_HEIGHT_ARG `IMG_HEIGHT_ARG `else 32 `endif;
                                                  // Image height in pixels (default: 32)
parameter PIXEL_WIDTH = `ifdef PIXEL_WIDTH_ARG `PIXEL_WIDTH_ARG `else 8 `endif;
                                                  // Bits per pixel (default: 8)
parameter NUM_CLASSES = `ifdef NUM_CLASSES_ARG `NUM_CLASSES_ARG `else 2 `endif;
                                                  // Number of classification classes (default: 2)
parameter HDC_HV_DIM = `ifdef HV_DIM_ARG `HV_DIM_ARG `else 5000 `endif;
                                                  // Hypervector dimension (default: 5000)
parameter HDC_PROJ_WEIGHT_WIDTH = `ifdef PROJ_WEIGHT_WIDTH_ARG `PROJ_WEIGHT_WIDTH_ARG `else 4 `endif;
                                                  // Projection weight bit width (default: 4-bit signed)
parameter CONV1_WEIGHT_WIDTH = `ifdef CONV1_WEIGHT_WIDTH_ARG `CONV1_WEIGHT_WIDTH_ARG `else `CONV1_WEIGHT_WIDTH_VH `endif;
                                                  // Conv1 weight width (from Python-generated params)
parameter CONV2_WEIGHT_WIDTH = `ifdef CONV2_WEIGHT_WIDTH_ARG `CONV2_WEIGHT_WIDTH_ARG `else `CONV2_WEIGHT_WIDTH_VH `endif;
                                                  // Conv2 weight width (from Python-generated params)
parameter FC_WEIGHT_WIDTH = `ifdef FC_WEIGHT_WIDTH_ARG `FC_WEIGHT_WIDTH_ARG `else `FC_WEIGHT_WIDTH_VH `endif;
                                                  // FC weight width (from Python-generated params)
parameter FC_BIAS_WIDTH = `ifdef FC_BIAS_WIDTH_ARG `FC_BIAS_WIDTH_ARG `else `FC_BIAS_WIDTH_VH `endif;
                                                  // FC bias width (from Python-generated params)
parameter ENCODING_LEVELS = `ifdef ENCODING_LEVELS_ARG `ENCODING_LEVELS_ARG `else 4 `endif;
                                                  // HDC encoding levels (default: 4)
parameter USE_PER_FEATURE_THRESHOLDS = `ifdef USE_PER_FEATURE_THRESHOLDS_ARG `USE_PER_FEATURE_THRESHOLDS_ARG `else 1 `endif;
                                                  // 1=per-feature thresholds, 0=global thresholds
parameter USE_LFSR_PROJECTION = `ifdef USE_LFSR_PROJECTION_ARG `USE_LFSR_PROJECTION_ARG `else 0 `endif;
                                                  // 1=on-the-fly LFSR projection, 0=stored matrix
parameter NUM_TEST_IMAGES = `ifdef NUM_TEST_IMAGES `NUM_TEST_IMAGES `else 200 `endif;
                                                  // Number of test images to process
parameter NUM_FEATURES = `ifdef NUM_FEATURES_ARG `NUM_FEATURES_ARG `else 64 `endif;
                                                  // FC layer output size (default: 64)
parameter CLASS_WIDTH = $clog2(NUM_CLASSES);      // Bits needed to represent class ID
parameter HDC_CONF_WIDTH = 4;                     // Confidence score width (4-bit = 0-15)

//======================================================================================
// DERIVED PARAMETERS (Calculated from testbench parameters)
//======================================================================================
// FC input size calculation - dynamic based on image size
localparam POOL2_OUT_DIM = IMG_WIDTH / 4;         // After two 2×2 pooling operations
                                                  // 32 → 16 (pool1) → 8 (pool2)
localparam POOL1_OUT_DIM = IMG_WIDTH / 2;         // After first 2×2 pooling (32 → 16)
localparam CONV1_OUT_CH_TB = 8;                   // Conv1 output channels (fixed at 8)
localparam CONV2_OUT_CH_TB = 16;                  // Conv2 output channels (fixed at 16)
localparam FC_INPUT_SIZE = 16 * POOL2_OUT_DIM * POOL2_OUT_DIM;
                                                  // 16 channels × spatial dimensions
localparam FC_WEIGHT_COUNT = NUM_FEATURES * FC_INPUT_SIZE; // Total FC weights
localparam EXPANDED_FEATURES = NUM_FEATURES * (ENCODING_LEVELS - 1);
                                                  // Binary features from FC output
                                                  // NUM_FEATURES × (levels-1) expansions
localparam FEATURE_THRESH_COUNT = NUM_FEATURES * (ENCODING_LEVELS - 1);
                                                  // Per-feature thresholds for each encoding level
localparam THRESHOLD_COUNT = USE_PER_FEATURE_THRESHOLDS ? (FEATURE_THRESH_COUNT + 1) : ENCODING_LEVELS;
                                                  // Feature thresholds + projection threshold
localparam THRESHOLD_BITS = 32 * THRESHOLD_COUNT;

//======================================================================================
// CONFIGURATION MEMORY SIZE CALCULATION
// Must match hdc_classifier.v memory layout exactly
//======================================================================================
localparam TOTAL_CFG_BITS = (8 * 1 * 3 * 3 * CONV1_WEIGHT_WIDTH) +  // Conv1 weights
                           (8 * CONV1_WEIGHT_WIDTH) +                // Conv1 biases
                           (16 * 8 * 3 * 3 * CONV2_WEIGHT_WIDTH) +   // Conv2 weights
                           (16 * CONV2_WEIGHT_WIDTH) +               // Conv2 biases
                           (NUM_FEATURES * FC_INPUT_SIZE * FC_WEIGHT_WIDTH) + // FC weights (FC_WEIGHT_WIDTH-bit)
                           (NUM_FEATURES * FC_BIAS_WIDTH) +                   // FC biases (8-bit)
                           THRESHOLD_BITS +                           // HDC thresholds (global or per-feature)
                           (USE_LFSR_PROJECTION ? 0 : (EXPANDED_FEATURES * HDC_HV_DIM * HDC_PROJ_WEIGHT_WIDTH)) + // Projection matrix (0 in LFSR mode)
                           (NUM_CLASSES * HDC_HV_DIM) +              // Class hypervectors
                           (`TB_CONFIDENCE_LUT_SIZE * 4) +           // Confidence LUT
                           1;                                        // Online learning enable bit

//======================================================================================
// PIPELINE LATENCY ESTIMATION (for timeout detection)
//======================================================================================
localparam CONV1_CYCLES = IMG_WIDTH * IMG_HEIGHT * 8;  // Conv1: 8 output channels
localparam FC_CYCLES = NUM_FEATURES;                   // FC: variable output features
localparam PROJ_CYCLES = HDC_HV_DIM;                   // Projection: 5000 dimensions
localparam HAM_CYCLES = NUM_CLASSES;                   // Hamming: 2-10 classes
localparam OTHER_CYCLES = 200;                         // Pool, encode, classify, delays

// Calculate timeout based on pipeline stages
// Use 2× expected cycles for safety margin
localparam EXPECTED_CYCLES = CONV1_CYCLES + FC_CYCLES + PROJ_CYCLES + HAM_CYCLES + OTHER_CYCLES;
localparam TIMEOUT_CYCLES = 2 * EXPECTED_CYCLES;       // Abort if pipeline stalls

//======================================================================================
// CLOCK AND RESET GENERATION
//======================================================================================
reg clk = 0;                                           // System clock
reg reset_b = 0;                                       // Active-low asynchronous reset
always #5 clk = ~clk;                                  // 10ns period = 100MHz clock

//======================================================================================
// CYCLE COUNTER - For performance measurement
//======================================================================================
always @(posedge clk) begin
    if (!reset_b)
        cycle_count <= 0;
    else
        cycle_count <= cycle_count + 1;
end

// Track which true label is associated with the current online learning transaction
always @(posedge clk) begin
    if (!reset_b) begin
        ol_label_valid <= 0;
        ol_active_true_label <= 0;
        ol_active_pred_class <= 0;
    end else begin
        if (dut.hdc_classifier_instance.ol_state == 0) begin
            ol_label_valid <= 0;
        end else if (!ol_label_valid && last_label_valid) begin
            ol_active_true_label <= last_true_label;
            ol_active_pred_class <= last_pred_class;
            ol_label_valid <= 1;
        end
    end
end

// Online learning update counters (counts actual memory writes)
always @(posedge clk) begin
    integer ol_i;
    if (!reset_b) begin
        online_learning_updates <= 0;
        for (ol_i = 0; ol_i < NUM_CLASSES; ol_i = ol_i + 1) begin
            online_learning_updates_by_class[ol_i] <= 0;
            online_learning_updates_by_true_label[ol_i] <= 0;
        end
    end else if (loading_complete && dut.hdc_classifier_instance.ol_we) begin
        online_learning_updates <= online_learning_updates + 1;
        online_learning_updates_by_class[predicted_class] <=
            online_learning_updates_by_class[predicted_class] + 1;
        if (ol_label_valid) begin
            online_learning_updates_by_true_label[ol_active_true_label] <=
                online_learning_updates_by_true_label[ol_active_true_label] + 1;
        end
    end
end

//======================================================================================
// DUT (Design Under Test) INTERFACE SIGNALS
//======================================================================================

// --- Input Signals (Driven by Testbench) ---
reg valid = 0;                                         // Asserted to start classification
reg [IMG_WIDTH*IMG_HEIGHT*PIXEL_WIDTH-1:0] image_data = 0;
                                                       // Flattened input image
reg data_in = 0;                                       // Serial configuration data bit
reg write_enable = 0;                                  // Write enable for configuration loading

// Online learning control - set from Makefile ONLINE_LEARNING parameters
`ifndef ENABLE_ONLINE_LEARNING_ARG
    `define ENABLE_ONLINE_LEARNING_ARG 1
`endif
reg online_learning_enable = `ENABLE_ONLINE_LEARNING_ARG; // Enable online learning (default: OFF)
`ifndef ONLINE_LEARNING_IF_CONFIDENCE_HIGH_ARG
    `define ONLINE_LEARNING_IF_CONFIDENCE_HIGH_ARG 0
`endif

// --- Output Signals (Driven by DUT) ---
wire [$clog2(NUM_CLASSES)-1:0] predicted_class;        // Predicted class ID (0 to NUM_CLASSES-1)
wire [3:0] confidence;                                 // Confidence score (0-15, higher = more confident)
wire valid_out;                                        // Asserted when prediction is ready
wire loading_complete;                                 // Asserted when configuration loaded
wire ready;                                            // Asserted when DUT ready for new image

// --- Scan Chain Signals (Currently Unused) ---
// Reserved for future Design-for-Test (DFT) features
wire scan_en = 1'b0;                                   // Scan chain enable (disabled)
wire scan_clk = 1'b0;                                  // Scan chain clock (disabled)
wire scan_in = 1'b0;                                   // Scan chain data input (disabled)
wire scan_out_wire;
wire [7:0] status_out_wire;

// Instantiate DUT - black box interface only
hdc_top #(
    .ROW_SZ(IMG_WIDTH),
    .HEIGHT_SZ(IMG_HEIGHT),
    .ADC_BIT_WIDTH(PIXEL_WIDTH),
    .NUM_CLASSES(NUM_CLASSES),
    .FC_OUT_SIZE(NUM_FEATURES),
    .HDC_HV_DIM(HDC_HV_DIM),
    .HDC_CONF_WIDTH(HDC_CONF_WIDTH),
    .CONFIDENCE_LUT_SIZE(`TB_CONFIDENCE_LUT_SIZE),
    .ENABLE_ONLINE_LEARNING(`ENABLE_ONLINE_LEARNING_ARG),
    .ONLINE_LEARNING_IF_CONFIDENCE_HIGH(`ONLINE_LEARNING_IF_CONFIDENCE_HIGH_ARG),
    .HDC_PROJ_WEIGHT_WIDTH(HDC_PROJ_WEIGHT_WIDTH),
    .ENCODING_LEVELS(ENCODING_LEVELS),
    .USE_PER_FEATURE_THRESHOLDS(USE_PER_FEATURE_THRESHOLDS),
    .USE_LFSR_PROJECTION(USE_LFSR_PROJECTION)
) dut (
    .clk_in(clk),
    .reset_b(reset_b),
    .conf_valid_in(write_enable),
    .scan_en(scan_en),
    .scan_clk(scan_clk),
    .scan_in(scan_in),
    .write_enable(valid),  // Classification start trigger
    .conf_in(data_in),     // Configuration data bit
    .pix_in(image_data),   // Image data
    .scan_out(scan_out_wire),
    .status(status_out_wire),
    .predicted_class(predicted_class),
    .confidence(confidence),
    .valid_out(valid_out),
    .loading_complete(loading_complete),
    .ready(ready)
);

// Test data storage
reg [PIXEL_WIDTH-1:0] test_images [0:NUM_TEST_IMAGES-1][0:IMG_WIDTH*IMG_HEIGHT-1];
reg [CLASS_WIDTH-1:0] test_labels [0:NUM_TEST_IMAGES-1];
reg [CLASS_WIDTH-1:0] python_predictions [0:NUM_TEST_IMAGES-1];  // Python's predictions for comparison
integer actual_loaded_images;  // Track how many images were actually loaded
integer python_predictions_available = 0;  // Flag if Python predictions were loaded

// Performance tracking
integer correct_predictions = 0;
integer total_predictions = 0;

// Python vs Verilog comparison tracking
integer python_verilog_matches = 0;  // Both predict the same class
integer python_verilog_mismatches = 0;  // Different predictions

// Confidence statistics
integer min_confidence;
integer max_confidence;
integer total_confidence;

// Per-class confidence tracking
real class_confidence_sum[0:255];
integer class_confidence_count[0:255];

// Confidence distribution histogram
integer confidence_histogram[0:15];

// Class HV drift tracking (testbench only)
reg initial_class_hv[0:255][0:HDC_HV_DIM-1];
integer hv_drift_count[0:255];

// Loop variable for checksum verification
integer c;
integer header_done;

// Class statistics arrays (up to 256 classes)
integer class_correct[0:255];
integer class_total[0:255];

// Online learning effectiveness tracking - accuracy by decile
integer decile_correct[0:9];
integer decile_total[0:9];
integer online_learning_updates = 0;  // Count of online learning bit updates (ol_we pulses)
integer online_learning_updates_by_class[0:255];
integer online_learning_updates_by_true_label[0:255];
integer last_true_label = 0;
integer last_pred_class = 0;
reg last_label_valid = 0;
integer ol_active_true_label = 0;
integer ol_active_pred_class = 0;
reg ol_label_valid = 0;
integer class_predictions[0:255];

// Dataset name for display
`ifdef DATASET_NAME
    reg [8*20-1:0] dataset_name = `DATASET_NAME;
`else
    // Error: DATASET_NAME must be defined via command line
    // e.g. -DDATASET_NAME="QuickDraw"
    // Failing to declare dataset_name will cause a compilation error
`endif

// Test data files
integer weights_file;
integer test_file;
integer bit_count = 0;
integer weight_value;
integer proj_value;
integer hv_value;
integer i, j, k;
integer idx;  // For main loop
reg [31:0] temp_value;

// Cycle counter for performance measurement
reg [63:0] cycle_count = 0;
reg [63:0] image_start_cycle [0:NUM_TEST_IMAGES-1];
reg [63:0] image_end_cycle [0:NUM_TEST_IMAGES-1];
reg [1023:0] line;  // Increased size to handle long lines in header
integer proj_thresh, fc_thresh1, fc_thresh2;
integer file_pixel_width;  // PIXEL_WIDTH from weights file
integer file_seed;         // SEED from weights file
reg [31:0] thresholds[0:THRESHOLD_COUNT-1];
integer dummy;  // To capture unused return values
`ifdef USE_DPI
string bit_string;  // For DPI bit string output
`endif

// Task to load bits either via slow serial bus or fast backdoor access
task load_bits;
    input [31:0] value;
    input integer count;
    integer j;
    begin
        for (j = 0; j < count; j = j + 1) begin
            `ifdef BACKDOOR_LOAD
                dut.hdc_classifier_instance.loaded_data_mem[bit_count] = value[j];
            `else
                @(negedge clk);
                data_in = value[j];
            `endif
            bit_count = bit_count + 1;
        end
    end
endtask

// Helper tasks to update statistics arrays
task update_class_total;
    input integer class_idx;
    input integer increment;
    integer m;
    begin
        for (m = 0; m < NUM_CLASSES; m = m + 1) begin
            if (m == class_idx) begin
                class_total[m] = class_total[m] + increment;
            end
        end
    end
endtask

task update_class_predictions;
    input integer class_idx;
    input integer increment;
    integer m;
    begin
        for (m = 0; m < NUM_CLASSES; m = m + 1) begin
            if (m == class_idx) begin
                class_predictions[m] = class_predictions[m] + increment;
            end
        end
    end
endtask

task update_class_correct;
    input integer class_idx;
    input integer increment;
    integer m;
    begin
        for (m = 0; m < NUM_CLASSES; m = m + 1) begin
            if (m == class_idx) begin
                class_correct[m] = class_correct[m] + increment;
            end
        end
    end
endtask

// Load test images task
task load_test_images;
    integer m, n;
    reg [PIXEL_WIDTH-1:0] pixel_val;
    reg [CLASS_WIDTH-1:0] label;
    integer scan_result;
    integer actual_num_images;
    reg done_loading;

    begin
        test_file = $fopen("test_images.txt", "r");
        if (test_file == 0) begin
            $display("Error: Could not open test_images.txt");
            $display("Generating synthetic test data instead...");

            // Fallback to synthetic data
            for (m = 0; m < NUM_TEST_IMAGES; m = m + 1) begin
                test_labels[m] = m % NUM_CLASSES;
                for (n = 0; n < IMG_WIDTH*IMG_HEIGHT; n = n + 1) begin
                    test_images[m][n] = $random & 16'hFFFF;
                end
            end
            $display("Generated %d synthetic test images\n", NUM_TEST_IMAGES);
            actual_loaded_images = NUM_TEST_IMAGES;  // All synthetic images are valid
        end else begin
            // File opened successfully, load real test images
            $display("\n=== Loading test images ===");
            $display("Dataset: %s", dataset_name);
            $display("Loading up to %d test images (%d x %d pixels each)...",
                    NUM_TEST_IMAGES, IMG_WIDTH, IMG_HEIGHT);

            actual_num_images = 0;
            done_loading = 0;

            for (m = 0; m < NUM_TEST_IMAGES && !done_loading; m = m + 1) begin
                // Read label
                scan_result = $fscanf(test_file, "%d\n", label);
                if (scan_result != 1) begin
                    $display("  Reached end of file after %d images", m);
                    actual_num_images = m;
                    done_loading = 1;
                end else begin
                    test_labels[m] = label;

                    // Read image pixels
                    for (n = 0; n < IMG_WIDTH*IMG_HEIGHT; n = n + 1) begin
                        scan_result = $fscanf(test_file, "%d\n", pixel_val);
                        if (scan_result != 1) begin
                            $display("ERROR: Failed to read pixel %d for image %d", n, m);
                            $finish;
                        end
                        test_images[m][n] = pixel_val;
                    end

                    // Debug first image
                    if (m == 0) begin
                        $display("\n  First test image:");
                        $display("    Label: %d", label);
                        $display("    First 10 pixels: %d %d %d %d %d %d %d %d %d %d",
                                test_images[0][0], test_images[0][1], test_images[0][2],
                                test_images[0][3], test_images[0][4], test_images[0][5],
                                test_images[0][6], test_images[0][7], test_images[0][8],
                                test_images[0][9]);
                    end

                    actual_num_images = m + 1;
                end
            end

            $fclose(test_file);
            $display("  Loaded %d test images successfully\n", actual_num_images);
            actual_loaded_images = actual_num_images;  // Save for later use
        end
    end
endtask

// Load Python predictions for comparison
task load_python_predictions;
    integer python_file;
    integer m, scan_result;
    integer img_num, py_label, py_pred;
    real py_conf;
    begin
        python_file = $fopen("python_saved_100_predictions.txt", "r");
        if (python_file == 0) begin
            $display("\n=== Python Predictions File Not Found ===");
            $display("File 'python_saved_100_predictions.txt' not available.");
            $display("Skipping Python/Verilog prediction comparison.\n");
            python_predictions_available = 0;
        end else begin
            $display("\n=== Loading Python Predictions ===");
            m = 0;
            while (!$feof(python_file) && m < NUM_TEST_IMAGES) begin
                // Parse: "Image 0: Label=4, Predicted=2, Confidence=0.741"
                scan_result = $fscanf(python_file, "Image %d: Label=%d, Predicted=%d, Confidence=%f\n",
                                     img_num, py_label, py_pred, py_conf);
                if (scan_result == 4) begin
                    python_predictions[img_num] = py_pred;
                    m = m + 1;
                end
            end
            $fclose(python_file);
            $display("  Loaded %d Python predictions successfully\n", m);
            python_predictions_available = 1;
        end
    end
endtask

// Dump internal state for mismatched images (Python vs Verilog)
task dump_mismatch_debug;
    input integer img_idx;
    input integer true_label;
    input integer pred_class;
    input integer py_pred;
    input integer confidence_val;
    integer fh;
    integer i, ch, y, x, ky, kx;
    reg signed [63:0] fc_sum;
    reg signed [31:0] fc_min_v, fc_max_v;
    reg signed [63:0] proj_sum;
    reg signed [31:0] proj_min_v, proj_max_v;
    integer ones_query;
    integer ones_features;
    begin
        fh = $fopen($sformatf("verilog_debug_image_%0d.txt", img_idx), "w");
        if (fh == 0) begin
            $display("WARNING: Could not open debug file for image %0d", img_idx);
            disable dump_mismatch_debug;
        end

        $fdisplay(fh, "Image %0d", img_idx);
        $fdisplay(fh, "Label=%0d Verilog=%0d Python=%0d Confidence=%0d/15", true_label, pred_class, py_pred, confidence_val);

        // FC stats and values
        fc_sum = 0;
        fc_min_v = 32'sh7fffffff;
        fc_max_v = 32'sh80000000;
        for (i = 0; i < NUM_FEATURES; i = i + 1) begin
            if ($signed(dut.hdc_classifier_instance.fc_out[i]) < fc_min_v) fc_min_v = $signed(dut.hdc_classifier_instance.fc_out[i]);
            if ($signed(dut.hdc_classifier_instance.fc_out[i]) > fc_max_v) fc_max_v = $signed(dut.hdc_classifier_instance.fc_out[i]);
            fc_sum = fc_sum + $signed(dut.hdc_classifier_instance.fc_out[i]);
        end
        $fdisplay(fh, "\nFC: min=%0d max=%0d sum=%0d", fc_min_v, fc_max_v, fc_sum);
        $fdisplay(fh, "FC[0..15]: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                  $signed(dut.hdc_classifier_instance.fc_out[0]),  $signed(dut.hdc_classifier_instance.fc_out[1]),
                  $signed(dut.hdc_classifier_instance.fc_out[2]),  $signed(dut.hdc_classifier_instance.fc_out[3]),
                  $signed(dut.hdc_classifier_instance.fc_out[4]),  $signed(dut.hdc_classifier_instance.fc_out[5]),
                  $signed(dut.hdc_classifier_instance.fc_out[6]),  $signed(dut.hdc_classifier_instance.fc_out[7]),
                  $signed(dut.hdc_classifier_instance.fc_out[8]),  $signed(dut.hdc_classifier_instance.fc_out[9]),
                  $signed(dut.hdc_classifier_instance.fc_out[10]), $signed(dut.hdc_classifier_instance.fc_out[11]),
                  $signed(dut.hdc_classifier_instance.fc_out[12]), $signed(dut.hdc_classifier_instance.fc_out[13]),
                  $signed(dut.hdc_classifier_instance.fc_out[14]), $signed(dut.hdc_classifier_instance.fc_out[15]));

        // Thresholds and projection threshold
        $fdisplay(fh, "\nThresholds:");
        $fdisplay(fh, "  thresh[0]=%0d thresh[1]=%0d thresh[2]=%0d",
                  $signed(dut.hdc_classifier_instance.get_thresh(16'd0)),
                  $signed(dut.hdc_classifier_instance.get_thresh(16'd1)),
                  $signed(dut.hdc_classifier_instance.get_thresh(16'd2)));
        $fdisplay(fh, "  proj_thresh_raw=%0d proj_thresh_shifted=%0d",
                  $signed(dut.hdc_classifier_instance.query_proj_thresh),
                  $signed(dut.hdc_classifier_instance.query_proj_thresh >>> 16));

        // Binary feature summary
        ones_features = 0;
        for (i = 0; i < NUM_FEATURES * (ENCODING_LEVELS - 1); i = i + 1) begin
            if (dut.hdc_classifier_instance.binary_features[i]) ones_features = ones_features + 1;
        end
        $fdisplay(fh, "\nBinary features: ones=%0d / %0d", ones_features, NUM_FEATURES * (ENCODING_LEVELS - 1));
        $fdisplay(fh, "Features[0..31]: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                  dut.hdc_classifier_instance.binary_features[0],  dut.hdc_classifier_instance.binary_features[1],
                  dut.hdc_classifier_instance.binary_features[2],  dut.hdc_classifier_instance.binary_features[3],
                  dut.hdc_classifier_instance.binary_features[4],  dut.hdc_classifier_instance.binary_features[5],
                  dut.hdc_classifier_instance.binary_features[6],  dut.hdc_classifier_instance.binary_features[7],
                  dut.hdc_classifier_instance.binary_features[8],  dut.hdc_classifier_instance.binary_features[9],
                  dut.hdc_classifier_instance.binary_features[10], dut.hdc_classifier_instance.binary_features[11],
                  dut.hdc_classifier_instance.binary_features[12], dut.hdc_classifier_instance.binary_features[13],
                  dut.hdc_classifier_instance.binary_features[14], dut.hdc_classifier_instance.binary_features[15],
                  dut.hdc_classifier_instance.binary_features[16], dut.hdc_classifier_instance.binary_features[17],
                  dut.hdc_classifier_instance.binary_features[18], dut.hdc_classifier_instance.binary_features[19],
                  dut.hdc_classifier_instance.binary_features[20], dut.hdc_classifier_instance.binary_features[21],
                  dut.hdc_classifier_instance.binary_features[22], dut.hdc_classifier_instance.binary_features[23],
                  dut.hdc_classifier_instance.binary_features[24], dut.hdc_classifier_instance.binary_features[25],
                  dut.hdc_classifier_instance.binary_features[26], dut.hdc_classifier_instance.binary_features[27],
                  dut.hdc_classifier_instance.binary_features[28], dut.hdc_classifier_instance.binary_features[29],
                  dut.hdc_classifier_instance.binary_features[30], dut.hdc_classifier_instance.binary_features[31]);

        // Projection sum summary
        proj_sum = 0;
        proj_min_v = 32'sh7fffffff;
        proj_max_v = 32'sh80000000;
        for (i = 0; i < HDC_HV_DIM; i = i + 1) begin
            if ($signed(dut.hdc_classifier_instance.projection_sums[i]) < proj_min_v) proj_min_v = $signed(dut.hdc_classifier_instance.projection_sums[i]);
            if ($signed(dut.hdc_classifier_instance.projection_sums[i]) > proj_max_v) proj_max_v = $signed(dut.hdc_classifier_instance.projection_sums[i]);
            proj_sum = proj_sum + $signed(dut.hdc_classifier_instance.projection_sums[i]);
        end
        $fdisplay(fh, "\nProjection sums: min=%0d max=%0d sum=%0d", proj_min_v, proj_max_v, proj_sum);
        $fdisplay(fh, "Proj[0..9]: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                  $signed(dut.hdc_classifier_instance.projection_sums[0]), $signed(dut.hdc_classifier_instance.projection_sums[1]),
                  $signed(dut.hdc_classifier_instance.projection_sums[2]), $signed(dut.hdc_classifier_instance.projection_sums[3]),
                  $signed(dut.hdc_classifier_instance.projection_sums[4]), $signed(dut.hdc_classifier_instance.projection_sums[5]),
                  $signed(dut.hdc_classifier_instance.projection_sums[6]), $signed(dut.hdc_classifier_instance.projection_sums[7]),
                  $signed(dut.hdc_classifier_instance.projection_sums[8]), $signed(dut.hdc_classifier_instance.projection_sums[9]));
        $fdisplay(fh, "\nProjection sums full:");
        for (i = 0; i < HDC_HV_DIM; i = i + 1) begin
            $fdisplay(fh, "  Proj[%0d] = %0d", i, $signed(dut.hdc_classifier_instance.projection_sums[i]));
        end

        // Query HV summary
        ones_query = 0;
        for (i = 0; i < HDC_HV_DIM; i = i + 1) begin
            if (dut.hdc_classifier_instance.query_hv[i]) ones_query = ones_query + 1;
        end
        $fdisplay(fh, "\nQuery HV: ones=%0d / %0d", ones_query, HDC_HV_DIM);
        $fdisplay(fh, "Query[0..31]: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                  dut.hdc_classifier_instance.query_hv[0],  dut.hdc_classifier_instance.query_hv[1],
                  dut.hdc_classifier_instance.query_hv[2],  dut.hdc_classifier_instance.query_hv[3],
                  dut.hdc_classifier_instance.query_hv[4],  dut.hdc_classifier_instance.query_hv[5],
                  dut.hdc_classifier_instance.query_hv[6],  dut.hdc_classifier_instance.query_hv[7],
                  dut.hdc_classifier_instance.query_hv[8],  dut.hdc_classifier_instance.query_hv[9],
                  dut.hdc_classifier_instance.query_hv[10], dut.hdc_classifier_instance.query_hv[11],
                  dut.hdc_classifier_instance.query_hv[12], dut.hdc_classifier_instance.query_hv[13],
                  dut.hdc_classifier_instance.query_hv[14], dut.hdc_classifier_instance.query_hv[15],
                  dut.hdc_classifier_instance.query_hv[16], dut.hdc_classifier_instance.query_hv[17],
                  dut.hdc_classifier_instance.query_hv[18], dut.hdc_classifier_instance.query_hv[19],
                  dut.hdc_classifier_instance.query_hv[20], dut.hdc_classifier_instance.query_hv[21],
                  dut.hdc_classifier_instance.query_hv[22], dut.hdc_classifier_instance.query_hv[23],
                  dut.hdc_classifier_instance.query_hv[24], dut.hdc_classifier_instance.query_hv[25],
                  dut.hdc_classifier_instance.query_hv[26], dut.hdc_classifier_instance.query_hv[27],
                  dut.hdc_classifier_instance.query_hv[28], dut.hdc_classifier_instance.query_hv[29],
                  dut.hdc_classifier_instance.query_hv[30], dut.hdc_classifier_instance.query_hv[31]);
        $fdisplay(fh, "\nQuery HV full:");
        for (i = 0; i < HDC_HV_DIM; i = i + 1) begin
            $fdisplay(fh, "  Query[%0d] = %0d", i, dut.hdc_classifier_instance.query_hv[i]);
        end

        // Hamming distances
        $fdisplay(fh, "\nHamming distances:");
        for (i = 0; i < NUM_CLASSES; i = i + 1) begin
            $fdisplay(fh, "  class %0d: %0d", i, dut.hdc_classifier_instance.hamming_distances[i]);
        end

        // Conv1 weights/bias for all channels (for direct comparison with Python)
        $fdisplay(fh, "\nConv1 weights (och, ky, kx):");
        for (ch = 0; ch < CONV1_OUT_CH_TB; ch = ch + 1) begin
            for (ky = 0; ky < 3; ky = ky + 1) begin
                for (kx = 0; kx < 3; kx = kx + 1) begin
                    $fdisplay(fh, "  W[%0d][%0d][%0d] = %0d",
                              ch, ky, kx, $signed(dut.hdc_classifier_instance.get_conv1_weight(5'(ch), 5'd0, 3'(ky), 3'(kx))));
                end
            end
        end
        $fdisplay(fh, "Conv1 bias ch0 = %0d", $signed(dut.hdc_classifier_instance.get_conv1_bias(5'd0)));

        // Full Conv1 dump for detailed diffing (8x32x32)
        $fdisplay(fh, "\nConv1 full:");
        for (ch = 0; ch < CONV1_OUT_CH_TB; ch = ch + 1) begin
            for (y = 0; y < IMG_HEIGHT; y = y + 1) begin
                for (x = 0; x < IMG_WIDTH; x = x + 1) begin
                    $fdisplay(fh, "  Conv1[%0d][%0d][%0d] = %0d",
                              ch, y, x, dut.hdc_classifier_instance.conv1_out[ch][y][x]);
                end
            end
        end

        // Full Pool1 dump for detailed diffing (8x16x16)
        $fdisplay(fh, "\nPool1 full:");
        for (ch = 0; ch < CONV1_OUT_CH_TB; ch = ch + 1) begin
            for (y = 0; y < POOL1_OUT_DIM; y = y + 1) begin
                for (x = 0; x < POOL1_OUT_DIM; x = x + 1) begin
                    $fdisplay(fh, "  Pool1[%0d][%0d][%0d] = %0d",
                              ch, y, x, dut.hdc_classifier_instance.pool1_out[ch][y][x]);
                end
            end
        end

        // Pool2 sample (sanity check for CNN path)
        $fdisplay(fh, "\nPool2 samples:");
        $fdisplay(fh, "  Pool2[0][0][0..7] = %0d %0d %0d %0d %0d %0d %0d %0d",
                  dut.hdc_classifier_instance.pool2_out[0][0][0], dut.hdc_classifier_instance.pool2_out[0][0][1],
                  dut.hdc_classifier_instance.pool2_out[0][0][2], dut.hdc_classifier_instance.pool2_out[0][0][3],
                  dut.hdc_classifier_instance.pool2_out[0][0][4], dut.hdc_classifier_instance.pool2_out[0][0][5],
                  dut.hdc_classifier_instance.pool2_out[0][0][6], dut.hdc_classifier_instance.pool2_out[0][0][7]);
        $fdisplay(fh, "  Pool2[1][0][0] = %0d", dut.hdc_classifier_instance.pool2_out[1][0][0]);
        $fdisplay(fh, "  Pool2[15][3][3] = %0d", dut.hdc_classifier_instance.pool2_out[15][3][3]);

        // Full Pool2 dump for detailed diffing (16x8x8)
        $fdisplay(fh, "\nPool2 full:");
        for (ch = 0; ch < CONV2_OUT_CH_TB; ch = ch + 1) begin
            for (y = 0; y < POOL2_OUT_DIM; y = y + 1) begin
                for (x = 0; x < POOL2_OUT_DIM; x = x + 1) begin
                    $fdisplay(fh, "  Pool2[%0d][%0d][%0d] = %0d",
                              ch, y, x, dut.hdc_classifier_instance.pool2_out[ch][y][x]);
                end
            end
        end

        $fclose(fh);
    end
endtask

// Reciprocal LUT verification removed - hardware now uses division directly

// Process single image task
task process_image;
    input integer img_idx;
    integer m, n;
    integer timeout_counter;
    integer image_latency;
    reg found_valid_out;
    real confidence_normalized;
    integer true_label;
    integer pred_class;

    begin
        `ifndef QUIET_RESULTS
        $display("\nProcessing image %d...", img_idx);
        `endif

        // Get the true label
        true_label = test_labels[img_idx];

        // Clear image data first
        image_data = 0;

        // Load image data
        for (m = 0; m < IMG_HEIGHT; m = m + 1) begin
            for (n = 0; n < IMG_WIDTH; n = n + 1) begin
                image_data[(m*IMG_WIDTH+n)*PIXEL_WIDTH +: PIXEL_WIDTH] = test_images[img_idx][m*IMG_WIDTH+n];
            end
        end

        // Debug: print first 10 pixels and checksum
        `ifdef DEBUG_CNN
        $display("DEBUG_TB: Image %0d first 10 pixels: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 img_idx, test_images[img_idx][0], test_images[img_idx][1], test_images[img_idx][2],
                 test_images[img_idx][3], test_images[img_idx][4], test_images[img_idx][5],
                 test_images[img_idx][6], test_images[img_idx][7], test_images[img_idx][8],
                 test_images[img_idx][9]);
        $display("DEBUG_TB: Image %0d pixels[100:105]: %0d %0d %0d %0d %0d %0d",
                 img_idx, test_images[img_idx][100], test_images[img_idx][101], test_images[img_idx][102],
                 test_images[img_idx][103], test_images[img_idx][104], test_images[img_idx][105]);
        `endif

        // Wait for pipeline to be ready
        timeout_counter = 0;
        while (!ready && timeout_counter < 20000) begin
            @(posedge clk);
            timeout_counter = timeout_counter + 1;
        end

        if (!ready) begin
            $display("ERROR: Pipeline not ready after %d cycles!", timeout_counter);
            $finish;
        end

        // Apply valid signal
        @(negedge clk);  // Change on negedge for clean timing
        valid = 1;
        @(posedge clk);  // Hardware processes on this edge
        image_start_cycle[img_idx] = cycle_count;  // Record start time
        @(negedge clk);  // Clear on negedge
        valid = 0;

        // Wait for result with timeout
        timeout_counter = 0;
        found_valid_out = 0;

        while (!found_valid_out && timeout_counter < TIMEOUT_CYCLES) begin
            @(posedge clk);

            // Check for valid_out
            if (valid_out) begin
                found_valid_out = 1;
                image_end_cycle[img_idx] = cycle_count;  // Record end time
                pred_class = predicted_class;

                // Calculate latency (input to output delay in clock cycles)
                image_latency = image_end_cycle[img_idx] - image_start_cycle[img_idx];
                // Record last label/prediction for online learning attribution
                last_true_label = true_label;
                last_pred_class = pred_class;
                last_label_valid = 1;

                // Update statistics
                total_predictions = total_predictions + 1;

                // DEBUG: Print Hamming Distances
                `ifndef QUIET_RESULTS
                $display("    Hamming Distances:");
                for (m = 0; m < NUM_CLASSES; m = m + 1) begin
                   $display("      Class %d: %d", m, dut.hdc_classifier_instance.hamming_distances[m]);
                end
                `endif

                // Update class statistics
                update_class_total(true_label, 1);
                update_class_predictions(pred_class, 1);

                if (pred_class == true_label) begin
                    correct_predictions = correct_predictions + 1;
                    update_class_correct(true_label, 1);
                end

                // Update decile statistics (divide images into 10 groups)
                begin
                    integer decile_idx;
                    decile_idx = (img_idx * 10) / actual_loaded_images;
                    if (decile_idx > 9) decile_idx = 9;  // Clamp to valid range
                    decile_total[decile_idx] = decile_total[decile_idx] + 1;
                    if (pred_class == true_label) begin
                        decile_correct[decile_idx] = decile_correct[decile_idx] + 1;
                    end
                end

                // Update confidence statistics
                total_confidence = total_confidence + confidence;
                if (confidence < min_confidence) begin
                    min_confidence = confidence;
                end
                if (confidence > max_confidence) begin
                    max_confidence = confidence;
                end

                // Update per-class confidence statistics
                class_confidence_sum[true_label] = class_confidence_sum[true_label] + confidence;
                class_confidence_count[true_label] = class_confidence_count[true_label] + 1;

                // Update confidence histogram
                confidence_histogram[confidence] = confidence_histogram[confidence] + 1;

                // Calculate normalized confidence (0-1 range)
                confidence_normalized = confidence / 15.0;

                // Compare with Python prediction if available
                if (python_predictions_available) begin
                    if (python_predictions[img_idx] == pred_class) begin
                        // Verilog and Python agree
                        python_verilog_matches = python_verilog_matches + 1;
                        `ifndef QUIET_RESULTS
                        $display("  Image %10d: Label=%10d, Predicted=%10d, Confidence=%d/15 (%.2f), Latency=%0d cycles, %s [Python agrees: %d]",
                                 img_idx, true_label, pred_class, confidence,
                                 confidence_normalized, image_latency,
                                 (pred_class == true_label) ? "CORRECT" : "  WRONG",
                                 python_predictions[img_idx]);
                        `endif
                    end else begin
                        // Verilog and Python DISAGREE - this is the key issue!
                        python_verilog_mismatches = python_verilog_mismatches + 1;
                        `ifndef QUIET_RESULTS
                        $display("  Image %10d: Label=%10d, Verilog=%10d, Python=%10d, Latency=%0d cycles, %s [MISMATCH! Verilog Confidence=%d/15 (%.2f)]",
                                 img_idx, true_label, pred_class, python_predictions[img_idx],
                                 image_latency,
                                 (pred_class == true_label) ? "CORRECT" : "  WRONG",
                                 confidence, confidence_normalized);
                        `endif
                        `ifdef DEBUG_MISMATCH_DUMP
                            dump_mismatch_debug(img_idx, true_label, pred_class, python_predictions[img_idx], confidence);
                        `endif
                    end
                end else begin
                    `ifndef QUIET_RESULTS
                    $display("  Image %10d: Label=%10d, Predicted=%10d, Confidence=%d/15 (%.2f), Latency=%0d cycles, %s",
                             img_idx, true_label, pred_class, confidence,
                             confidence_normalized, image_latency,
                             (pred_class == true_label) ? "CORRECT" : "  WRONG");
                    `endif
                end

                // Online learning updates are counted globally via ol_we pulses.
            end

            timeout_counter = timeout_counter + 1;
        end

        if (!found_valid_out) begin
            $display("ERROR: Timeout waiting for valid_out on image %d after %d/%d cycles",
                    img_idx, timeout_counter, TIMEOUT_CYCLES);
            $finish;
        end
    end
endtask

// Test sequence
initial begin
    // VCD dump for waveform viewing
`ifdef DUMP_WAVEFORMS
    $dumpfile("hdc_classifier.vcd");
    $dumpvars(0, hdc_classifier_tb);
`endif

    // Initialize statistics
    min_confidence = 15;
    max_confidence = 0;
    total_confidence = 0;

    // Initialize all arrays
    for (i = 0; i < 256; i = i + 1) begin
        class_correct[i] = 0;
        class_total[i] = 0;
        class_predictions[i] = 0;
        class_confidence_sum[i] = 0.0;
        class_confidence_count[i] = 0;
    end

    // Initialize decile tracking
    for (i = 0; i < 10; i = i + 1) begin
        decile_correct[i] = 0;
        decile_total[i] = 0;
    end
    online_learning_updates = 0;

    // Initialize confidence histogram
    for (i = 0; i <= 15; i = i + 1) begin
        confidence_histogram[i] = 0;
    end

    // Initialize test data arrays to prevent undefined values
    actual_loaded_images = 0;  // Will be set by load_test_images
    for (i = 0; i < NUM_TEST_IMAGES; i = i + 1) begin
        test_labels[i] = 0;
        for (j = 0; j < IMG_WIDTH*IMG_HEIGHT; j = j + 1) begin
            test_images[i][j] = 0;
        end
    end

    $display("\n========================================");
    $display("HDC Classifier Testbench Started");
    $display("========================================");
    $display("Dataset: %s", dataset_name);
    $display("Parameters:");
    $display("  Image size: %d x %d", IMG_WIDTH, IMG_HEIGHT);
    $display("  Number of classes: %d", NUM_CLASSES);
    $display("  Hypervector dimension: %d", HDC_HV_DIM);
    $display("  Number of test images: %d", NUM_TEST_IMAGES);
    $display("  Calculated FC_WEIGHT_COUNT: %d", FC_WEIGHT_COUNT);
    $display("  Calculated FC_INPUT_SIZE: %d", FC_INPUT_SIZE);

    // Moved from hdc_classifier.v
    $display("HDC_CLASSIFIER INIT: IMG_WIDTH=%d, TOTAL_BITS=%d", IMG_WIDTH, TOTAL_CFG_BITS);

    $display("  Configuration Memory Size:");
    $display("    Total Bits:  %d", TOTAL_CFG_BITS);
    $display("    Total Bytes: %d", (TOTAL_CFG_BITS + 7) / 8);

`ifdef USE_DPI
    $display("  File I/O Mode: DPI (Fast C-based I/O)");
`else
    $display("  File I/O Mode: Verilog $fscanf (Standard)");
`endif
    $display("========================================\n");

    // Reset sequence
    #20;
    reset_b = 1;
    #20;

    // Open file and load thresholds
`ifdef USE_DPI
    // DPI-based file opening and threshold loading
    if (!dpi_open_weights_file("weights_and_hvs.txt")) begin
        $display("ERROR: DPI cannot open weights_and_hvs.txt");
        $finish;
    end
    $display("[TB] File opened via DPI");

    // Read thresholds from header for logging only (stream load handles actual values)
    proj_thresh = dpi_get_threshold(0);
    fc_thresh1 = dpi_get_threshold(1);
    fc_thresh2 = dpi_get_threshold(2);

    // PIXEL_WIDTH validation for DPI path - read file directly for validation
    test_file = $fopen("weights_and_hvs.txt", "r");
    if (test_file != 0) begin
        // Skip first 3 lines, read PIXEL_WIDTH from line 4
        dummy = $fgets(line, test_file); // IMG_SIZE
        dummy = $fgets(line, test_file); // NUM_CLASSES
        dummy = $fgets(line, test_file); // HDC_HV_DIM
        dummy = $fgets(line, test_file); // PIXEL_WIDTH
        dummy = $sscanf(line, "PIXEL_WIDTH %d", file_pixel_width);
        
        // Skip next few lines to get SEED at line 13
        for (k = 0; k < 8; k = k + 1) dummy = $fgets(line, test_file);
        dummy = $sscanf(line, "SEED %d", file_seed);
        $display("[TB] Random Seed used for generation: %d", file_seed);
        
        $fclose(test_file);

        // Validate PIXEL_WIDTH consistency
        $display("[TB] File PIXEL_WIDTH: %d, Testbench PIXEL_WIDTH: %d", file_pixel_width, PIXEL_WIDTH);
        if (file_pixel_width != PIXEL_WIDTH) begin
            $display("========================================");
            $display("ERROR: PIXEL_WIDTH MISMATCH DETECTED!");
            $display("========================================");
            $display("Python generated weights file with PIXEL_WIDTH = %d", file_pixel_width);
            $display("Verilog testbench compiled with PIXEL_WIDTH = %d", PIXEL_WIDTH);
            $display("");
            $display("This mismatch causes incorrect pixel interpretation:");
            $display("- Python saves %d-bit pixel values", file_pixel_width);
            $display("- Verilog reads them as %d-bit values", PIXEL_WIDTH);
            $display("");
            $display("Solution: Recompile with correct PIXEL_WIDTH:");
            $display("  make simulate PIXEL_WIDTH=%d", file_pixel_width);
            $display("  or update makefile: PIXEL_WIDTH = %d", file_pixel_width);
            $display("========================================");
            $finish;
        end
        $display("[TB] PIXEL_WIDTH validation passed: %d bits", PIXEL_WIDTH);
    end else begin
        $display("WARNING: Could not validate PIXEL_WIDTH (file read failed)");
    end
`else
    // Traditional Verilog file I/O
    weights_file = $fopen("weights_and_hvs.txt", "r");
    if (weights_file == 0) begin
        $display("ERROR: Cannot open weights_and_hvs.txt");
        $finish;
    end
    $display("[TB] File opened via Verilog $fopen");

    // Parse header and extract thresholds robustly
    // Keep reading until we find a line starting with a digit or minus sign (data)
    header_done = 0;
    while (header_done == 0) begin
        c = $fgetc(weights_file);
        if (c == -1) begin
            header_done = 1; // EOF
        end else if ((c >= 48 && c <= 57) || c == 45) begin
            // Digit or minus sign. Start of data.
            dummy = $ungetc(c, weights_file); // Put it back
            header_done = 1;
        end else begin
            // It's a header line, read it
            dummy = $ungetc(c, weights_file);
            dummy = $fgets(line, weights_file);
            
            // Parse parameters of interest
            if ($sscanf(line, "PIXEL_WIDTH %d", file_pixel_width) == 1) begin end
            else if ($sscanf(line, "SEED %d", file_seed) == 1) begin end
            else if ($sscanf(line, "PROJECTION_THRESHOLD %d", proj_thresh) == 1) begin end
            else if ($sscanf(line, "FC_THRESH1 %d", fc_thresh1) == 1) begin end
            else if ($sscanf(line, "FC_THRESH2 %d", fc_thresh2) == 1) begin end
        end
    end

    $display("[TB] Random Seed used for generation: %d", file_seed);

    // CRITICAL: Validate PIXEL_WIDTH parameter consistency
    $display("[TB] File PIXEL_WIDTH: %d, Testbench PIXEL_WIDTH: %d", file_pixel_width, PIXEL_WIDTH);
    if (file_pixel_width != PIXEL_WIDTH) begin
        $display("========================================");
        $display("ERROR: PIXEL_WIDTH MISMATCH DETECTED!");
        $display("========================================");
        $display("Python generated weights file with PIXEL_WIDTH = %d", file_pixel_width);
        $display("Verilog testbench compiled with PIXEL_WIDTH = %d", PIXEL_WIDTH);
        $display("");
        $display("This mismatch causes incorrect pixel interpretation:");
        $display("- Python saves %d-bit pixel values", file_pixel_width);
        $display("- Verilog reads them as %d-bit values", PIXEL_WIDTH);
        $display("");
        $display("Solution: Recompile with correct PIXEL_WIDTH:");
        $display("  make simulate PIXEL_WIDTH=%d", file_pixel_width);
        $display("  or update makefile: PIXEL_WIDTH = %d", file_pixel_width);
        $display("========================================");
        $finish;
    end
    $display("[TB] PIXEL_WIDTH validation passed: %d bits", PIXEL_WIDTH);
`endif
    $display("[TB] Header thresholds: FC1=%d, FC2=%d, PROJ=%d",
             fc_thresh1, fc_thresh2, proj_thresh);

    $display("[TB] Starting data loading...");
`ifdef BACKDOOR_LOAD
    $display("[TB] Load mode: BACKDOOR (direct memory write, serial load skipped)");
    // Prevent DUT load logic from clobbering backdoor-written bits
    write_enable = 0;
`else
    $display("[TB] Load mode: SERIAL (bit-by-bit via data_in/write_enable)");
    write_enable = 1;
`endif

    // Load Conv1 weights (72 values, CONV1_WEIGHT_WIDTH bits each)
    for (i = 0; i < 72; i = i + 1) begin
`ifdef USE_DPI
        if (dpi_read_weight(weight_value)) begin
`else
        if ($fscanf(weights_file, "%d", weight_value) == 1) begin
`endif
            temp_value = weight_value;

            // Debug: Print text value and binary representation
            if (i < 5 || i == 71) begin
                $display("[TB] Conv1 weight[%d]:", i);
                $display("      Text value: %d (signed)", $signed(weight_value));
                $display("      Binary (%0d-bit): %b", CONV1_WEIGHT_WIDTH, temp_value[CONV1_WEIGHT_WIDTH-1:0]);
                $write("      Bit stream (LSB first): ");
`ifdef USE_DPI
                dpi_get_bit_string(weight_value, CONV1_WEIGHT_WIDTH, bit_string);
                $display("%s", bit_string);
`else
                for (j = 0; j < CONV1_WEIGHT_WIDTH; j = j + 1)
                    $write("%b", temp_value[j]);
                $display("");
`endif
            end

            // Load bits LSB first
            load_bits(temp_value, CONV1_WEIGHT_WIDTH);
        end else begin
            $display("ERROR: Failed to read Conv1 weight %d", i);
            $finish;
        end
    end

    // Load Conv1 bias (8 values, CONV1_WEIGHT_WIDTH bits each)
    for (i = 0; i < 8; i = i + 1) begin
`ifdef USE_DPI
        if (dpi_read_weight(weight_value)) begin
`else
        if ($fscanf(weights_file, "%d", weight_value) == 1) begin
`endif
            temp_value = weight_value;
            load_bits(temp_value, CONV1_WEIGHT_WIDTH);
            if (i < 2) $display("[TB] Conv1 bias[%d] = %d", i, weight_value);
        end
    end

    // Load Conv2 weights (1152 values, CONV2_WEIGHT_WIDTH bits each)
    for (i = 0; i < 1152; i = i + 1) begin
`ifdef USE_DPI
        if (dpi_read_weight(weight_value)) begin
`else
        if ($fscanf(weights_file, "%d", weight_value) == 1) begin
`endif
            temp_value = weight_value;
            load_bits(temp_value, CONV2_WEIGHT_WIDTH);
            if (i < 3) $display("[TB] Conv2 weight[%d] = %d", i, weight_value);
        end
    end

    // Load Conv2 bias (16 values, CONV2_WEIGHT_WIDTH bits each)
    for (i = 0; i < 16; i = i + 1) begin
`ifdef USE_DPI
        if (dpi_read_weight(weight_value)) begin
`else
        if ($fscanf(weights_file, "%d", weight_value) == 1) begin
`endif
            temp_value = weight_value;
            load_bits(temp_value, CONV2_WEIGHT_WIDTH);
            if (i < 2) $display("[TB] Conv2 bias[%d] = %d", i, weight_value);
        end
    end

    // Load FC weights
    for (i = 0; i < FC_WEIGHT_COUNT; i = i + 1) begin
`ifdef USE_DPI
        if (dpi_read_weight(weight_value)) begin
`else
        if ($fscanf(weights_file, "%d", weight_value) == 1) begin
`endif
            temp_value = weight_value;

            // Debug: Print text value and binary for first few and samples
            if (i < 3 || (i % 5000) == 0) begin
                $display("[TB] FC weight[%d]:", i);
                $display("      Text value: %d (signed)", $signed(weight_value));
                $display("      Binary (%0d-bit): %b", FC_WEIGHT_WIDTH, temp_value[FC_WEIGHT_WIDTH-1:0]);
                $write("      Bit stream (LSB first): ");
                for (j = 0; j < FC_WEIGHT_WIDTH; j = j + 1)
                    $write("%b", temp_value[j]);
                $display(" (total bits: %d)", bit_count + FC_WEIGHT_WIDTH);
                $fflush();
                #1; // Yield to simulator to avoid hang detection
            end

            load_bits(temp_value, FC_WEIGHT_WIDTH);
        end
    end

    // Load FC bias (NUM_FEATURES values, FC_BIAS_WIDTH bits each)
    for (i = 0; i < NUM_FEATURES; i = i + 1) begin
`ifdef USE_DPI
        if (dpi_read_weight(weight_value)) begin
`else
        if ($fscanf(weights_file, "%d", weight_value) == 1) begin
`endif
            temp_value = weight_value;
            load_bits(temp_value, FC_BIAS_WIDTH);  // Use FC_BIAS_WIDTH (8-bit) for biases
            if (i < 2) $display("[TB] FC bias[%d] = %d", i, weight_value);
        end
    end

    // Load thresholds from file stream (global or per-feature)
    $display("[TB] Loading %0d thresholds (ENCODING_LEVELS=%0d, per-feature=%0d)...",
             THRESHOLD_COUNT, ENCODING_LEVELS, USE_PER_FEATURE_THRESHOLDS);
    for (i = 0; i < THRESHOLD_COUNT; i = i + 1) begin
`ifdef USE_DPI
        if (dpi_read_weight(weight_value)) begin
`else
        if ($fscanf(weights_file, "%d", weight_value) == 1) begin
`endif
            temp_value = weight_value;
            thresholds[i] = temp_value;
            load_bits(temp_value, 32);
            if (i < 4 || i == (THRESHOLD_COUNT - 1)) begin
                if (i == (THRESHOLD_COUNT - 1))
                    $display("[TB] Loaded Projection Threshold = %d", temp_value);
                else
                    $display("[TB] Loaded Feature Threshold[%0d] = %d", i, temp_value);
            end
        end else begin
            $display("ERROR: Failed to read threshold %d", i);
            $finish;
        end
    end

    $display("[TB] CNN weights and thresholds loading complete: %d total bits", bit_count);

    // Load projection matrix (skip in LFSR mode — generated on-the-fly)
    if (USE_LFSR_PROJECTION == 0) begin
        for (i = 0; i < dut.hdc_classifier_instance.PROJ_MATRIX_ROWS * dut.hdc_classifier_instance.PROJ_MATRIX_COLS; i = i + 1) begin
`ifdef USE_DPI
            if (dpi_read_weight(proj_value)) begin
`else
            if ($fscanf(weights_file, "%d", proj_value) == 1) begin
`endif
                temp_value = proj_value;
                load_bits(temp_value, HDC_PROJ_WEIGHT_WIDTH);

                if ((i % 100000) == 0) begin
                    $display("[TB] Loaded projection value %d (total_bits=%d)", i, bit_count);
                end
            end else begin
                $display("ERROR: Failed to read projection value %d", i);
                $finish;
            end
        end
        $display("[TB] Projection matrix loading complete: %d total bits", bit_count);
    end else begin
        $display("[TB] LFSR projection enabled — skipping projection matrix load");
    end

    // Load hypervectors
    for (i = 0; i < dut.hdc_classifier_instance.NUM_CLASSES * dut.hdc_classifier_instance.HDC_HV_DIM; i = i + 1) begin
`ifdef USE_DPI
        if (dpi_read_weight(hv_value)) begin
`else
        if ($fscanf(weights_file, "%d", hv_value) == 1) begin
`endif
            load_bits(hv_value, 1);

            if ((i % 10000) == 0) begin
                $display("[TB] Loaded HV bit %d (total_bits=%d)", i, bit_count);
            end
        end else begin
            $display("ERROR: Failed to read HV bit %d", i);
            $finish;
        end
    end

    // Load confidence LUT
    $display("\n[TESTBENCH] Loading confidence LUT...");
    for (i = 0; i < dut.hdc_classifier_instance.CONFIDENCE_LUT_SIZE; i = i + 1) begin
`ifdef USE_DPI
        if (dpi_read_weight(temp_value)) begin
`else
        if ($fscanf(weights_file, "%d", temp_value) == 1) begin
`endif
            load_bits(temp_value, 4);

            if ((i % 100) == 0) begin
                $display("[TB] Loaded confidence LUT entry %d, value=%d (total_bits=%d)", i, temp_value, bit_count);
            end
        end else begin
            $display("ERROR: Failed to read confidence LUT entry %d", i);
            $finish;
        end
    end
    $display("[TESTBENCH] Loaded %0d confidence LUT entries", `TB_CONFIDENCE_LUT_SIZE);

    // Reciprocal LUT loading removed - hardware now uses division directly
    $display("[TESTBENCH] (Reciprocal LUT removed - using division)");

    // ============================================================================
    // 11. ONLINE LEARNING ENABLE BIT
    // ============================================================================
    $display("[TB] Loading online learning enable bit (bit %0d)...", bit_count);

    // Load OL enable bit (0=disabled, 1=enabled)
    // Value comes from ONLINE_LEARNING makefile parameter
    load_bits(online_learning_enable, 1);

    $display("[TB]   Online learning enable: %0d (%s)", online_learning_enable,
             online_learning_enable ? "enabled" : "disabled");
    $display("[TB] Online learning enable bit loaded. Total bits: %0d", bit_count);

    // Close file
`ifdef USE_DPI
    dpi_close_file();
    dpi_print_stats();
`else
    $fclose(weights_file);
`endif

`ifdef BACKDOOR_LOAD
    // Force the DUT state to indicate loading is complete
    // We need to set the load_counter to the end so the DUT logic (if it runs) aligns,
    // but primarily we just need to set the memory (already done) and the completion flag.
    // Since loading_complete is a reg in DUT, we can write to it.
    dut.hdc_classifier_instance.loading_complete = 1;
    dut.hdc_classifier_instance.load_counter = bit_count;
    // Backdoor load bypasses the serial capture of online_learning_enable_reg,
    // so set it explicitly to match the config bit we loaded.
    dut.hdc_classifier_instance.online_learning_enable_reg = online_learning_enable;
    $display("[TB] Backdoor OL enable set: %0d", online_learning_enable);
    $display("[TB] Backdoor loading complete. Forced loading_complete=1. Cycles saved: Millions!");
`else
    // Standard loading completion
    // Wait for hardware to process the last bit
    @(posedge clk);  // Let hardware process the last bit with write_enable=1
    @(negedge clk);  // De-assert at negedge for clean timing
    write_enable = 0;
`endif

    $display("[TB] All data loading complete: %d total bits", bit_count);

    // Wait for loading completion
    $display("[TB] Waiting for loading_complete signal...");
    wait(loading_complete);
    $display("[TB] Loading complete! Performing memory verification...");
    
    // Moved from hdc_classifier.v initial block
    #10;
    $display("DEBUG_INTERNAL: HV_START = %d", dut.hdc_classifier_instance.HV_START);
    $display("DEBUG_INTERNAL: First 10 bits at HV_START:");
    $display("  %b %b %b %b %b %b %b %b %b %b", 
        dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START+0], 
        dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START+1], 
        dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START+2], 
        dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START+3], 
        dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START+4],
        dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START+5],
        dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START+6],
        dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START+7],
        dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START+8],
        dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START+9]);

    // CRITICAL DEBUG: Verify FC weights are readable after loading
    $display("\n=== FC WEIGHT LOADING VERIFICATION ===");
    $display("FC_WEIGHT_START (expected 12640) = %d", dut.hdc_classifier_instance.FC_WEIGHT_START);
    $display("FC_BIAS_START (expected 274784) = %d", dut.hdc_classifier_instance.FC_BIAS_START);
    $display("\nDirect memory readback of FC weight[0] (bits 12640-12643):");
    $display("  Bit 12640: %b", dut.hdc_classifier_instance.loaded_data_mem[12640]);
    $display("  Bit 12641: %b", dut.hdc_classifier_instance.loaded_data_mem[12641]);
    $display("  Bit 12642: %b", dut.hdc_classifier_instance.loaded_data_mem[12642]);
    $display("  Bit 12643: %b", dut.hdc_classifier_instance.loaded_data_mem[12643]);
    $display("  Combined 4-bit value: %b%b%b%b",
             dut.hdc_classifier_instance.loaded_data_mem[12643],
             dut.hdc_classifier_instance.loaded_data_mem[12642],
             dut.hdc_classifier_instance.loaded_data_mem[12641],
             dut.hdc_classifier_instance.loaded_data_mem[12640]);

    $display("\nAccessor function test:");
    $display("  get_fc_weight(0,0) = %b (%d signed)",
             dut.hdc_classifier_instance.get_fc_weight(7'd0, 10'd0),
             $signed(dut.hdc_classifier_instance.get_fc_weight(7'd0, 10'd0)));
    $display("  get_fc_weight(0,1) = %b (%d signed)",
             dut.hdc_classifier_instance.get_fc_weight(7'd0, 10'd1),
             $signed(dut.hdc_classifier_instance.get_fc_weight(7'd0, 10'd1)));
    $display("  get_fc_bias(0) = %b (%d signed)",
             dut.hdc_classifier_instance.get_fc_bias(7'd0),
             $signed(dut.hdc_classifier_instance.get_fc_bias(7'd0)));
    $display("  get_fc_bias(1) = %b (%d signed)",
             dut.hdc_classifier_instance.get_fc_bias(7'd1),
             $signed(dut.hdc_classifier_instance.get_fc_bias(7'd1)));
    $display("=== END VERIFICATION ===\n");

    // Verify class hypervectors by computing checksums (for Python comparison)
    $display("\n=== Class HV Checksums (compare with Python) ===");
    for (c = 0; c < NUM_CLASSES; c = c + 1) begin
        integer ones_count, checksum_val;
        ones_count = 0;
        checksum_val = 0;
        for (idx = 0; idx < HDC_HV_DIM; idx = idx + 1) begin
            if (dut.hdc_classifier_instance.loaded_data_mem[dut.hdc_classifier_instance.HV_START + c * HDC_HV_DIM + idx] == 1'b1) begin
                ones_count = ones_count + 1;
                checksum_val = (checksum_val + idx) % 1000000;
            end
        end
        $display("  Class %0d: ones=%0d/%0d, checksum=%0d", c, ones_count, HDC_HV_DIM, checksum_val);
    end
    $display("===============================================\n");

    // Snapshot initial class HVs for drift tracking
    $display("\n=== Capturing Initial Class HVs ===");
    for (c = 0; c < NUM_CLASSES; c = c + 1) begin
        for (idx = 0; idx < HDC_HV_DIM; idx = idx + 1) begin
            initial_class_hv[c][idx] = dut.hdc_classifier_instance.loaded_data_mem[
                dut.hdc_classifier_instance.HV_START + c * HDC_HV_DIM + idx
            ];
        end
    end
    $display("==================================\n");

    // Reciprocal LUT verification removed - hardware now uses division directly

    // Load test images
    load_test_images();

    // Load Python predictions for comparison
    load_python_predictions();

    // Wait a bit before starting classification
    repeat(10) @(posedge clk);

    // Process all test images
    $display("\n========================================");
    $display("=== Starting Classification Test ===");
    $display("========================================");
    $display("Processing %d test images...", actual_loaded_images);
    $display("NOTE: Latency for each image measured from 'valid' to 'valid_out'");
    $display("      (clock cycles from input arrival to prediction ready)");
    $display("");

    for (idx = 0; idx < actual_loaded_images; idx = idx + 1) begin
        process_image(idx);
        // Add small delay between images to ensure clean separation
        repeat(10) @(posedge clk);
    end

    // Report final accuracy
    $display("\n========================================");
    $display("========================================");
    $display("Final Results:");
    $display("========================================");
    $display("NOTE: Latency measured from 'valid' input to 'valid_out' output");
    $display("      (clock cycles from image arrival to prediction ready)");
    $display("========================================");
    $display("Dataset: %s", dataset_name);
    $display("Total Images: %d", total_predictions);
    $display("Correct Predictions: %d", correct_predictions);
    if (total_predictions > 0) begin
        $display("Final Accuracy: %.2f%%", (correct_predictions * 100.0) / total_predictions);
    end

    // Online learning effectiveness: accuracy by decile
    $display("\n========================================");
    $display("Online Learning Effectiveness");
    $display("========================================");
    $display("Accuracy by test image decile:");
    $display("(Shows if online learning improves accuracy over time)");
    $display("");
    $display("Decile    Images      Correct/Total    Accuracy");
    $display("------    ------      -------------    --------");
    for (i = 0; i < 10; i = i + 1) begin
        if (decile_total[i] > 0) begin
            integer start_img, end_img;
            real decile_acc;
            start_img = (i * actual_loaded_images) / 10;
            end_img = ((i + 1) * actual_loaded_images) / 10 - 1;
            decile_acc = (decile_correct[i] * 100.0) / decile_total[i];
            $display("  %0d       %3d-%-3d     %3d/%-3d          %.1f%%",
                    i, start_img, end_img, decile_correct[i], decile_total[i], decile_acc);
        end
    end
    `ifdef ENABLE_ONLINE_LEARNING_ARG
        $display("\nOnline learning updates: %0d", online_learning_updates);
        if (online_learning_updates > 0) begin
            $display("Online learning updates by class:");
            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                $display("  Class %0d: %0d (%.1f%%)", i, online_learning_updates_by_class[i],
                         (online_learning_updates_by_class[i] * 100.0) / online_learning_updates);
            end
            $display("Online learning updates by true label:");
            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                $display("  Class %0d: %0d (%.1f%%)", i, online_learning_updates_by_true_label[i],
                         (online_learning_updates_by_true_label[i] * 100.0) / online_learning_updates);
            end
        end else begin
            $display("Online learning updates by class: none");
        end
    `endif

    // Latency statistics
    if (actual_loaded_images > 0) begin
        real min_latency, max_latency, avg_latency, latency_sum;
        reg [63:0] latency;
        min_latency = 1e12;  // Start with very large value
        max_latency = 0;
        latency_sum = 0;

        $display("\n========================================");
        $display("Latency Statistics (Input to Output)");
        $display("========================================");
        $display("NOTE: Latency = cycles from 'valid' assertion to 'valid_out' assertion");
        $display("      (end-to-end pipeline delay from image input to prediction output)");
        $display("");
        for (i = 0; i < actual_loaded_images; i = i + 1) begin
            latency = image_end_cycle[i] - image_start_cycle[i];
            latency_sum = latency_sum + latency;
            if (latency < min_latency) min_latency = latency;
            if (latency > max_latency) max_latency = latency;

            // Print first 5 and last 5 image latencies
            if (i < 5 || i >= actual_loaded_images - 5) begin
                $display("  Image %d: %0d cycles (start=%0d, end=%0d)",
                        i, latency, image_start_cycle[i], image_end_cycle[i]);
            end else if (i == 5) begin
                $display("  ...");
            end
        end

        avg_latency = latency_sum / actual_loaded_images;
        $display("");
        $display("  Minimum latency: %.0f cycles", min_latency);
        $display("  Maximum latency: %.0f cycles", max_latency);
        $display("  Average latency: %.0f cycles/image", avg_latency);
        $display("  Total test time: %0d cycles", cycle_count);
        $display("");
        $display("  Throughput @ 500 MHz: %.0f images/second", 500_000_000.0 / avg_latency);
        $display("  Throughput @ 100 MHz: %.0f images/second", 100_000_000.0 / avg_latency);
    end

    // Confidence statistics
    $display("\nConfidence Statistics:");
    $display("  Minimum: %d/15 (%.2f)", min_confidence, min_confidence/15.0);
    $display("  Maximum: %d/15 (%.2f)", max_confidence, max_confidence/15.0);
    if (total_predictions > 0) begin
        $display("  Average: %.2f/15 (%.2f)", total_confidence*1.0/total_predictions,
                 (total_confidence*1.0/total_predictions)/15.0);
    end

    // Per-class confidence statistics
    $display("\nPer-Class Confidence:");
    for (i = 0; i < NUM_CLASSES; i = i + 1) begin
        if (class_confidence_count[i] > 0) begin
            $display("  Class %d avg confidence: %.2f/15 (%.2f)", i,
                    class_confidence_sum[i]/class_confidence_count[i],
                    (class_confidence_sum[i]/class_confidence_count[i])/15.0);
        end
    end

    // Confidence distribution histogram
    $display("\nConfidence Distribution:");
    for (i = 0; i <= 15; i = i + 1) begin
        if (confidence_histogram[i] > 0) begin
            $display("  Confidence %d/15: %d times (%.1f%%)", i,
                    confidence_histogram[i],
                    100.0 * confidence_histogram[i] / total_predictions);
        end
    end

    // Per-class accuracy
    $display("\nPer-Class Accuracy:");
    for (i = 0; i < NUM_CLASSES; i = i + 1) begin
        if (class_total[i] > 0) begin
            $display("  Class %d: %d/%d = %.1f%%", i, class_correct[i], class_total[i],
                     (class_correct[i] * 100.0) / class_total[i]);
        end else begin
            $display("  Class %d: No samples", i);
        end
    end

    // Prediction distribution
    $display("\nPrediction Distribution:");
    for (i = 0; i < NUM_CLASSES; i = i + 1) begin
        if (total_predictions > 0) begin
            $display("  Predicted as class %d: %d times (%.1f%%)", i, class_predictions[i],
                     (class_predictions[i] * 100.0) / total_predictions);
        end
    end

    // Class HV drift summary (unique bit flips since load)
    $display("\nClass HV Drift Summary:");
    for (c = 0; c < NUM_CLASSES; c = c + 1) begin
        hv_drift_count[c] = 0;
        for (idx = 0; idx < HDC_HV_DIM; idx = idx + 1) begin
            if (initial_class_hv[c][idx] != dut.hdc_classifier_instance.loaded_data_mem[
                dut.hdc_classifier_instance.HV_START + c * HDC_HV_DIM + idx
            ]) begin
                hv_drift_count[c] = hv_drift_count[c] + 1;
            end
        end
        $display("  Class %0d: %0d/%0d bits changed (%.2f%%)",
                 c, hv_drift_count[c], HDC_HV_DIM,
                 (hv_drift_count[c] * 100.0) / HDC_HV_DIM);
    end

    // Python vs Verilog comparison summary
    if (python_predictions_available) begin
        $display("\n========================================");
        $display("Python vs Verilog Prediction Comparison:");
        $display("========================================");
        $display("  Agreements (same prediction): %d (%.1f%%)",
                 python_verilog_matches,
                 (python_verilog_matches * 100.0) / total_predictions);
        $display("  Disagreements (different prediction): %d (%.1f%%)",
                 python_verilog_mismatches,
                 (python_verilog_mismatches * 100.0) / total_predictions);
        $display("\nThis %d disagreement gap explains the accuracy difference!", python_verilog_mismatches);
        $display("Each mismatch is a case where Python and Verilog computed different results.");
    end

    $display("\n========================================");
    $display("=== Test Complete ===");
    $finish;
end

// Timeout protection (loading + all test images)
// Each image needs TIMEOUT_CYCLES, plus time for loading
initial begin
    // Calculate total timeout: loading time + (NUM_TEST_IMAGES * TIMEOUT_CYCLES * clock_period)
    // Use 10ns clock period, add extra margin for loading (10 seconds)
    #(10_000_000_000 + NUM_TEST_IMAGES * TIMEOUT_CYCLES * 10);
    $display("ERROR: Global test timeout!");
    $display("  Expected cycles per image: %d", EXPECTED_CYCLES);
    $display("  Timeout cycles per image: %d", TIMEOUT_CYCLES);
    $display("  Total test images: %d", NUM_TEST_IMAGES);
    $finish;
end

// Include the debug monitor which accesses DUT internals
`include "hdc_debug.vh"

endmodule
