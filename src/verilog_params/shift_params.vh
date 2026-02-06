// Auto-generated shift parameters from Python training
// Generated at: 2026-02-05 15:24:04.082141

// Shift values determined by profiling actual data to prevent overflow
// while preserving as much precision as possible

`define PIXEL_SHIFT_OVERRIDE 0
`define CONV1_SHIFT_OVERRIDE 7
`define CONV2_SHIFT_OVERRIDE 6
`define FC_SHIFT_OVERRIDE 8

// Expected FC output statistics after shifting
`define FC_95_PERCENTILE 1286080.050000

// Quantization scales used during training
`define CONV1_WEIGHT_SCALE 512.0
`define CONV2_WEIGHT_SCALE 1.0
`define FC_WEIGHT_SCALE 8.0

// Bias rescaling parameters for all layers
// Python rescales biases during forward pass: rescaled_bias = bias * (weight_scale / bias_scale)

// Conv1 weight_scale/bias_scale = 512.0/512.0 = 1.000
// Hardware approximation: multiply by 1 and shift right by 0 (1/1 = 1.000)
`define CONV1_BIAS_RESCALE_MULT 1
`define CONV1_BIAS_RESCALE_SHIFT 0

// Conv2 weight_scale/bias_scale = 1.0/1.0 = 1.000
// Hardware approximation: multiply by 1 and shift right by 0 (1/1 = 1.000)
`define CONV2_BIAS_RESCALE_MULT 1
`define CONV2_BIAS_RESCALE_SHIFT 0

// FC weight_scale/bias_scale = 8.0/64.0 = 0.125
// Hardware approximation: multiply by 1 and shift right by 3 (1/8 = 0.125)
`define FC_BIAS_RESCALE_MULT 1
`define FC_BIAS_RESCALE_SHIFT 3

// For HDC encoding normalization
`define GLOBAL_FEAT_MAX_SCALED 30576

// Min-max normalization encoding (adaptive per image)
`define USE_MINMAX_ENCODING
`define MINMAX_THRESHOLD_1 85   // ~0.33 * 256
`define MINMAX_THRESHOLD_2 170  // ~0.67 * 256
`define TOPK_LEVEL1 86  // Top 86/128 features for level 1
`define TOPK_LEVEL2 43  // Top 43/128 features for level 2

// ============================================
// NORMALIZATION CONFIGURATION
// ============================================
`define NORM_ENABLED 0  // Normalization is NOT ACTIVE
// No normalization parameters - using raw features
