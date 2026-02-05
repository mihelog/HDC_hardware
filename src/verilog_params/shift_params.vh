// Auto-generated shift parameters from Python training
// Generated at: 2026-02-05 14:12:07.755219

// Shift values determined by profiling actual data to prevent overflow
// while preserving as much precision as possible

`define PIXEL_SHIFT_OVERRIDE 0
`define CONV1_SHIFT_OVERRIDE 7
`define CONV2_SHIFT_OVERRIDE 6
`define FC_SHIFT_OVERRIDE 10

// Expected FC output statistics after shifting
`define FC_95_PERCENTILE 5546777.200000

// Quantization scales used during training
`define CONV1_WEIGHT_SCALE 512.0
`define CONV2_WEIGHT_SCALE 1.0
`define FC_WEIGHT_SCALE 32.0

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

// FC weight_scale/bias_scale = 32.0/64.0 = 0.500
// Hardware approximation: multiply by 1 and shift right by 1 (1/2 = 0.500)
`define FC_BIAS_RESCALE_MULT 1
`define FC_BIAS_RESCALE_SHIFT 1

// For HDC encoding normalization
`define GLOBAL_FEAT_MAX_SCALED 32034

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
