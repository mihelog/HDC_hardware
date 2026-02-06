// Auto-generated confidence lookup table
// Replaces expensive division: confidence = 15 - (15 * min_dist / HV_DIM)
// Generated at: 2026-02-05 15:24:04.082464
// HV_DIM = 5000
// LUT covers distances 0 to 4999
// Size: 5000 entries × 4 bits = 2500 bytes

// Confidence LUT size definitions
`define CONFIDENCE_LUT_SIZE 5000
`define CONFIDENCE_LUT_BITS 20000

// Expected confidence values for verification
// Distance -> Confidence mapping (for reference only)
// Distance    0: confidence = 15
// Distance    1: confidence = 15
// Distance    2: confidence = 15
// Distance    3: confidence = 15
// Distance    4: confidence = 15
// Distance    5: confidence = 15
// Distance    6: confidence = 15
// Distance    7: confidence = 15
// Distance    8: confidence = 15
// Distance    9: confidence = 15
// Distance  100: confidence = 15
// Distance  200: confidence = 14
// Distance  300: confidence = 14
// Distance  400: confidence = 14
// Distance  500: confidence = 14
// Distance  600: confidence = 13
// Distance  700: confidence = 13
// Distance  800: confidence = 13
// Distance  900: confidence = 12
// Distance 1000: confidence = 12
// Distance 1100: confidence = 12
// Distance 1200: confidence = 11
// Distance 1300: confidence = 11
// Distance 1400: confidence = 11
// Distance 1500: confidence = 10
// Distance 1600: confidence = 10
// Distance 1700: confidence = 10
// Distance 1800: confidence = 10
// Distance 1900: confidence =  9
// Distance 2000: confidence =  9
// Distance 2100: confidence =  9
// Distance 2200: confidence =  8
// Distance 2300: confidence =  8
// Distance 2400: confidence =  8
// Distance 2500: confidence =  8
// Distance 2600: confidence =  7
// Distance 2700: confidence =  7
// Distance 2800: confidence =  7
// Distance 2900: confidence =  6
// Distance 3000: confidence =  6
// Distance 3100: confidence =  6
// Distance 3200: confidence =  5
// Distance 3300: confidence =  5
// Distance 3400: confidence =  5
// Distance 3500: confidence =  4
// Distance 3600: confidence =  4
// Distance 3700: confidence =  4
// Distance 3800: confidence =  4
// Distance 3900: confidence =  3
// Distance 4000: confidence =  3
// Distance 4100: confidence =  3
// Distance 4200: confidence =  2
// Distance 4300: confidence =  2
// Distance 4400: confidence =  2
// Distance 4500: confidence =  2
// Distance 4600: confidence =  1
// Distance 4700: confidence =  1
// Distance 4800: confidence =  1
// Distance 4900: confidence =  0
// Hardware savings:
// Before: 16-bit × 4-bit multiplication + 16-bit ÷ 14-bit division (~50 LUTs)
// After: 5000-entry × 4-bit LUT (~9 BRAM blocks)
// Speed improvement: ~10x (single cycle LUT vs multi-cycle divider)
