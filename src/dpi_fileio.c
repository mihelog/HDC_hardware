// DPI functions for fast file I/O in Verilog testbench
// Compile with: gcc -shared -fPIC -o dpi_fileio.so dpi_fileio.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "svdpi.h"

// File handle
static FILE* weights_file = NULL;
static int projection_threshold = 0;
static int fc_thresh1 = 0;
static int fc_thresh2 = 0;
static int has_prefetched = 0;
static int prefetched_value = 0;

// Open the weights file and parse header
int dpi_open_weights_file(const char* filename) {
    weights_file = fopen(filename, "r");
    if (!weights_file) {
        printf("[DPI] ERROR: Cannot open %s\n", filename);
        return 0;
    }

    projection_threshold = 0;
    fc_thresh1 = 0;
    fc_thresh2 = 0;
    has_prefetched = 0;

    // Read and parse header until first numeric data line
    char line[256];
    while (fgets(line, sizeof(line), weights_file) != NULL) {
        // Check if this line starts with a number (data section)
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '-' || (*p >= '0' && *p <= '9')) {
            if (sscanf(p, "%d", &prefetched_value) == 1) {
                has_prefetched = 1;
            }
            break;
        }

        // Parse thresholds by key (order-independent)
        if (sscanf(line, "PROJECTION_THRESHOLD %d", &projection_threshold) == 1) {
            continue;
        }
        if (sscanf(line, "FC_THRESH1 %d", &fc_thresh1) == 1) {
            continue;
        }
        if (sscanf(line, "FC_THRESH2 %d", &fc_thresh2) == 1) {
            continue;
        }
    }

    printf("[DPI] Opened %s successfully\n", filename);
    printf("[DPI] Thresholds: PROJ=%d, FC1=%d, FC2=%d\n",
           projection_threshold, fc_thresh1, fc_thresh2);
    return 1;
}

// Get threshold values
int dpi_get_threshold(int thresh_id) {
    switch(thresh_id) {
        case 0: return projection_threshold;
        case 1: return fc_thresh1;
        case 2: return fc_thresh2;
        default: return 0;
    }
}

// Read next integer value from file
int dpi_read_weight(int* value) {
    if (!weights_file) {
        printf("[DPI] ERROR: File not open\n");
        return 0;
    }

    if (has_prefetched) {
        *value = prefetched_value;
        has_prefetched = 0;
        return 1;
    }

    if (fscanf(weights_file, "%d", value) == 1) {
        return 1;
    }
    return 0;
}

// Read multiple weights at once (batch reading for performance)
int dpi_read_weights_batch(int* values, int count) {
    if (!weights_file) {
        printf("[DPI] ERROR: File not open\n");
        return 0;
    }

    int read_count = 0;
    if (has_prefetched && count > 0) {
        values[0] = prefetched_value;
        has_prefetched = 0;
        read_count = 1;
    }
    for (int i = 0; i < count; i++) {
        int idx = i + read_count;
        if (idx >= count) {
            break;
        }
        if (fscanf(weights_file, "%d", &values[idx]) == 1) {
            read_count++;
        } else {
            break;
        }
    }
    return read_count;
}

// Convert integer to bit array (LSB first)
void dpi_int_to_bits(int value, int bit_width, svBitVecVal* bits) {
    for (int i = 0; i < bit_width; i++) {
        if (value & (1 << i)) {
            svPutBitselBit(bits, i, 1);
        } else {
            svPutBitselBit(bits, i, 0);
        }
    }
}

// Helper function to get bit string for debugging
void dpi_get_bit_string(int value, int bit_width, char* str) {
    for (int i = 0; i < bit_width; i++) {
        str[i] = (value & (1 << i)) ? '1' : '0';
    }
    str[bit_width] = '\0';
}

// Close the file
void dpi_close_file() {
    if (weights_file) {
        fclose(weights_file);
        weights_file = NULL;
        printf("[DPI] File closed\n");
    }
}

// Performance monitoring
static long bytes_read = 0;
static int values_read = 0;

void dpi_print_stats() {
    printf("[DPI] Statistics: %d values read, ~%ld KB processed\n",
           values_read, bytes_read / 1024);
}
