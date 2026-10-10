/**
 * @file micro_vision_weights.h
 * @brief Quantized Flash-Resident (XIP) Weights for Edge Vision Pipeline.
 * @details Architecture:
 *          - Input: 28x28x1 Grayscale Image (INT8, range [-128, 127])
 *          - Layer 1: Conv2D 3x3 (1 in -> 4 out, stride 2, pad 1) -> 14x14x4
 *          - Layer 2: DepthwiseConv2D 3x3 (4 channels, stride 2, pad 1) -> 7x7x4
 *          - Layer 3: Linear Classifier (196 in -> 10 out) -> Logits [10]
 * 
 * Storage: 100% const Flash Memory (Zero runtime SRAM footprint for weights).
 */

#pragma once

#include <cstdint>

namespace aeroedge {
namespace models {
namespace vision {

constexpr int32_t IN_HEIGHT = 28;
constexpr int32_t IN_WIDTH = 28;
constexpr int32_t IN_CHANNELS = 1;

constexpr int32_t L1_OUT_CHANNELS = 4;
constexpr int32_t L1_KERNEL_H = 3;
constexpr int32_t L1_KERNEL_W = 3;
// Filter size: 4 * 3 * 3 * 1 = 36 bytes
alignas(16) static const int8_t L1_CONV_WEIGHTS[36] = {
    // Filter 0 (Edge detector horizontal)
    -12, -25, -12,
      0,   0,   0,
     12,  25,  12,
    // Filter 1 (Edge detector vertical)
    -12,   0,  12,
    -25,   0,  25,
    -12,   0,  12,
    // Filter 2 (Laplacian / High-pass)
      0, -20,   0,
    -20,  80, -20,
      0, -20,   0,
    // Filter 3 (Smoothing / Low-pass)
     10,  15,  10,
     15,  25,  15,
     10,  15,  10
};

alignas(16) static const int32_t L1_CONV_BIAS[4] = { 10, -5, 0, 15 };

// Layer 2: Depthwise Separable Filter (4 channels * 3 * 3 = 36 bytes)
alignas(16) static const int8_t L2_DW_WEIGHTS[36] = {
    10, 15, 10, 15, 30, 15, 10, 15, 10,
    -5,  0,  5, -5, 10,  5, -5,  0,  5,
     8,  8,  8,  8, 16,  8,  8,  8,  8,
    12, -4, 12, -4, 20, -4, 12, -4, 12
};

alignas(16) static const int32_t L2_DW_BIAS[4] = { 0, 2, -2, 5 };

// Layer 3: Linear Classifier (7 * 7 * 4 = 196 inputs -> 10 output classes)
// 196 * 10 = 1960 bytes
constexpr int32_t L3_IN_FEATURES = 196;
constexpr int32_t L3_CLASSES = 10;

alignas(16) static const int8_t L3_LINEAR_WEIGHTS[L3_CLASSES * L3_IN_FEATURES] = {
    #include "vision_linear_weights.inc"
};

alignas(16) static const int32_t L3_LINEAR_BIAS[10] = {
    120, -45, 80, -110, 95, 30, -70, 15, -40, 60
};

} // namespace vision
} // namespace models
} // namespace aeroedge
