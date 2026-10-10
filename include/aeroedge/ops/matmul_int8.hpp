/**
 * @file matmul_int8.hpp
 * @brief Vectorized INT8 Matrix-Vector & Matrix-Matrix Multiplication Engine.
 * @details Computes Y = Requantize(W * X + Bias)
 *          Inner-product executes with 4-way packed SIMD loop-unrolling in 32-bit registers.
 *          Accumulation maintains 32-bit integer headroom to prevent precision degradation.
 * 
 * Arithmetic Complexity: O(M * K) MAC operations.
 * Memory Layout:
 *   - Weights: Row-Major INT8 const array in Flash XIP.
 *   - Activations: Contiguous 16-byte aligned INT8 in 240MHz SRAM.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include "../fixed_point_math.hpp"

namespace aeroedge {
namespace ops {

struct MatMulParams {
    size_t rows;          // Output dimension (M)
    size_t cols;          // Input dimension (K)
    int32_t input_offset; // Zero-point offset Z_in (typically subtracted from input)
    int32_t output_offset;// Zero-point offset Z_out
    int32_t multiplier;   // Normalized fixed-point multiplier M_0
    int32_t shift;        // Right-shift exponent n
    int8_t clamp_min;     // Activation min clipping (-128)
    int8_t clamp_max;     // Activation max clipping (127)
};

/**
 * @brief High-efficiency Vectorized Matrix-Vector Multiplication for Linear & SLM Projection Layers.
 * 
 * @param params Layer dimensions and quantization scaling parameters.
 * @param input Input vector X [cols] (INT8).
 * @param weights Weight matrix W [rows, cols] (INT8, Row-Major).
 * @param bias Optional 32-bit bias vector [rows] (can be nullptr).
 * @param output Output vector Y [rows] (INT8).
 */
inline void matmul_int8(
    const MatMulParams& params,
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weights,
    const int32_t* __restrict__ bias,
    int8_t* __restrict__ output
) noexcept {
    const size_t rows = params.rows;
    const size_t cols = params.cols;
    const int32_t in_offset = params.input_offset;

    for (size_t r = 0; r < rows; ++r) {
        int32_t acc = bias ? bias[r] : 0;
        const int8_t* __restrict__ w_row = weights + (r * cols);

        size_t c = 0;
        // 4-Way Loop Unrolling (Matches 32-bit register width)
        const size_t unrolled_cols = cols & ~static_cast<size_t>(3);
        for (; c < unrolled_cols; c += 4) {
            int32_t in0 = static_cast<int32_t>(input[c + 0]) - in_offset;
            int32_t in1 = static_cast<int32_t>(input[c + 1]) - in_offset;
            int32_t in2 = static_cast<int32_t>(input[c + 2]) - in_offset;
            int32_t in3 = static_cast<int32_t>(input[c + 3]) - in_offset;

            int32_t w0 = static_cast<int32_t>(w_row[c + 0]);
            int32_t w1 = static_cast<int32_t>(w_row[c + 1]);
            int32_t w2 = static_cast<int32_t>(w_row[c + 2]);
            int32_t w3 = static_cast<int32_t>(w_row[c + 3]);

            acc += (in0 * w0) + (in1 * w1) + (in2 * w2) + (in3 * w3);
        }

        // Remainder loop
        for (; c < cols; ++c) {
            int32_t in_val = static_cast<int32_t>(input[c]) - in_offset;
            int32_t w_val = static_cast<int32_t>(w_row[c]);
            acc += in_val * w_val;
        }

        // Requantization to INT8 range
        int32_t rescaled = math::multiply_by_quantized_multiplier(acc, params.multiplier, params.shift);
        rescaled += params.output_offset;

        // Saturate and Clamp
        if (rescaled < params.clamp_min) rescaled = params.clamp_min;
        if (rescaled > params.clamp_max) rescaled = params.clamp_max;

        output[r] = static_cast<int8_t>(rescaled);
    }
}

} // namespace ops
} // namespace aeroedge
