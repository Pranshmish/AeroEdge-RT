/**
 * @file activations.hpp
 * @brief High-Speed Integer Non-Linear Activation & Normalization Kernels.
 * @details Implements ReLU, LeakyReLU, and pure-integer Softmax. Eliminates all FPU
 *          instructions and avoids transcendental library dependencies on bare-metal cores.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include "../fixed_point_math.hpp"

namespace aeroedge {
namespace ops {

/**
 * @brief In-place INT8 Rectified Linear Unit (ReLU) with saturation clamping.
 */
inline void relu_int8(int8_t* data, size_t size, int8_t zero_point = 0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] < zero_point) {
            data[i] = zero_point;
        }
    }
}

/**
 * @brief In-place Fixed-Point LeakyReLU.
 *        Negative slope alpha in Q8 format (e.g., 0.1 * 256 = 26).
 */
inline void leaky_relu_int8(int8_t* data, size_t size, int32_t alpha_q8 = 26, int8_t zero_point = 0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] < zero_point) {
            int32_t diff = static_cast<int32_t>(data[i]) - zero_point;
            int32_t scaled = (diff * alpha_q8) >> 8;
            data[i] = math::saturate_int8(scaled + zero_point);
        }
    }
}

/**
 * @brief Pure-Integer Deterministic Softmax for Classifier & SLM Logit Selection.
 *        Avoids floating-point expf() and floating-point divisions.
 * 
 * @param logits Input raw INT8 logits array [size].
 * @param probabilities Output INT8 probabilities array [size] (sums to approx 1.0 / 127).
 * @param size Array length (number of classes or vocabulary tokens).
 */
inline void softmax_int8(const int8_t* logits, int8_t* probabilities, size_t size) noexcept {
    if (size == 0) return;

    // 1. Find max logit to prevent integer exponent overflow
    int8_t max_val = logits[0];
    for (size_t i = 1; i < size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // 2. Compute fixed-point exponents and sum
    int32_t sum_exp = 0;
    // Stack-allocated or temporary static buffer for exponents
    // For constrained microcontrollers, we support up to 256 classes/tokens in direct pass
    constexpr size_t MAX_SOFTMAX_ELEMENTS = 256;
    int32_t exp_vals[MAX_SOFTMAX_ELEMENTS];
    const size_t process_size = (size > MAX_SOFTMAX_ELEMENTS) ? MAX_SOFTMAX_ELEMENTS : size;

    for (size_t i = 0; i < process_size; ++i) {
        int8_t diff = logits[i] - max_val; // <= 0
        exp_vals[i] = math::fast_fixed_point_exp(diff);
        sum_exp += exp_vals[i];
    }

    if (sum_exp <= 0) sum_exp = 1;

    // 3. Normalize into signed INT8 range [-128, 127]
    for (size_t i = 0; i < process_size; ++i) {
        int32_t p = (exp_vals[i] * 255) / sum_exp; // in range [0, 255]
        probabilities[i] = static_cast<int8_t>(p - 128); // map to [-128, 127]
    }
}

/**
 * @brief Fast integer ArgMax: returns index of maximum value.
 */
inline size_t argmax_int8(const int8_t* data, size_t size) noexcept {
    if (size == 0) return 0;
    size_t max_idx = 0;
    int8_t max_val = data[0];
    for (size_t i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

} // namespace ops
} // namespace aeroedge
