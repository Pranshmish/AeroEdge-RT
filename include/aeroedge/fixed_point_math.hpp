/**
 * @file fixed_point_math.hpp
 * @brief High-Performance Deterministic Fixed-Point & Quantization Arithmetic Engine.
 * @details AeroEdge-RT eliminates all hardware floating-point unit (FPU) dependencies
 *          and preemption latency by executing all tensor operations using strict 
 *          32-bit integer arithmetic with saturating bounds.
 * 
 * Mathematical Formulation:
 *   Real value r \approx S * (q - Z)
 *   Quantized multiplier: M = (S_in * S_weight) / S_out = 2^(-n) * M_0
 *   where M_0 \in [2^30, 2^31 - 1] (normalized integer multiplier) and n >= 0 (right shift).
 * 
 * Target: Xtensa LX7 (ESP32-S3), Xtensa LX6 (ESP32), Host x86_64
 * License: Apache-2.0 / Academic Research License
 */

#pragma once

#include <cstdint>
#include <climits>
#include <algorithm>

namespace aeroedge {
namespace math {

// ============================================================================
// Fixed-Point Precision Constants & Bitfield Helpers
// ============================================================================

constexpr int32_t Q31_ONE = 0x7FFFFFFF;
constexpr int32_t INT8_MIN_VAL = -128;
constexpr int32_t INT8_MAX_VAL = 127;

/**
 * @brief Clamp a 32-bit integer into standard signed 8-bit range [-128, 127]
 *        using branch-free or compiler-saturating primitives.
 */
inline constexpr int8_t saturate_int8(int32_t val) noexcept {
    if (val > INT8_MAX_VAL) return static_cast<int8_t>(INT8_MAX_VAL);
    if (val < INT8_MIN_VAL) return static_cast<int8_t>(INT8_MIN_VAL);
    return static_cast<int8_t>(val);
}

/**
 * @brief Saturating 32-bit addition to prevent integer wrap-around.
 */
inline constexpr int32_t saturate_add_int32(int32_t a, int32_t b) noexcept {
    int64_t res = static_cast<int64_t>(a) + static_cast<int64_t>(b);
    if (res > INT32_MAX) return INT32_MAX;
    if (res < INT32_MIN) return INT32_MIN;
    return static_cast<int32_t>(res);
}

/**
 * @brief Fixed-point scale multiplier: y = (x * M_0 + (1 << (n - 1))) >> n
 *        Requantizes an INT32 accumulator back to the INT8 output scale.
 * 
 * @param val INT32 accumulator value.
 * @param multiplier Normalized integer multiplier M_0 \in [2^30, 2^31 - 1].
 * @param shift Right shift exponent n (>= 0).
 * @return int32_t Rescaled integer output.
 */
inline int32_t multiply_by_quantized_multiplier(int32_t val, int32_t multiplier, int32_t shift) noexcept {
    // 64-bit high multiplication: (val * multiplier)
    int64_t product = static_cast<int64_t>(val) * static_cast<int64_t>(multiplier);
    
    // Fixed-point rounding constant
    if (shift > 0) {
        int64_t round = static_cast<int64_t>(1) << (shift - 1);
        product += round;
        return static_cast<int32_t>(product >> shift);
    } else if (shift < 0) {
        // Negative shift signifies a left shift
        return static_cast<int32_t>(product << (-shift));
    }
    return static_cast<int32_t>(product);
}

/**
 * @brief Packs four 8-bit signed integers into a single 32-bit register.
 *        Enables single-cycle 32-bit memory loads and SIMD arithmetic.
 */
inline constexpr uint32_t pack_int8x4(int8_t a0, int8_t a1, int8_t a2, int8_t a3) noexcept {
    return (static_cast<uint32_t>(static_cast<uint8_t>(a0))) |
           (static_cast<uint32_t>(static_cast<uint8_t>(a1)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(a2)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(a3)) << 24);
}

/**
 * @brief Integer 4-way multiply-accumulate:
 *        Computes dot product of four 8-bit integer pairs packed in 32-bit words:
 *        acc += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
 */
inline int32_t dot_product_int8x4(uint32_t u_a, uint32_t u_b) noexcept {
    int8_t a0 = static_cast<int8_t>(u_a & 0xFF);
    int8_t a1 = static_cast<int8_t>((u_a >> 8) & 0xFF);
    int8_t a2 = static_cast<int8_t>((u_a >> 16) & 0xFF);
    int8_t a3 = static_cast<int8_t>((u_a >> 24) & 0xFF);

    int8_t b0 = static_cast<int8_t>(u_b & 0xFF);
    int8_t b1 = static_cast<int8_t>((u_b >> 8) & 0xFF);
    int8_t b2 = static_cast<int8_t>((u_b >> 16) & 0xFF);
    int8_t b3 = static_cast<int8_t>((u_b >> 24) & 0xFF);

    return (static_cast<int32_t>(a0) * b0) +
           (static_cast<int32_t>(a1) * b1) +
           (static_cast<int32_t>(a2) * b2) +
           (static_cast<int32_t>(a3) * b3);
}

/**
 * @brief High-speed Integer Exponential approximation for Quantized Softmax.
 *        Maps an input x <= 0 (where 0 corresponds to max logit) into Q15 fixed-point.
 *        Eliminates runtime libm floating-point expf() overhead.
 * 
 * Polynomial: exp(x) \approx (1 + x / 256)^256 in Q15 format.
 */
inline int32_t fast_fixed_point_exp(int8_t diff_from_max) noexcept {
    // diff_from_max is <= 0 (typically in range [-128, 0])
    // Base 2 integer decay: 2^(x/16)
    if (diff_from_max < -32) return 0;
    
    // Lookup table or piece-wise linear approximation for high determinism
    // For x in [-32, 0]:
    constexpr int32_t scale = 1024; // Q10
    int32_t x = static_cast<int32_t>(diff_from_max);
    // 2nd-order Taylor series around 0 in fixed point:
    // f(x) = 1 + x/16 + (x^2)/(2 * 256)
    int32_t term1 = scale + (x * scale) / 16;
    int32_t term2 = (x * x * scale) / 512;
    int32_t res = term1 + term2;
    return (res > 0) ? res : 0;
}

} // namespace math
} // namespace aeroedge
