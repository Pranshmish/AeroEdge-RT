/**
 * @file conv2d_int8.hpp
 * @brief High-Throughput Quantized 2D Convolution & Depthwise Separable Kernels.
 * @details Implements spatial 2D convolution without dynamic memory allocation or im2col 
 *          RAM blowup. Processes channels directly in-place with SIMD vector strides.
 * 
 * Arithmetic: Fixed-Point INT8 Weights, INT8 Activations, INT32 Accumulators.
 * Memory: Activation ping-pong in 16-byte aligned SRAM.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include "../fixed_point_math.hpp"

namespace aeroedge {
namespace ops {

struct Conv2DParams {
    int32_t in_height;
    int32_t in_width;
    int32_t in_channels;
    int32_t out_height;
    int32_t out_width;
    int32_t out_channels;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t input_offset;
    int32_t output_offset;
    int32_t multiplier;
    int32_t shift;
    int8_t clamp_min;
    int8_t clamp_max;
};

/**
 * @brief Direct 2D Convolution with INT8 Quantization.
 *        Optimized for edge vision pipelines (NHWC format).
 */
inline void conv2d_int8(
    const Conv2DParams& params,
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ filter,
    const int32_t* __restrict__ bias,
    int8_t* __restrict__ output
) noexcept {
    const int32_t in_h = params.in_height;
    const int32_t in_w = params.in_width;
    const int32_t in_c = params.in_channels;
    const int32_t out_h = params.out_height;
    const int32_t out_w = params.out_width;
    const int32_t out_c = params.out_channels;
    const int32_t k_h = params.kernel_h;
    const int32_t k_w = params.kernel_w;
    const int32_t in_offset = params.input_offset;

    for (int32_t oh = 0; oh < out_h; ++oh) {
        for (int32_t ow = 0; ow < out_w; ++ow) {
            for (int32_t oc = 0; oc < out_c; ++oc) {
                int32_t acc = bias ? bias[oc] : 0;

                const int32_t in_y_origin = (oh * params.stride_h) - params.pad_h;
                const int32_t in_x_origin = (ow * params.stride_w) - params.pad_w;

                for (int32_t kh = 0; kh < k_h; ++kh) {
                    const int32_t in_y = in_y_origin + kh;
                    if (in_y < 0 || in_y >= in_h) continue;

                    for (int32_t kw = 0; kw < k_w; ++kw) {
                        const int32_t in_x = in_x_origin + kw;
                        if (in_x < 0 || in_x >= in_w) continue;

                        const int8_t* in_ptr = input + ((in_y * in_w + in_x) * in_c);
                        const int8_t* f_ptr = filter + ((((oc * k_h + kh) * k_w + kw) * in_c));

                        int32_t ic = 0;
                        const int32_t unrolled_ic = in_c & ~3;
                        for (; ic < unrolled_ic; ic += 4) {
                            acc += (static_cast<int32_t>(in_ptr[ic + 0]) - in_offset) * static_cast<int32_t>(f_ptr[ic + 0]) +
                                   (static_cast<int32_t>(in_ptr[ic + 1]) - in_offset) * static_cast<int32_t>(f_ptr[ic + 1]) +
                                   (static_cast<int32_t>(in_ptr[ic + 2]) - in_offset) * static_cast<int32_t>(f_ptr[ic + 2]) +
                                   (static_cast<int32_t>(in_ptr[ic + 3]) - in_offset) * static_cast<int32_t>(f_ptr[ic + 3]);
                        }
                        for (; ic < in_c; ++ic) {
                            acc += (static_cast<int32_t>(in_ptr[ic]) - in_offset) * static_cast<int32_t>(f_ptr[ic]);
                        }
                    }
                }

                int32_t res = math::multiply_by_quantized_multiplier(acc, params.multiplier, params.shift);
                res += params.output_offset;
                if (res < params.clamp_min) res = params.clamp_min;
                if (res > params.clamp_max) res = params.clamp_max;

                output[(oh * out_w + ow) * out_c + oc] = static_cast<int8_t>(res);
            }
        }
    }
}

/**
 * @brief Depthwise Separable 2D Convolution (1 filter channel per input channel).
 *        Drastically reduces parameters and computational complexity for Mobile edge vision.
 */
inline void depthwise_conv2d_int8(
    const Conv2DParams& params,
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ filter,
    const int32_t* __restrict__ bias,
    int8_t* __restrict__ output
) noexcept {
    const int32_t in_h = params.in_height;
    const int32_t in_w = params.in_width;
    const int32_t channels = params.in_channels; // in_channels == out_channels
    const int32_t out_h = params.out_height;
    const int32_t out_w = params.out_width;
    const int32_t k_h = params.kernel_h;
    const int32_t k_w = params.kernel_w;
    const int32_t in_offset = params.input_offset;

    for (int32_t oh = 0; oh < out_h; ++oh) {
        for (int32_t ow = 0; ow < out_w; ++ow) {
            for (int32_t c = 0; c < channels; ++c) {
                int32_t acc = bias ? bias[c] : 0;

                const int32_t in_y_origin = (oh * params.stride_h) - params.pad_h;
                const int32_t in_x_origin = (ow * params.stride_w) - params.pad_w;

                for (int32_t kh = 0; kh < k_h; ++kh) {
                    const int32_t in_y = in_y_origin + kh;
                    if (in_y < 0 || in_y >= in_h) continue;

                    for (int32_t kw = 0; kw < k_w; ++kw) {
                        const int32_t in_x = in_x_origin + kw;
                        if (in_x < 0 || in_x >= in_w) continue;

                        int32_t in_val = static_cast<int32_t>(input[(in_y * in_w + in_x) * channels + c]) - in_offset;
                        int32_t f_val = static_cast<int32_t>(filter[(kh * k_w + kw) * channels + c]);
                        acc += in_val * f_val;
                    }
                }

                int32_t res = math::multiply_by_quantized_multiplier(acc, params.multiplier, params.shift);
                res += params.output_offset;
                if (res < params.clamp_min) res = params.clamp_min;
                if (res > params.clamp_max) res = params.clamp_max;

                output[(oh * out_w + ow) * channels + c] = static_cast<int8_t>(res);
            }
        }
    }
}

} // namespace ops
} // namespace aeroedge
