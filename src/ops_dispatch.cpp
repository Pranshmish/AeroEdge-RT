/**
 * @file ops_dispatch.cpp
 * @brief High-Performance Graph Interpreter & Layer Dispatcher.
 * @details Sequentially coordinates quantized operators across static memory buffers.
 *          Guarantees zero heap allocation throughout the entire forward pass.
 */

#include "../include/aeroedge/kernel_core.hpp"
#include "../include/aeroedge/ops/conv2d_int8.hpp"
#include "../include/aeroedge/ops/matmul_int8.hpp"
#include "../include/aeroedge/ops/activations.hpp"
#include "../models/micro_vision_weights.h"
#include "../models/micro_slm_weights.h"

namespace aeroedge {
namespace dispatch {

/**
 * @brief Runs the complete 3-layer Quantized Edge Vision Pipeline.
 * 
 * Pipeline:
 *   Input [28x28x1]
 *     -> Conv2D 3x3 s2 p1 -> [14x14x4 = 784 bytes]
 *     -> In-place ReLU
 *     -> DepthwiseConv2D 3x3 s2 p1 -> [7x7x4 = 196 bytes]
 *     -> In-place ReLU
 *     -> MatMul (196 -> 10) -> Logits [10 bytes]
 *     -> Integer Softmax -> Probabilities [10 bytes]
 *     -> ArgMax -> Class Index
 */
bool run_vision_pipeline(
    memory::TensorArena<>& arena,
    const int8_t* input_image,
    int32_t& out_predicted_class,
    int8_t& out_confidence
) noexcept {
    // 1. Static slice allocations from SRAM arena (ping-pong activation buffers)
    constexpr size_t L1_OUTPUT_SIZE = 14 * 14 * 4; // 784 bytes
    constexpr size_t L2_OUTPUT_SIZE = 7 * 7 * 4;   // 196 bytes
    constexpr size_t L3_OUTPUT_SIZE = 10;          // 10 bytes

    int8_t* l1_act = static_cast<int8_t*>(arena.allocate(L1_OUTPUT_SIZE));
    int8_t* l2_act = static_cast<int8_t*>(arena.allocate(L2_OUTPUT_SIZE));
    int8_t* l3_act = static_cast<int8_t*>(arena.allocate(L3_OUTPUT_SIZE));
    int8_t* probs  = static_cast<int8_t*>(arena.allocate(L3_OUTPUT_SIZE));

    if (!l1_act || !l2_act || !l3_act || !probs) {
        return false; // Insufficient arena memory
    }

    // 2. Layer 1: Conv2D
    ops::Conv2DParams l1_params = {
        models::vision::IN_HEIGHT, models::vision::IN_WIDTH, models::vision::IN_CHANNELS,
        14, 14, models::vision::L1_OUT_CHANNELS,
        models::vision::L1_KERNEL_H, models::vision::L1_KERNEL_W,
        2, 2, // stride
        1, 1, // pad
        0, 0, // offsets
        1073741824, 30, // Multiplier (approx 0.5 in Q31)
        -128, 127
    };
    ops::conv2d_int8(l1_params, input_image, models::vision::L1_CONV_WEIGHTS, models::vision::L1_CONV_BIAS, l1_act);
    ops::relu_int8(l1_act, L1_OUTPUT_SIZE, 0);

    // 3. Layer 2: DepthwiseConv2D
    ops::Conv2DParams l2_params = {
        14, 14, 4,
        7, 7, 4,
        3, 3,
        2, 2, // stride
        1, 1, // pad
        0, 0,
        1073741824, 30,
        -128, 127
    };
    ops::depthwise_conv2d_int8(l2_params, l1_act, models::vision::L2_DW_WEIGHTS, models::vision::L2_DW_BIAS, l2_act);
    ops::relu_int8(l2_act, L2_OUTPUT_SIZE, 0);

    // 4. Layer 3: Linear Classifier (196 -> 10)
    ops::MatMulParams l3_params = {
        models::vision::L3_CLASSES, models::vision::L3_IN_FEATURES,
        0, 0,
        1073741824, 30,
        -128, 127
    };
    ops::matmul_int8(l3_params, l2_act, models::vision::L3_LINEAR_WEIGHTS, models::vision::L3_LINEAR_BIAS, l3_act);

    // 5. Activation: Integer Softmax & Argmax
    ops::softmax_int8(l3_act, probs, models::vision::L3_CLASSES);
    size_t predicted = ops::argmax_int8(probs, models::vision::L3_CLASSES);

    out_predicted_class = static_cast<int32_t>(predicted);
    out_confidence = probs[predicted];
    return true;
}

/**
 * @brief Runs the Micro-SLM Projection & Token Selection.
 * 
 * Computes Y = Softmax(W_slm * hidden_state + b_slm) -> next_token
 */
bool run_slm_token_generation(
    memory::TensorArena<>& arena,
    const int8_t* hidden_state,
    int32_t& out_token_id,
    int8_t& out_confidence
) noexcept {
    int8_t* logits = static_cast<int8_t*>(arena.allocate(models::slm::VOCAB_SIZE));
    int8_t* probs  = static_cast<int8_t*>(arena.allocate(models::slm::VOCAB_SIZE));

    if (!logits || !probs) return false;

    ops::MatMulParams params = {
        models::slm::VOCAB_SIZE, models::slm::HIDDEN_DIM,
        0, 0,
        1073741824, 30,
        -128, 127
    };

    ops::matmul_int8(params, hidden_state, models::slm::SLM_LINEAR_WEIGHTS, models::slm::SLM_LINEAR_BIAS, logits);
    ops::softmax_int8(logits, probs, models::slm::VOCAB_SIZE);
    size_t top_token = ops::argmax_int8(probs, models::slm::VOCAB_SIZE);

    out_token_id = static_cast<int32_t>(top_token);
    out_confidence = probs[top_token];
    return true;
}

} // namespace dispatch
} // namespace aeroedge
