/**
 * @file kernel_core.cpp
 * @brief AeroEdge-RT Dual-Core Lifecycle & Hardware Execution Worker.
 * @details Establishes asymmetric core execution. Core 1 runs the computational worker
 *          with zero FreeRTOS tick interference. Core 0 serves as the I/O event dispatcher.
 */

#include "../include/aeroedge/kernel_core.hpp"

namespace aeroedge {
namespace dispatch {
    extern bool run_vision_pipeline(memory::TensorArena<>& arena, const int8_t* input_image, int32_t& out_predicted_class, int8_t& out_confidence) noexcept;
    extern bool run_slm_token_generation(memory::TensorArena<>& arena, const int8_t* hidden_state, int32_t& out_token_id, int8_t& out_confidence) noexcept;
}

namespace kernel {

#ifdef ESP_PLATFORM
static void core1_task_entry(void* arg) {
    auto* kernel = static_cast<AeroEdgeKernel*>(arg);
    kernel->core1_worker_loop();
}
#endif

bool AeroEdgeKernel::init() noexcept {
    if (initialized_) return true;

    arena_.reset();

#ifdef ESP_PLATFORM
    // Launch Core 1 bare-metal isolated worker pinned strictly to Core 1
    // High priority ensures preemption immunity from standard application tasks
    BaseType_t res = xTaskCreatePinnedToCore(
        core1_task_entry,
        "AeroEdge_Core1",
        8192,
        this,
        configMAX_PRIORITIES - 1,
        &core1_task_handle_,
        1 // Core 1
    );
    if (res != pdPASS) {
        return false;
    }
#endif

    is_running_ = true;
    initialized_ = true;
    return true;
}

bool AeroEdgeKernel::dispatch_async(const ipc::InferenceRequest& req) noexcept {
    bool success = req_queue_.push(req);
    if (!success) {
        telemetry_.queue_drops++;
    }
    return success;
}

bool AeroEdgeKernel::poll_result(ipc::InferenceResponse& res) noexcept {
    return res_queue_.pop(res);
}

void AeroEdgeKernel::core1_worker_loop() noexcept {
    ipc::InferenceRequest req;

    while (is_running_) {
        if (!req_queue_.pop(req)) {
#ifdef ESP_PLATFORM
            // Microsecond yield to let watchdog feed if queue is temporarily idle
            vTaskDelay(pdMS_TO_TICKS(1));
#endif
            continue;
        }

        // Measure execution cycles on isolated silicon
        uint32_t start_cycles = read_cpu_cycles();
        
        // Reset bump pointer in SRAM arena for zero-overhead buffer recycling
        arena_.reset();

        int32_t pred_class = -1;
        int8_t confidence = 0;
        bool ok = false;

        if (req.model_id == 0) {
            // Edge Vision Pipeline
            ok = dispatch::run_vision_pipeline(arena_, req.input_tensor, pred_class, confidence);
        } else if (req.model_id == 1) {
            // Micro-SLM Token Generation Pipeline
            ok = dispatch::run_slm_token_generation(arena_, req.input_tensor, pred_class, confidence);
        }

        uint32_t end_cycles = read_cpu_cycles();
        uint32_t elapsed_cycles = end_cycles - start_cycles;

        // Telemetry calculation (assumes 240MHz core frequency)
        uint32_t latency_us = elapsed_cycles / 240;

        telemetry_.total_inferences++;
        telemetry_.total_cycles += elapsed_cycles;
        if (elapsed_cycles < telemetry_.min_cycles) telemetry_.min_cycles = elapsed_cycles;
        if (elapsed_cycles > telemetry_.max_cycles) telemetry_.max_cycles = elapsed_cycles;
        telemetry_.last_latency_us = latency_us;
        telemetry_.sram_peak_bytes = arena_.peak_usage();

        ipc::InferenceResponse res = {
            req.request_id,
            pred_class,
            confidence,
            elapsed_cycles,
            latency_us,
            ok
        };

        res_queue_.push(res);
    }
}

} // namespace kernel
} // namespace aeroedge
