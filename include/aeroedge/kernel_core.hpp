/**
 * @file kernel_core.hpp
 * @brief AeroEdge-RT Asymmetric Dual-Core Dispatcher & Hardware Lifecycle Manager.
 * @details Implements asymmetric core partitioning:
 *          - CORE 0: FreeRTOS context, I/O dispatch, sensor DMA ping-pong, micro-ROS hub.
 *          - CORE 1: Isolated Bare-Metal Compute Worker, zero FreeRTOS tick jitter.
 * 
 * Hardware Target: ESP32-S3 (Xtensa LX7 Dual-Core @ 240MHz) / ESP32
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include "lockfree_queue.hpp"
#include "tensor_arena.hpp"

#ifdef ESP_PLATFORM
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_timer.h"
#endif

namespace aeroedge {
namespace kernel {

/**
 * @brief Real-time runtime execution telemetry.
 */
struct KernelTelemetry {
    uint32_t total_inferences;
    uint32_t min_cycles;
    uint32_t max_cycles;
    uint64_t total_cycles;
    uint32_t last_latency_us;
    uint32_t queue_drops;
    size_t sram_peak_bytes;
};

/**
 * @brief High-precision cycle counter abstraction across embedded hardware and host.
 */
inline uint32_t read_cpu_cycles() noexcept {
#ifdef ESP_PLATFORM
    return esp_cpu_get_cycle_count();
#elif defined(__x86_64__) || defined(_M_X64)
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return lo;
#else
    return 0;
#endif
}

/**
 * @brief Asymmetric Dual-Core Kernel Controller.
 */
class AeroEdgeKernel {
public:
    static AeroEdgeKernel& instance() noexcept {
        static AeroEdgeKernel s_instance;
        return s_instance;
    }

    /**
     * @brief Bootstraps the dual-core asymmetric runtime fabric.
     *        Initializes static arenas, spawns Core 1 isolated worker.
     */
    bool init() noexcept;

    /**
     * @brief Dispatches an inference workload from Core 0 to Core 1 via lock-free queue.
     */
    bool dispatch_async(const ipc::InferenceRequest& req) noexcept;

    /**
     * @brief Polls Core 1 completion queue from Core 0 (Zero-copy, non-blocking).
     */
    bool poll_result(ipc::InferenceResponse& res) noexcept;

    /**
     * @brief Returns current real-time telemetry metrics.
     */
    [[nodiscard]] KernelTelemetry get_telemetry() const noexcept {
        return telemetry_;
    }

    /**
     * @brief Returns reference to static SRAM tensor arena.
     */
    memory::TensorArena<>& arena() noexcept {
        return arena_;
    }

    /**
     * @brief Dedicated Core 1 computation execution loop.
     *        Executes continuously with zero FreeRTOS tick preemption.
     */
    void core1_worker_loop() noexcept;

private:
    AeroEdgeKernel() : initialized_(false), is_running_(false), telemetry_{0, 0xFFFFFFFF, 0, 0, 0, 0, 0} {}
    ~AeroEdgeKernel() = default;

    AeroEdgeKernel(const AeroEdgeKernel&) = delete;
    AeroEdgeKernel& operator=(const AeroEdgeKernel&) = delete;

    memory::TensorArena<64 * 1024> arena_;
    ipc::LockFreeQueue<ipc::InferenceRequest, 16> req_queue_;
    ipc::LockFreeQueue<ipc::InferenceResponse, 16> res_queue_;

    volatile bool initialized_;
    volatile bool is_running_;
    KernelTelemetry telemetry_;

#ifdef ESP_PLATFORM
    TaskHandle_t core1_task_handle_ = nullptr;
#endif
};

} // namespace kernel
} // namespace aeroedge
