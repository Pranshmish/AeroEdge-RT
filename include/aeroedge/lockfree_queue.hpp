/**
 * @file lockfree_queue.hpp
 * @brief Lock-Free Asymmetric Inter-Core Ring Buffer (Core 0 <-> Core 1 IPC).
 * @details Implements a Single-Producer Single-Consumer (SPSC) wait-free ring buffer
 *          operating across asymmetric dual-core boundaries. Completely eliminates
 *          FreeRTOS mutex contention, priority inversions, and inter-core memory stalls.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <type_traits>

namespace aeroedge {
namespace ipc {

/**
 * @brief Command payload dispatched from Core 0 to Core 1.
 */
struct InferenceRequest {
    uint32_t request_id;
    const int8_t* input_tensor;
    size_t input_size;
    int32_t model_id;
    uint32_t timestamp_cycles;
};

/**
 * @brief Inference result payload returned from Core 1 to Core 0.
 */
struct InferenceResponse {
    uint32_t request_id;
    int32_t predicted_class;
    int8_t max_confidence;
    uint32_t execution_cycles;
    uint32_t latency_us;
    bool success;
};

/**
 * @brief Lock-Free SPSC Circular Queue for hardware core synchronization.
 * 
 * @tparam T Payload element type.
 * @tparam Capacity Power-of-two queue depth.
 */
template <typename T, size_t Capacity = 16>
class LockFreeQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable for zero-copy DMA IPC");

public:
    LockFreeQueue() : head_(0), tail_(0) {}

    /**
     * @brief Pushes an item into the queue (Called strictly by the Producer core).
     * 
     * @param item The element to push.
     * @return true if pushed successfully, false if queue was full.
     */
    bool push(const T& item) noexcept {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= Capacity) {
            return false; // Queue is full
        }

        buffer_[current_tail & BUFFER_MASK] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief Pops an item from the queue (Called strictly by the Consumer core).
     * 
     * @param item Output reference to receive the popped item.
     * @return true if popped successfully, false if queue was empty.
     */
    bool pop(T& item) noexcept {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false; // Queue is empty
        }

        item = buffer_[current_head & BUFFER_MASK];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] size_t size() const noexcept {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t t = tail_.load(std::memory_order_relaxed);
        return (t >= h) ? (t - h) : 0;
    }

private:
    static constexpr size_t BUFFER_MASK = Capacity - 1;

    // Cache-line aligned / Core-isolated atomic variables to prevent false sharing
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    alignas(16) T buffer_[Capacity];
};

} // namespace ipc
} // namespace aeroedge
