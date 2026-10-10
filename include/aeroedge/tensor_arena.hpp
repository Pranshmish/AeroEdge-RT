/**
 * @file tensor_arena.hpp
 * @brief Deterministic Zero-Allocation Tensor Arena & Static Memory Fabric.
 * @details Enforces strict zero-malloc runtime operation. Allocates all intermediate
 *          feature maps and activation ping-pong buffers in contiguous, 16-byte aligned
 *          internal SRAM at compile time.
 * 
 * Target: High-speed 240MHz Internal SRAM (Zero SPI bus stall).
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>

namespace aeroedge {
namespace memory {

constexpr size_t DEFAULT_ARENA_CAPACITY = 64 * 1024; // 64 Kilobytes Internal SRAM
constexpr size_t MEMORY_ALIGNMENT_BYTES = 16;        // 128-bit vector memory alignment

/**
 * @brief Static zero-allocation memory arena for deterministic tensor buffers.
 *        Guarantees 16-byte alignment and 0.00% heap fragmentation.
 */
template <size_t Capacity = DEFAULT_ARENA_CAPACITY>
class TensorArena {
public:
    TensorArena() : offset_(0), peak_usage_(0) {
        reset();
    }

    // Prohibit dynamic copies and moves
    TensorArena(const TensorArena&) = delete;
    TensorArena& operator=(const TensorArena&) = delete;

    /**
     * @brief Allocates an aligned buffer slice within the static SRAM fabric.
     *        Only permitted during model graph initialization phase.
     * 
     * @param bytes Number of contiguous bytes requested.
     * @return void* Aligned pointer to the allocated memory, or nullptr if exhausted.
     */
    void* allocate(size_t bytes) noexcept {
        // Compute alignment padding
        size_t current_addr = reinterpret_cast<size_t>(storage_ + offset_);
        size_t padding = (MEMORY_ALIGNMENT_BYTES - (current_addr % MEMORY_ALIGNMENT_BYTES)) % MEMORY_ALIGNMENT_BYTES;
        size_t total_needed = padding + bytes;

        if (offset_ + total_needed > Capacity) {
            return nullptr; // Static budget exceeded
        }

        offset_ += padding;
        void* ptr = static_cast<void*>(storage_ + offset_);
        offset_ += bytes;

        if (offset_ > peak_usage_) {
            peak_usage_ = offset_;
        }
        return ptr;
    }

    /**
     * @brief Zeroes out all active arena memory.
     */
    void reset() noexcept {
        offset_ = 0;
        std::memset(storage_, 0, Capacity);
    }

    [[nodiscard]] constexpr size_t capacity() const noexcept { return Capacity; }
    [[nodiscard]] size_t used_bytes() const noexcept { return offset_; }
    [[nodiscard]] size_t free_bytes() const noexcept { return Capacity - offset_; }
    [[nodiscard]] size_t peak_usage() const noexcept { return peak_usage_; }

    /**
     * @brief Verifies that the internal memory address satisfies 16-byte alignment.
     */
    [[nodiscard]] bool is_aligned(const void* ptr) const noexcept {
        return (reinterpret_cast<uintptr_t>(ptr) % MEMORY_ALIGNMENT_BYTES) == 0;
    }

private:
    alignas(16) uint8_t storage_[Capacity];
    size_t offset_;
    size_t peak_usage_;
};

/**
 * @brief Zero-Copy Ping-Pong Tensor Buffer Fabric.
 *        Permits alternating layer execution without allocating new memory.
 *        Layer i reads from primary and writes to secondary; Layer i+1 swaps roles.
 */
struct PingPongBuffers {
    int8_t* buf_a;
    int8_t* buf_b;
    size_t buffer_size;
    bool current_is_a;

    void swap() noexcept {
        current_is_a = !current_is_a;
    }

    [[nodiscard]] int8_t* get_read_buffer() const noexcept {
        return current_is_a ? buf_a : buf_b;
    }

    [[nodiscard]] int8_t* get_write_buffer() const noexcept {
        return current_is_a ? buf_b : buf_a;
    }
};

} // namespace memory
} // namespace aeroedge
