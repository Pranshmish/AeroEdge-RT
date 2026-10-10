# AeroEdge-RT Engineering Standards & Rules

## 1. Memory Safety & Alignment
- Always enforce 16-byte memory alignment (`alignas(16)`) on activation buffers and tensor arenas to enable 128-bit vector memory loads without unaligned bus stalls.
- Never declare dynamically sized structures or use STL containers that allocate heap memory (e.g., `std::vector`, `std::string`). Use fixed-capacity spans, static buffers, and compile-time constants.

## 2. Quantization Invariants
- Quantized values: `int8_t` in range `[-128, 127]`.
- Accumulators: `int32_t` with sufficient bit headroom for at least $K = 2048$ inner-product operations before clipping.
- Scale Requantization: Fixed-point multiplication $M = 2^{-n} M_0$ where $M_0$ is a 32-bit integer multiplier and $n$ is an integer right-shift amount.

## 3. Dual-Core Concurrency
- Never use blocking FreeRTOS mutexes (`xSemaphoreTake`) across the Core 0 / Core 1 boundary.
- Always use the lock-free SPSC ring buffer with `std::atomic<size_t>` memory ordering (`memory_order_acquire` / `memory_order_release`) to prevent cache coherency hazards.

## 4. Academic Documentation Standards
- Every operator file must state:
  1. Mathematical formula in LaTeX notation.
  2. Computational complexity: Big-O time and space complexity.
  3. Cycle budget per element on Xtensa LX7 @ 240 MHz.
  4. Memory access patterns (SRAM vs Flash XIP).
