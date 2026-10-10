# AeroEdge-RT: Autonomous Micro-Kernel & Quantized Inference Engine

## Project Identity & Academic Target
AeroEdge-RT is a deterministic, zero-allocation micro-kernel and INT8 tensor execution engine designed for constrained dual-core microcontrollers (ESP32-S3 / ESP32). It targets elite research standards suitable for presentation to top academic laboratories (e.g., ETH Zürich Computer Engineering & Networks Laboratory, Max Planck Institute for Software Systems, Stanford TinyML).

## Core Architectural Invariants

1. **Zero Dynamic Allocation (Anti-malloc)**:
   - `malloc()`, `calloc()`, `realloc()`, and `free()` are strictly forbidden in runtime inference loops.
   - All activations, scratch spaces, and circular ping-pong buffers must reside in static, 16-byte aligned memory (`alignas(16)`) within internal SRAM.
   - All model weights are declared `const` and mapped for Flash Execution-in-Place (XIP), consuming zero SRAM for weights.

2. **Strict Integer Fixed-Point Arithmetic (Zero Float)**:
   - Zero floating-point math instructions (`float`, `double`) inside the tensor compute kernels.
   - Arithmetic must utilize symmetric INT8 quantization with INT32 accumulators and fixed-point scale multipliers with bit-shift requantization.
   - Saturating arithmetic must be applied on all clamping operations to avoid silent numerical overflow.

3. **Asymmetric Dual-Core Isolation**:
   - **Core 0 (I/O & System Services)**: Hosts the FreeRTOS context, sensor streaming (DMA ping-pong), telemetry, and micro-ROS / UART communication.
   - **Core 1 (Isolated Math Engine)**: Dedicated bare-metal computational worker. Zero FreeRTOS task preemption, zero OS tick jitter, deterministic execution time.
   - **Inter-Core Communication**: Single-Producer Single-Consumer (SPSC) lock-free circular ring buffer with atomic head/tail pointers and direct hardware task notifications.

4. **Hardware SIMD Acceleration**:
   - Leverage 32-bit registers to pack four 8-bit operands (`int8x4`) for parallel dot-product and multiply-accumulate operations.
   - Optimize inner loops with 4-way and 8-way unrolling to maximize pipeline occupancy on the Xtensa LX7 architecture.

5. **Research-Grade Code Quality**:
   - C++17 standard, `-Wall -Wextra -Werror -O3 -fno-exceptions -fno-rtti`.
   - Comprehensive inline documentation with theoretical mathematical formulations, cycle complexity analysis, and cache footprint metrics.
