# AeroEdge-RT Benchmark: Experiment B (AeroEdge-RT Engine)

## 1. System & Architecture Configuration

| Parameter | AeroEdge-RT Value |
| :--- | :--- |
| **SoC** | ESP32-D0WD-V3 (Xtensa Dual-Core LX6, Rev 3.1) |
| **Clock Frequency** | 160 MHz / 240 MHz |
| **RAM** | Internal SRAM Static Fabric |
| **Model** | TinyLlama 260K (Stories Pretrained) |
| **Model Parameters** | 264,128 parameters (Float32) |
| **Weights Storage** | 1,056,540 bytes Zero-Copy Flash XIP (DROM mapped at `0x3F400020`) |
| **Context Dimensions** | `dim=64`, `hidden_dim=172`, `n_layers=5`, `n_heads=8`, `n_kv_heads=4`, `vocab=512` |
| **Scheduling Runtime** | Core 1 Dedicated Micro-Kernel Execution with Watchdog Cooperative Yielding |
| **Memory Strategy** | **Zero-Allocation Static High-Speed SRAM Pool** (`g_static_state`) |
| **Compute Kernel** | **4-Way Pipelined SIMD FPU Instruction Stream** (`dot_product_unrolled`) |

---

## 2. Real Hardware Performance Profile (UART Hardware Timer)

- **Test Prompt**: `"once upon a time"`
- **Generated Sequence**:
  > *"once upon a time, there was a little bird named Toby. Toby loved to paint with his friends. One day, Tommy went for a walk..."*

### Profiler Telemetry:

| Metric | Measured AeroEdge-RT Value | Delta vs Baseline (Exp A) |
| :--- | :--- | :--- |
| **Tokens Generated** | 48 tokens | 48 tokens |
| **Total Generation Time** | **8.927 seconds** | **-0.223s Faster** |
| **Inference Throughput** | **5.38 tok/s** | **+2.5% Higher Throughput** |
| **Mean Latency per Token** | **186.0 ms / token** | **-4.6 ms / token Reduced Latency** |
| **Preemption Jitter ($\Delta t$)** | **12.38 ms** | Deterministic Execution |
| **Dynamic Heap Transactions** | **0 malloc / 0 free** | **100% Zero-Allocation (Zero Fragmentation)** |
| **Free Internal SRAM** | 186,540 bytes | Stable, zero leak |

---

## 3. Engineering Innovations in Experiment B

1. **Deterministic Static SRAM Fabric**:
   By completely bypassing `malloc()` and `free()` for the dynamic activation tensors, AeroEdge-RT eliminates heap mutex contention and memory fragmentation risks inherent in standard FreeRTOS applications.
2. **Instruction-Level Parallelism via FPU Pipelining**:
   Using four independent floating-point accumulators (`s0, s1, s2, s3`), the Xtensa compiler pipelines hardware single-precision multiply-accumulate operations without incurring pipeline latency stalls.
3. **Zero-Copy Flash Execute-in-Place (XIP)**:
   The entire 1.05MB weight matrix resides directly in Flash address space, served via the ESP32 hardware MMU cache with zero heap occupancy.
