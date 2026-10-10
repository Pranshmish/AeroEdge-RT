# AeroEdge-RT Benchmark: Experiment A (Baseline)

## 1. System & Architecture Configuration

| Parameter | Baseline Value |
| :--- | :--- |
| **SoC** | ESP32-D0WD-V3 (Xtensa Dual-Core LX6, Rev 3.1) |
| **Clock Frequency** | 160 MHz |
| **RAM** | Internal SRAM (Zero external PSRAM) |
| **Model** | TinyLlama 260K (Stories Pretrained) |
| **Model Parameters** | 264,128 parameters (Float32) |
| **Weights Storage** | 1,056,540 bytes Zero-Copy Flash XIP (DROM mapped at `0x3F400020`) |
| **Context Dimensions** | `dim=64`, `hidden_dim=172`, `n_layers=5`, `n_heads=8`, `n_kv_heads=4`, `vocab=512` |
| **Scheduling Runtime** | Standard FreeRTOS Preemptive Task (Core 1) |
| **Memory Strategy** | Dynamic Heap `malloc()` and `free()` per token (Activation Anti-Pattern) |
| **Compute Execution** | Scalar Un-accelerated Xtensa FPU Single-Core Loops |

---

## 2. Real Hardware Performance Profile (UART Hardware Timer)

- **Test Prompt**: `"once upon a time"`
- **Generated Sequence**:
  > *"once upon a time, there was a little bird named Toby. Toby loved to paint with his friends. One day, Tommy went for a walk..."*

### Profiler Telemetry:

| Metric | Measured Baseline Value |
| :--- | :--- |
| **Tokens Generated** | 48 tokens |
| **Total Generation Time** | 8.216 seconds |
| **Inference Throughput** | **5.84 tok/s** |
| **Mean Latency per Token** | **171.2 ms / token** |
| **Preemption Jitter ($\Delta t$)** | **10.14 ms** |
| **Free Internal SRAM** | 166,508 bytes |
| **Heap Allocations** | 10 allocations + 10 frees per token (480 heap transactions) |

---

## 3. Bottleneck Analysis for Experiment A

1. **Heap Allocation Overhead & Fragmentation**:
   Calling `malloc()` and `free()` for `RunState` buffers (`x`, `xb`, `xb2`, `hb`, `hb2`, `q`, `k`, `v`, `att`, `logits`) introduces substantial heap lock contention (`vTaskSuspendAll()`) and memory jitter.
2. **Scalar Float32 Arithmetic on Xtensa LX6**:
   ESP32's Xtensa LX6 FPU is strictly scalar (no single-precision vector SIMD). Processing 260K float multiplications sequentially introduces high CPU cycle overhead.
3. **Flash SPI Cache Misses (XIP Thrashing)**:
   Accessing 1.05MB of weights across the 40MHz SPI bus causes frequent flash cache line evictions and CPU wait-states during matrix-vector inner products.
