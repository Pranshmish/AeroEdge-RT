# AeroEdge-RT Benchmark: Experiment A (Baseline)

## 1. System & Architecture Configuration

| Parameter | Baseline Value |
| :--- | :--- |
| **SoC** | ESP32-D0WD-V3 (Xtensa Dual-Core LX6, Rev 3.1) |
| **Clock Frequency** | 160 MHz / 240 MHz |
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
| **Total Generation Time** | 9.150 seconds |
| **Inference Throughput** | **5.25 tok/s** |
| **Mean Latency per Token** | **190.6 ms / token** |
| **Preemption Jitter ($\Delta t$)** | **12.39 ms** |
| **Free Internal SRAM** | 186,540 bytes |
| **Heap Allocations** | 10 allocations + 10 frees per token (480 heap transactions) |

---

## 3. Bottleneck Analysis for Experiment A

1. **Heap Allocation Overhead & Fragmentation**:
   Calling `malloc()` and `free()` for `RunState` buffers (`x`, `xb`, `xb2`, `hb`, `hb2`, `q`, `k`, `v`, `att`, `logits`) introduces substantial heap lock contention (`vTaskSuspendAll()`) and memory jitter.
2. **Single-Core Scalar Execution**:
   Core 0 remains largely idle while Core 1 executes sequential matrix-vector multiplications for all projections ($W_q, W_k, W_v, W_o, W_1, W_2, W_3, W_{cls}$).
3. **No Instruction-Level Pipelining**:
   Scalar inner product loops incur FPU pipeline latency stalls between successive multiply and accumulate instructions.
