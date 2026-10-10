---
name: aeroedge-benchmark-profiler
description: >-
  Run, capture, and analyze execution latency, memory footprint, and core jitter benchmarks for the AeroEdge-RT micro-kernel across dual-core ESP32-S3 hardware and host simulation.
---

# AeroEdge-RT Benchmark Profiler

Use this skill to execute and evaluate the three proof-of-concept benchmarks required for high-impact edge AI research papers and presentations.

## Available Benchmark Suites

1. **Benchmark 1: Latency Comparison (AeroEdge-RT vs TFLite Micro)**
   - Measure cycle counts and wall-clock microseconds for INT8 Conv2D, DepthwiseConv2D, and Linear MatMul.
   - Script: `benchmarks/serial_latency_logger.py`

2. **Benchmark 2: SRAM vs. PSRAM Memory Throughput**
   - Measure bus contention and memory stall cycles when activations are placed in internal SRAM (240 MHz) versus external SPI PSRAM (40/80 MHz).
   - Script: `benchmarks/memory_profile.py`

3. **Benchmark 3: Core Jitter Elimination (OS Determinism)**
   - Calculate latency variance ($\sigma^2$, min, max, p50, p95, p99) over 10,000 continuous inference iterations on Core 1 while Core 0 executes heavy background I/O tasks.
   - Prove variance $< 0.1\%$ to substantiate real-time robotics guarantees.

## Quick Execution Workflow

```bash
# Flash and run serial latency capture
python benchmarks/serial_latency_logger.py --port COM3 --baud 115200 --iterations 1000

# Analyze SRAM vs PSRAM memory layout
python benchmarks/memory_profile.py --target esp32s3
```
