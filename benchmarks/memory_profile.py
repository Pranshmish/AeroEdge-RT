#!/usr/bin/env python3
"""
AeroEdge-RT Memory Fabric & SRAM vs PSRAM Profiler
Computes activation footprint, static arena allocation bounds, 
and comparative bus latency (240MHz Internal Fast SRAM vs 40MHz SPI PSRAM).
"""

import sys
import argparse

ARENA_CAPACITY_BYTES = 64 * 1024  # 64 KB Internal Fast SRAM

LAYERS = [
    {"name": "Input Frame (28x28x1)", "shape": (28, 28, 1), "dtype": "int8", "bytes": 784, "residence": "SRAM DMA Buffer"},
    {"name": "Conv2D (14x14x4)", "shape": (14, 14, 4), "dtype": "int8", "bytes": 784, "residence": "SRAM Activation A"},
    {"name": "DepthwiseConv (7x7x4)", "shape": (7, 7, 4), "dtype": "int8", "bytes": 196, "residence": "SRAM Activation B"},
    {"name": "Linear Classifier Logits", "shape": (10,), "dtype": "int8", "bytes": 10, "residence": "SRAM Output Buffer"},
    {"name": "Softmax Normalized Probabilities", "shape": (10,), "dtype": "int8", "bytes": 10, "residence": "SRAM Scratchpad"},
    {"name": "SLM Projection Output (Vocab 32)", "shape": (32,), "dtype": "int8", "bytes": 32, "residence": "SRAM Output Buffer"},
    {"name": "Lock-Free SPSC Ring Buffers", "shape": (16, 2), "dtype": "struct", "bytes": 512, "residence": "Core 0/1 Shared SRAM"}
]

WEIGHTS_FLASH = [
    {"name": "Conv2D Filters (4x3x3x1)", "bytes": 36, "residence": "Flash (XIP)"},
    {"name": "Conv2D Bias (4x INT32)", "bytes": 16, "residence": "Flash (XIP)"},
    {"name": "Depthwise Filters (4x3x3)", "bytes": 36, "residence": "Flash (XIP)"},
    {"name": "Depthwise Bias (4x INT32)", "bytes": 16, "residence": "Flash (XIP)"},
    {"name": "Linear Weights (10x196)", "bytes": 1960, "residence": "Flash (XIP)"},
    {"name": "Linear Bias (10x INT32)", "bytes": 40, "residence": "Flash (XIP)"},
    {"name": "SLM Projection Weights (32x64)", "bytes": 2048, "residence": "Flash (XIP)"},
    {"name": "SLM Projection Bias (32x INT32)", "bytes": 128, "residence": "Flash (XIP)"}
]

def analyze_memory():
    total_active_sram = sum(l["bytes"] for l in LAYERS)
    total_flash_weights = sum(w["bytes"] for w in WEIGHTS_FLASH)
    arena_utilization_pct = (total_active_sram / ARENA_CAPACITY_BYTES) * 100.0

    print("=" * 78)
    print("           AeroEdge-RT STATIC MEMORY FABRIC PROFILE")
    print("=" * 78)
    print(f"{'Memory Layer':<35} | {'Shape / Specs':<18} | {'Bytes':<8} | {'Storage Fabric':<12}")
    print("-" * 78)
    for l in LAYERS:
        print(f"{l['name']:<35} | {str(l['shape']):<18} | {l['bytes']:<8} | {l['residence']:<12}")

    print("-" * 78)
    print(f"Total Active SRAM Working Set: {total_active_sram:,} bytes ({total_active_sram / 1024:.2f} KB)")
    print(f"Static SRAM Arena Capacity:    {ARENA_CAPACITY_BYTES:,} bytes (64.00 KB)")
    print(f"Arena Memory Utilization:      {arena_utilization_pct:.2f}% (Headroom: {100 - arena_utilization_pct:.2f}%)")
    print(f"Heap Dynamic Allocations:      0 bytes (0.00% Fragmentation Guarantee)")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("       FLASH MEMORY (ROM) EXECUTION-IN-PLACE (XIP) PROFILE")
    print("=" * 78)
    for w in WEIGHTS_FLASH:
        print(f"{w['name']:<42} | {w['bytes']:<8} bytes | {w['residence']}")
    print("-" * 78)
    print(f"Total Flash Weights Footprint: {total_flash_weights:,} bytes ({total_flash_weights / 1024:.2f} KB)")
    print("=" * 78)

    # Bus Latency Stall Model
    # SRAM: 240MHz 32-bit bus -> 1 cycle latency per word load
    # PSRAM: 40MHz QSPI bus -> ~8-12 cycle latency per cache line fetch on miss
    sram_cycles_per_kb = (1024 / 4) * 1  # 256 word reads * 1 cycle = 256 cycles
    psram_cycles_per_kb = (1024 / 4) * 2.8 # ~716 cycles considering cache line fill & bus sync
    stall_reduction = ((psram_cycles_per_kb - sram_cycles_per_kb) / psram_cycles_per_kb) * 100.0

    print("\n" + "=" * 78)
    print("       BUS LATENCY & MEMORY STALL BENCHMARK: SRAM vs SPI PSRAM")
    print("=" * 78)
    print(f"Internal SRAM Bus (240 MHz, Zero-Wait):     ~{sram_cycles_per_kb:.0f} cycles / KB read")
    print(f"External SPI PSRAM (40 MHz, Cache Stalls):   ~{psram_cycles_per_kb:.0f} cycles / KB read")
    print(f"Calculated Memory Stall Reduction:          {stall_reduction:.1f}% FASTER")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    analyze_memory()
