import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load Experiment C Metrics
JSON_PATH = os.path.join(os.path.dirname(__file__), "experiment_c_model_scaling_metrics.json")
with open(JSON_PATH, "r") as f:
    data = json.load(f)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)
BRAIN_DIR = r"C:\Users\ASUS\.gemini\antigravity-ide\brain\accf4005-321a-45a2-8ab4-83f8f35577bd"

models = data["models"]
names = [m["name"] for m in models]
param_labels = ["260K", "1.05M", "3.20M"]
params_num = [m["architecture"]["params"] for m in models]

base_lat = [m["performance"]["baseline_freertos"]["latency_ms"] for m in models]
aero_lat = [m["performance"]["aeroedge_rt"]["latency_ms"] for m in models]
base_tok = [m["performance"]["baseline_freertos"]["throughput_tokens_sec"] for m in models]
aero_tok = [m["performance"]["aeroedge_rt"]["throughput_tokens_sec"] for m in models]
speedups = [m["performance"]["delta"]["speedup_factor"] for m in models]
arena_kb = [m["memory_footprint"]["static_tensor_arena_kb"] for m in models]
free_sram_kb = [m["memory_footprint"]["remaining_free_sram_kb"] for m in models]
flash_kb = [m["memory_footprint"]["weights_int8_kb"] for m in models]
energy_base = [m["performance"]["baseline_freertos"]["energy_per_token_mj"] for m in models]
energy_aero = [m["performance"]["aeroedge_rt"]["energy_per_token_mj"] for m in models]
cycles_base = [m["performance"]["baseline_freertos"]["cpu_cycles_per_token"] / 1e6 for m in models]
cycles_aero = [m["performance"]["aeroedge_rt"]["cpu_cycles_per_token"] / 1e6 for m in models]

# Style settings
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# ==============================================================================
# FIGURE 13: Model Scaling Curves (Throughput, Latency, Speedup)
# ==============================================================================
fig13, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
fig13.suptitle("EXPERIMENT C: SLM MODEL SCALING CURVES ON ESP32-WROOM (240 MHz, ZERO PSRAM)\nTinyLlama-260K  →  BabyLLaMA-1M  →  Stories-3.2M",
               fontsize=14, fontweight='bold', y=1.02)

# Panel 1: Throughput (tok/s)
x_pos = np.arange(len(names))
width = 0.35

ax = axes[0]
rects1 = ax.bar(x_pos - width/2, base_tok, width, label='Baseline FreeRTOS', color='#e74c3c', alpha=0.9, edgecolor='black')
rects2 = ax.bar(x_pos + width/2, aero_tok, width, label='AeroEdge-RT (Dual-Core SIMD)', color='#2ecc71', alpha=0.9, edgecolor='black')

ax.set_ylabel('Inference Throughput (tokens/s)', fontsize=11, fontweight='bold')
ax.set_title('(a) Token Throughput Scaling', fontsize=12, fontweight='bold', pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{n}\n({p})" for n, p in zip(names, param_labels)], fontsize=10)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, axis='y')

for r in rects1:
    h = r.get_height()
    ax.annotate(f"{h:.1f}", (r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#c0392b')
for r in rects2:
    h = r.get_height()
    ax.annotate(f"{h:.1f}", (r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#27ae60')

# Panel 2: Latency per Token (ms)
ax = axes[1]
ax.plot(x_pos, base_lat, 'o--', color='#e74c3c', linewidth=2.5, markersize=8, label='Baseline FreeRTOS')
ax.plot(x_pos, aero_lat, 's-', color='#27ae60', linewidth=2.5, markersize=8, label='AeroEdge-RT')

ax.set_ylabel('Latency per Token (ms/token)', fontsize=11, fontweight='bold')
ax.set_title('(b) Inference Latency vs Parameter Count', fontsize=12, fontweight='bold', pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{n}\n({p})" for n, p in zip(names, param_labels)], fontsize=10)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True)

for i, (b, a) in enumerate(zip(base_lat, aero_lat)):
    ax.annotate(f"{b:.1f} ms", (i, b), xytext=(-10, 10), textcoords="offset points", fontweight='bold', color='#c0392b', fontsize=9)
    ax.annotate(f"{a:.1f} ms", (i, a), xytext=(10, -15), textcoords="offset points", fontweight='bold', color='#27ae60', fontsize=9)

# Panel 3: Speedup Factor Curve
ax = axes[2]
bars = ax.bar(x_pos, speedups, width=0.5, color=['#3498db', '#9b59b6', '#e67e22'], edgecolor='black', alpha=0.9)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Baseline (1.0x)')
ax.set_ylabel('AeroEdge-RT Speedup Factor', fontsize=11, fontweight='bold')
ax.set_title('(c) Acceleration Speedup Factor', fontsize=12, fontweight='bold', pad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{n}\n({p})" for n, p in zip(names, param_labels)], fontsize=10)
ax.set_ylim(0, 4.2)
ax.grid(True, axis='y')

for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.2f}x\nFaster", (b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

plt.tight_layout()
fig13_png = os.path.join(FIG_DIR, "figure13_experiment_c_model_scaling_curves.png")
fig13_svg = os.path.join(FIG_DIR, "figure13_experiment_c_model_scaling_curves.svg")
fig13.savefig(fig13_png)
fig13.savefig(fig13_svg)
try:
    if os.path.exists(BRAIN_DIR):
        fig13.savefig(os.path.join(BRAIN_DIR, "figure13_experiment_c_model_scaling_curves.png"))
except Exception as e:
    print(f"Notice: skipped saving to BRAIN_DIR ({e})")
plt.close(fig13)
print("Saved Figure 13!")

# ==============================================================================
# FIGURE 14: 4-Stage Breakdown Across 3 Models
# ==============================================================================
fig14, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=300)
fig14.suptitle("EXPERIMENT C: 4-STAGE INFERENCE MICROARCHITECTURE BREAKDOWN ACROSS 3 MODELS\nComparing Single-Core Baseline FreeRTOS vs AeroEdge-RT Dual-Core SIMD",
               fontsize=13, fontweight='bold', y=1.02)

stages = [
    ("Stage 1\nRMSNorm+RoPE", "stage_1_rmsnorm_rope"),
    ("Stage 2\nQKV Attention", "stage_2_qkv_attention"),
    ("Stage 3\nSwiGLU FFN", "stage_3_swiglu_ffn"),
    ("Stage 4\nLogits+Sample", "stage_4_logits_sampling")
]
stage_names = [s[0] for s in stages]
x_s = np.arange(len(stages))
w = 0.35

colors_base = '#e74c3c'
colors_aero = '#27ae60'

for idx, m in enumerate(models):
    ax = axes[idx]
    s_base = [m["stage_breakdown_ms"][k]["baseline"] for _, k in stages]
    s_aero = [m["stage_breakdown_ms"][k]["aeroedge"] for _, k in stages]
    s_speed = [m["stage_breakdown_ms"][k]["speedup"] for _, k in stages]
    
    r1 = ax.bar(x_s - w/2, s_base, w, label='Baseline FreeRTOS', color=colors_base, alpha=0.85, edgecolor='black')
    r2 = ax.bar(x_s + w/2, s_aero, w, label='AeroEdge-RT', color=colors_aero, alpha=0.85, edgecolor='black')
    
    ax.set_title(f"({chr(97+idx)}) {m['name']} ({param_labels[idx]})\nBaseline: {m['performance']['baseline_freertos']['latency_ms']:.1f}ms | AeroEdge: {m['performance']['aeroedge_rt']['latency_ms']:.1f}ms",
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Stage Latency (ms)', fontsize=10, fontweight='bold')
    ax.set_xticks(x_s)
    ax.set_xticklabels(stage_names, fontsize=9)
    ax.grid(True, axis='y')
    ax.legend(loc='upper left', fontsize=9)
    
    # Annotate speedups
    for i, sp in enumerate(s_speed):
        max_h = max(s_base[i], s_aero[i])
        ax.annotate(f"{sp:.1f}x", (x_s[i], max_h), xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2980b9')

plt.tight_layout()
fig14_png = os.path.join(FIG_DIR, "figure14_experiment_c_stage_profiler_3models.png")
fig14_svg = os.path.join(FIG_DIR, "figure14_experiment_c_stage_profiler_3models.svg")
fig14.savefig(fig14_png)
fig14.savefig(fig14_svg)
try:
    if os.path.exists(BRAIN_DIR):
        fig14.savefig(os.path.join(BRAIN_DIR, "figure14_experiment_c_stage_profiler_3models.png"))
except Exception as e:
    print(f"Notice: skipped saving fig14 to BRAIN_DIR ({e})")
plt.close(fig14)
print("Saved Figure 14!")

# ==============================================================================
# FIGURE 15: Master Hardware Physics & Feasibility Dashboard
# ==============================================================================
fig15, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
fig15.suptitle("EXPERIMENT C MASTER DASHBOARD: HARDWARE PHYSICS & SCALING VERIFICATION ON ESP32-WROOM\nVerification of Pure Internal SRAM Pipeline (Zero PSRAM) Across 260K, 1M, and 3.2M Models",
               fontsize=14, fontweight='bold', y=0.98)

# Subplot (0,0): SRAM Tensor Arena vs Remaining Usable SRAM
ax = axes[0, 0]
arena = np.array(arena_kb)
free_s = np.array(free_sram_kb)
p1 = ax.bar(x_pos, arena, width=0.45, label='Static Tensor Arena (Allocated)', color='#e67e22', edgecolor='black')
p2 = ax.bar(x_pos, free_s, width=0.45, bottom=arena, label='Remaining Usable SRAM (Free)', color='#2ecc71', edgecolor='black')
ax.axhline(186.0, color='red', linestyle='--', linewidth=1.5, label='Total Free Usable SRAM Limit (186 KB)')

ax.set_ylabel('Internal SRAM Budget (KB)', fontsize=11, fontweight='bold')
ax.set_title('(a) Internal SRAM Allocation & Headroom (Zero PSRAM)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{n}\n({p})" for n, p in zip(names, param_labels)], fontsize=10)
ax.set_ylim(0, 220)
ax.grid(True, axis='y')
ax.legend(loc='lower left', frameon=True, facecolor='white')

for i in range(len(names)):
    ax.annotate(f"Arena:\n{arena[i]:.1f} KB", (x_pos[i], arena[i]/2), ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    ax.annotate(f"Free SRAM:\n{free_s[i]:.1f} KB", (x_pos[i], arena[i] + free_s[i]/2), ha='center', va='center',
                fontsize=9, fontweight='bold', color='#1e392a')

# Subplot (0,1): Flash ROM Storage Breakdown (4MB SPI Flash)
ax = axes[0, 1]
flash_weights = np.array(flash_kb)
flash_other = np.array([4096 - fw for fw in flash_weights])
p1 = ax.bar(x_pos, flash_weights, width=0.45, label='INT8 Model Weights in Flash ROM', color='#3498db', edgecolor='black')
p2 = ax.bar(x_pos, flash_other, width=0.45, bottom=flash_weights, label='Available Flash Storage', color='#ecf0f1', edgecolor='black')
ax.axhline(4096.0, color='black', linestyle=':', linewidth=1.5, label='Total 4MB Flash Boundary (4096 KB)')

ax.set_ylabel('Flash ROM Budget (KB)', fontsize=11, fontweight='bold')
ax.set_title('(b) 4MB Flash ROM Storage Partitioning', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{n}\n({p})" for n, p in zip(names, param_labels)], fontsize=10)
ax.set_ylim(0, 4800)
ax.grid(True, axis='y')
ax.legend(loc='upper left', frameon=True, facecolor='white')

for i in range(len(names)):
    pct = (flash_weights[i] / 4096.0) * 100
    ax.annotate(f"{flash_weights[i]:.0f} KB\n({pct:.1f}% Flash)", (x_pos[i], flash_weights[i] + 100),
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1f4e78')

# Subplot (1,0): Energy Consumption per Token (mJ)
ax = axes[1, 0]
r1 = ax.bar(x_pos - width/2, energy_base, width, label='Baseline FreeRTOS', color='#e74c3c', edgecolor='black', alpha=0.9)
r2 = ax.bar(x_pos + width/2, energy_aero, width, label='AeroEdge-RT', color='#27ae60', edgecolor='black', alpha=0.9)

ax.set_ylabel('Energy per Token (mJ/token)', fontsize=11, fontweight='bold')
ax.set_title('(c) Energy Consumption per Generated Token', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{n}\n({p})" for n, p in zip(names, param_labels)], fontsize=10)
ax.grid(True, axis='y')
ax.legend(frameon=True, facecolor='white')

for i in range(len(names)):
    saving = ((energy_base[i] - energy_aero[i]) / energy_base[i]) * 100
    ax.annotate(f"-{saving:.1f}%\nEnergy", (x_pos[i] + width/2, energy_aero[i]), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#196f3d')

# Subplot (1,1): CPU Cycle Consumption per Token (Millions of Cycles)
ax = axes[1, 1]
r1 = ax.bar(x_pos - width/2, cycles_base, width, label='Baseline FreeRTOS', color='#95a5a6', edgecolor='black', alpha=0.9)
r2 = ax.bar(x_pos + width/2, cycles_aero, width, label='AeroEdge-RT', color='#8e44ad', edgecolor='black', alpha=0.9)

ax.set_ylabel('CPU Cycles per Token (Millions)', fontsize=11, fontweight='bold')
ax.set_title('(d) CPU Execution Cycles per Token (@ 240 MHz)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{n}\n({p})" for n, p in zip(names, param_labels)], fontsize=10)
ax.grid(True, axis='y')
ax.legend(frameon=True, facecolor='white')

for i in range(len(names)):
    ax.annotate(f"{cycles_base[i]:.1f}M", (x_pos[i] - width/2, cycles_base[i]), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#555555')
    ax.annotate(f"{cycles_aero[i]:.1f}M", (x_pos[i] + width/2, cycles_aero[i]), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#5b2c6f')

plt.tight_layout()
fig15_png = os.path.join(FIG_DIR, "figure15_master_experiment_c_dashboard.png")
fig15_svg = os.path.join(FIG_DIR, "figure15_master_experiment_c_dashboard.svg")
fig15.savefig(fig15_png)
fig15.savefig(fig15_svg)
try:
    if os.path.exists(BRAIN_DIR):
        fig15.savefig(os.path.join(BRAIN_DIR, "figure15_master_experiment_c_dashboard.png"))
except Exception as e:
    print(f"Notice: skipped saving fig15 to BRAIN_DIR ({e})")
plt.close(fig15)
print("Saved Figure 15!")
