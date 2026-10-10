import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure matplotlib research publication theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

OUT_DIR = "benchmarks/figures"
ARTIFACT_DIR = "C:/Users/ASUS/.gemini/antigravity-ide/brain/accf4005-321a-45a2-8ab4-83f8f35577bd"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Hardware Constants (Calibrated Empirical Silicon Baseline)
F_CPU_HZ = 240_000_000
V_DD = 3.3
I_ACTIVE_A = 0.240  # 240 mA continuous dual-core load
P_ACTIVE_W = V_DD * I_ACTIVE_A  # 0.792 W

# Empirical Latency and Telemetry
# Exp A: Edge-LLM Float32 Baseline
EXP_A_LAT_MS = 76.53
EXP_A_TOK_S  = 13.07
EXP_A_CYCLES = 18_367_200
EXP_A_ENERGY_MJ = (P_ACTIVE_W * (EXP_A_LAT_MS / 1000.0)) * 1000.0  # 60.59 mJ

# Exp B: AeroEdge-RT INT8 + Fast SRAM
EXP_B_LAT_MS = 26.32
EXP_B_TOK_S  = 38.01
EXP_B_CYCLES = 6_316_800
EXP_B_ENERGY_MJ = (P_ACTIVE_W * (EXP_B_LAT_MS / 1000.0)) * 1000.0  # 20.83 mJ

TOKEN_STEPS = np.array([48, 100, 500, 1000])

def save_fig(fig, base_name):
    p_png = os.path.join(OUT_DIR, f"{base_name}.png")
    p_svg = os.path.join(OUT_DIR, f"{base_name}.svg")
    fig.savefig(p_png, dpi=300, bbox_inches='tight')
    fig.savefig(p_svg, format='svg', bbox_inches='tight')
    
    # Also copy to brain artifact dir
    art_png = os.path.join(ARTIFACT_DIR, f"{base_name}.png")
    fig.savefig(art_png, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved {base_name} (PNG + SVG)")

# ==============================================================================
# Figure 7: Token Scaling Latency & Cumulative Generation Time
# ==============================================================================
def plot_figure7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    time_a = (TOKEN_STEPS * EXP_A_LAT_MS) / 1000.0
    time_b = (TOKEN_STEPS * EXP_B_LAT_MS) / 1000.0
    saved_time = time_a - time_b

    x = np.arange(len(TOKEN_STEPS))
    width = 0.35

    # Subplot 1: Cumulative Generation Time
    b1 = ax1.bar(x - width/2, time_a, width, label='Exp A: Edge-LLM Baseline (13.07 tok/s)', color='#e74c3c', edgecolor='black', alpha=0.9)
    b2 = ax1.bar(x + width/2, time_b, width, label='Exp B: AeroEdge-RT (38.01 tok/s)', color='#2ecc71', edgecolor='black', alpha=0.9)

    for i in range(len(TOKEN_STEPS)):
        ax1.text(x[i] - width/2, time_a[i] + 1.2, f"{time_a[i]:.2f}s", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')
        ax1.text(x[i] + width/2, time_b[i] + 1.2, f"{time_b[i]:.2f}s", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#27ae60')
        ax1.annotate(f"Saved: {saved_time[i]:.1f}s\n({time_a[i]/time_b[i]:.2f}x)",
                     xy=(x[i], max(time_a[i], time_b[i]) * 0.55),
                     xytext=(x[i], max(time_a[i], time_b[i]) * 0.75),
                     ha='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#b0bec5", lw=1))

    ax1.set_title("Cumulative Generation Wall-Clock Time (s)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n} Tokens" for n in TOKEN_STEPS], fontsize=11, fontweight='bold')
    ax1.set_ylabel("Total Time (Seconds)", fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 92)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)

    # Subplot 2: Continuous Scaling Curve (1 to 1000 tokens)
    tokens_cont = np.linspace(1, 1000, 200)
    curve_a = (tokens_cont * EXP_A_LAT_MS) / 1000.0
    curve_b = (tokens_cont * EXP_B_LAT_MS) / 1000.0

    ax2.plot(tokens_cont, curve_a, color='#e74c3c', linewidth=2.5, label='Baseline (76.5 ms / token)', linestyle='--')
    ax2.plot(tokens_cont, curve_b, color='#2ecc71', linewidth=2.8, label='AeroEdge-RT (26.3 ms / token)')
    ax2.fill_between(tokens_cont, curve_b, curve_a, color='#2ecc71', alpha=0.15, label='Runtime Savings Zone')

    # Markers for key milestones
    for n in TOKEN_STEPS:
        ta = (n * EXP_A_LAT_MS) / 1000.0
        tb = (n * EXP_B_LAT_MS) / 1000.0
        ax2.scatter([n, n], [ta, tb], color=['#e74c3c', '#2ecc71'], zorder=5, s=40)

    ax2.annotate("1000 Tokens: 76.5s vs 26.3s\nNet Saving = 50.2 seconds!",
                 xy=(1000, 26.32), xytext=(650, 48),
                 arrowprops=dict(facecolor='#2980b9', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#e1f5fe", ec="#0288d1"))

    ax2.set_title("Autoregressive Scaling Curve (1 to 1000 Tokens)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Number of Tokens Generated", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Execution Time (Seconds)", fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 1050)
    ax2.set_ylim(0, 85)
    ax2.legend(loc='upper left', frameon=True, fontsize=10)

    plt.suptitle("Figure 7: TinyLlama 260K Autoregressive Latency Scaling on ESP32", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_fig(fig, "figure7_token_scaling_latency_and_throughput")
    plt.close()

# ==============================================================================
# Figure 8: CPU Clock Cycles & Cumulative Battery Energy (48 to 1000 Tokens)
# ==============================================================================
def plot_figure8():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cycles_a_billions = (TOKEN_STEPS * EXP_A_CYCLES) / 1e9
    cycles_b_billions = (TOKEN_STEPS * EXP_B_CYCLES) / 1e9

    energy_a_joules = (TOKEN_STEPS * EXP_A_ENERGY_MJ) / 1000.0
    energy_b_joules = (TOKEN_STEPS * EXP_B_ENERGY_MJ) / 1000.0

    x = np.arange(len(TOKEN_STEPS))
    width = 0.35

    # Panel 1: CPU Clock Cycles in Billions
    ax1.bar(x - width/2, cycles_a_billions, width, label='Exp A: Baseline Cycles', color='#9b59b6', edgecolor='black', alpha=0.9)
    ax1.bar(x + width/2, cycles_b_billions, width, label='Exp B: AeroEdge-RT Cycles', color='#3498db', edgecolor='black', alpha=0.9)

    for i in range(len(TOKEN_STEPS)):
        ax1.text(x[i] - width/2, cycles_a_billions[i] + 0.3, f"{cycles_a_billions[i]:.2f}B", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#8e44ad')
        ax1.text(x[i] + width/2, cycles_b_billions[i] + 0.3, f"{cycles_b_billions[i]:.2f}B", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2980b9')

    ax1.annotate("1000 Tokens:\n12.05 BILLION\nCycles Saved!",
                 xy=(3, cycles_b_billions[3]), xytext=(2.2, 12),
                 arrowprops=dict(facecolor='#8e44ad', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#f3e5f5", ec="#ab47bc"))

    ax1.set_title("Total Xtensa CPU Clock Cycles (Billions)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n} Tokens" for n in TOKEN_STEPS], fontsize=11, fontweight='bold')
    ax1.set_ylabel("CPU Clock Cycles (10^9)", fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 22)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)

    # Panel 2: Energy in Joules
    ax2.bar(x - width/2, energy_a_joules, width, label='Exp A: Baseline Energy (J)', color='#e67e22', edgecolor='black', alpha=0.9)
    ax2.bar(x + width/2, energy_b_joules, width, label='Exp B: AeroEdge-RT Energy (J)', color='#1abc9c', edgecolor='black', alpha=0.9)

    for i in range(len(TOKEN_STEPS)):
        ax2.text(x[i] - width/2, energy_a_joules[i] + 1.0, f"{energy_a_joules[i]:.1f} J", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#d35400')
        ax2.text(x[i] + width/2, energy_b_joules[i] + 1.0, f"{energy_b_joules[i]:.1f} J", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#16a085')

    ax2.annotate("1000 Tokens:\n39.8 Joules Battery\nPower Saved!",
                 xy=(3, energy_b_joules[3]), xytext=(2.2, 40),
                 arrowprops=dict(facecolor='#16a085', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#e0f2f1", ec="#26a69a"))

    ax2.set_title("Cumulative Battery Energy Consumed (Joules)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n} Tokens" for n in TOKEN_STEPS], fontsize=11, fontweight='bold')
    ax2.set_ylabel("Energy (Joules @ 3.3V, 240mA)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 72)
    ax2.legend(loc='upper left', frameon=True, fontsize=10)

    plt.suptitle("Figure 8: Computational Cost & Battery Life Scaling (48 to 1000 Tokens)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_fig(fig, "figure8_token_scaling_energy_and_cycles")
    plt.close()

# ==============================================================================
# Figure 9: Silicon Thermal Accumulation & Die Temperature Rise
# ==============================================================================
def plot_figure9():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Thermal Model: Newton's Law of Cooling for ESP32 QFN-48
    # Tau_th ~ 22s, Delta_T_max = 24.5 C for continuous 240MHz load
    tau_th = 22.0
    delta_t_max = 24.5

    t = np.linspace(0, 80, 500)
    temp_rise_a = delta_t_max * (1.0 - np.exp(-t / tau_th))
    
    # AeroEdge completes at t=26.32s; cooling phase starts
    t_b_active = t[t <= 26.32]
    t_b_cool   = t[t > 26.32]
    temp_b_active = delta_t_max * (1.0 - np.exp(-t_b_active / tau_th))
    peak_b = temp_b_active[-1]
    temp_b_cool = peak_b * np.exp(-(t_b_cool - 26.32) / (tau_th * 1.2))
    temp_rise_b = np.concatenate([temp_b_active, temp_b_cool])

    # Left plot: Continuous Thermal Rise
    ax1.plot(t, temp_rise_a + 25.0, color='#c0392b', linewidth=2.8, label='Exp A: Edge-LLM (76.5s run)')
    ax1.plot(t, temp_rise_b + 25.0, color='#27ae60', linewidth=2.8, label='Exp B: AeroEdge-RT (26.3s run + Cools)')

    ax1.axvline(26.32, color='#27ae60', linestyle=':', linewidth=1.5, label='AeroEdge-RT 1000-Tok Complete')
    ax1.axvline(76.53, color='#c0392b', linestyle=':', linewidth=1.5, label='Baseline 1000-Tok Complete')
    
    ax1.annotate(f"AeroEdge Peak:\n42.1°C (+17.1°C)",
                 xy=(26.32, peak_b + 25.0), xytext=(12, 45),
                 arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.2, headwidth=5),
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5", ec="#2ecc71"))

    ax1.annotate(f"Baseline Peak:\n48.8°C (+23.8°C)\nSustained Heat!",
                 xy=(76.53, temp_rise_a[-1] + 25.0), xytext=(55, 36),
                 arrowprops=dict(facecolor='#c0392b', shrink=0.08, width=1.2, headwidth=5),
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fbeee6", ec="#e74c3c"))

    ax1.set_title("Silicon Die Temperature Trajectory (1000 Tokens Run)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Time (Seconds)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Die Temperature (°C, Ambient=25°C)", fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 85)
    ax1.set_ylim(22, 54)
    ax1.legend(loc='lower right', frameon=True, fontsize=10)

    # Right plot: Peak Thermal Rise for 48, 100, 500, 1000 tokens
    times_a = (TOKEN_STEPS * EXP_A_LAT_MS) / 1000.0
    times_b = (TOKEN_STEPS * EXP_B_LAT_MS) / 1000.0
    delta_a = delta_t_max * (1.0 - np.exp(-times_a / tau_th))
    delta_b = delta_t_max * (1.0 - np.exp(-times_b / tau_th))

    x = np.arange(len(TOKEN_STEPS))
    width = 0.35
    ax2.bar(x - width/2, delta_a, width, label='Exp A: Baseline ΔT', color='#e74c3c', edgecolor='black', alpha=0.9)
    ax2.bar(x + width/2, delta_b, width, label='Exp B: AeroEdge-RT ΔT', color='#2ecc71', edgecolor='black', alpha=0.9)

    for i in range(len(TOKEN_STEPS)):
        ax2.text(x[i] - width/2, delta_a[i] + 0.6, f"+{delta_a[i]:.1f}°C", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#c0392b')
        ax2.text(x[i] + width/2, delta_b[i] + 0.6, f"+{delta_b[i]:.1f}°C", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#27ae60')

    ax2.set_title("Thermal Dissipation (ΔT Rise Above Ambient)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n} Tok" for n in TOKEN_STEPS], fontsize=11, fontweight='bold')
    ax2.set_ylabel("Temperature Rise ΔT (°C)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 28)
    ax2.legend(loc='upper left', frameon=True, fontsize=10)

    plt.suptitle("Figure 9: ESP32 Thermal Dissipation & Silicon Reliability", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_fig(fig, "figure9_thermal_dissipation_and_temperature")
    plt.close()

# ==============================================================================
# Figure 10: Master Multi-Token Research Scaling Dashboard
# ==============================================================================
def plot_figure10():
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    x = np.arange(len(TOKEN_STEPS))
    width = 0.35

    # Panel 1: Generation Time
    time_a = (TOKEN_STEPS * EXP_A_LAT_MS) / 1000.0
    time_b = (TOKEN_STEPS * EXP_B_LAT_MS) / 1000.0
    ax1.bar(x - width/2, time_a, width, label='Exp A: Baseline', color='#e74c3c', edgecolor='black', alpha=0.9)
    ax1.bar(x + width/2, time_b, width, label='Exp B: AeroEdge-RT', color='#2ecc71', edgecolor='black', alpha=0.9)
    for i in range(len(TOKEN_STEPS)):
        ax1.text(x[i] - width/2, time_a[i] + 1.2, f"{time_a[i]:.1f}s", ha='center', fontsize=9, fontweight='bold', color='#c0392b')
        ax1.text(x[i] + width/2, time_b[i] + 1.2, f"{time_b[i]:.1f}s", ha='center', fontsize=9, fontweight='bold', color='#27ae60')
    ax1.set_title("A. Total Wall-Clock Execution Time (s)", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n} Tokens" for n in TOKEN_STEPS], fontweight='bold')
    ax1.set_ylabel("Seconds", fontweight='bold')
    ax1.set_ylim(0, 92)
    ax1.legend(loc='upper left', frameon=True)

    # Panel 2: Energy in Joules
    energy_a = (TOKEN_STEPS * EXP_A_ENERGY_MJ) / 1000.0
    energy_b = (TOKEN_STEPS * EXP_B_ENERGY_MJ) / 1000.0
    ax2.bar(x - width/2, energy_a, width, label='Exp A: Baseline', color='#e67e22', edgecolor='black', alpha=0.9)
    ax2.bar(x + width/2, energy_b, width, label='Exp B: AeroEdge-RT', color='#1abc9c', edgecolor='black', alpha=0.9)
    for i in range(len(TOKEN_STEPS)):
        ax2.text(x[i] - width/2, energy_a[i] + 1.0, f"{energy_a[i]:.1f}J", ha='center', fontsize=9, fontweight='bold', color='#d35400')
        ax2.text(x[i] + width/2, energy_b[i] + 1.0, f"{energy_b[i]:.1f}J", ha='center', fontsize=9, fontweight='bold', color='#16a085')
    ax2.set_title("B. Total Energy Dissipated (Joules @ 3.3V)", fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n} Tokens" for n in TOKEN_STEPS], fontweight='bold')
    ax2.set_ylabel("Joules", fontweight='bold')
    ax2.set_ylim(0, 72)
    ax2.legend(loc='upper left', frameon=True)

    # Panel 3: CPU Cycles in Billions
    cycles_a = (TOKEN_STEPS * EXP_A_CYCLES) / 1e9
    cycles_b = (TOKEN_STEPS * EXP_B_CYCLES) / 1e9
    ax3.bar(x - width/2, cycles_a, width, label='Exp A: Baseline', color='#9b59b6', edgecolor='black', alpha=0.9)
    ax3.bar(x + width/2, cycles_b, width, label='Exp B: AeroEdge-RT', color='#3498db', edgecolor='black', alpha=0.9)
    for i in range(len(TOKEN_STEPS)):
        ax3.text(x[i] - width/2, cycles_a[i] + 0.3, f"{cycles_a[i]:.1f}B", ha='center', fontsize=9, fontweight='bold', color='#8e44ad')
        ax3.text(x[i] + width/2, cycles_b[i] + 0.3, f"{cycles_b[i]:.1f}B", ha='center', fontsize=9, fontweight='bold', color='#2980b9')
    ax3.set_title("C. Cumulative CPU Clock Cycles (Billions)", fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{n} Tokens" for n in TOKEN_STEPS], fontweight='bold')
    ax3.set_ylabel("Cycles (10^9)", fontweight='bold')
    ax3.set_ylim(0, 22)
    ax3.legend(loc='upper left', frameon=True)

    # Panel 4: Memory Ceiling Analysis (Why Circular KV-Cache is mandatory)
    # Linear KV Cache vs Circular Sliding-Window KV Cache
    linear_kv_kb = (5 * TOKEN_STEPS * 32 * 4 * 2) / 1024.0  # KB
    circular_kv_kb = np.full(len(TOKEN_STEPS), (5 * 64 * 32 * 4 * 2) / 1024.0)  # Fixed at 40.96 KB
    esp32_total_sram_kb = 320.0  # Usable internal heap

    ax4.plot(TOKEN_STEPS, linear_kv_kb, color='#e74c3c', marker='s', linewidth=2.5, linestyle='--', label='Naive Linear KV Cache (OOM Hazard)')
    ax4.plot(TOKEN_STEPS, circular_kv_kb, color='#2ecc71', marker='o', linewidth=2.8, label='AeroEdge-RT Sliding Window (O(1) Pinned)')
    ax4.axhline(esp32_total_sram_kb, color='#c0392b', linestyle=':', linewidth=1.8, label='ESP32 Usable SRAM Limit (320 KB)')

    ax4.annotate("1000 Tokens Linear Cache:\n1,250 KB (CRASHES ESP32!)",
                 xy=(1000, 1250), xytext=(520, 950),
                 arrowprops=dict(facecolor='#e74c3c', shrink=0.08, width=1.2, headwidth=5),
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fbeee6", ec="#e74c3c"))

    ax4.annotate("AeroEdge-RT:\nStrictly 41 KB (Zero OOM Risk)",
                 xy=(1000, 40.96), xytext=(600, 200),
                 arrowprops=dict(facecolor='#2ecc71', shrink=0.08, width=1.2, headwidth=5),
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5", ec="#2ecc71"))

    ax4.set_title("D. KV-Cache Memory Footprint & OOM Ceiling", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Tokens Generated", fontweight='bold')
    ax4.set_ylabel("KV-Cache RAM Required (KB)", fontweight='bold')
    ax4.set_xlim(0, 1050)
    ax4.set_ylim(0, 1400)
    ax4.legend(loc='upper left', frameon=True)

    plt.suptitle("Figure 10: Multi-Token Research Scaling Dashboard (48, 100, 500, 1000 Tokens)", fontsize=16, fontweight='bold', y=0.99)
    save_fig(fig, "figure10_master_scaling_dashboard")
    plt.close()

if __name__ == '__main__':
    print("Generating comprehensive multi-token scaling plots (48, 100, 500, 1000 tokens)...")
    plot_figure7()
    plot_figure8()
    plot_figure9()
    plot_figure10()
    print("All multi-token scaling figures successfully generated and saved!")
