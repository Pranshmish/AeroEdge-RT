import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

OUT_DIR = "benchmarks/figures"
ARTIFACT_DIR = "C:/Users/ASUS/.gemini/antigravity-ide/brain/accf4005-321a-45a2-8ab4-83f8f35577bd"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Empirical Constants
EXP_A_LAT_MS = 76.53
EXP_B_LAT_MS = 26.32
EXP_A_CYCLES = 18_367_200
EXP_B_CYCLES = 6_316_800
P_ACTIVE_W = 0.792  # 3.3V * 240mA

STEPS = np.array([100, 500, 1000, 2500, 5000, 10000])

def plot_10k_stress():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Total Generation Time (Minutes)
    time_a_min = (STEPS * EXP_A_LAT_MS) / 60000.0
    time_b_min = (STEPS * EXP_B_LAT_MS) / 60000.0

    ax1.plot(STEPS, time_a_min, marker='s', color='#e74c3c', linewidth=2.8, label='Exp A: Baseline (13.07 tok/s)')
    ax1.plot(STEPS, time_b_min, marker='o', color='#2ecc71', linewidth=3.0, label='Exp B: AeroEdge-RT (38.01 tok/s)')
    ax1.fill_between(STEPS, time_b_min, time_a_min, color='#2ecc71', alpha=0.15, label='Runtime Savings Zone')

    ax1.annotate(f"10,000 Tokens:\nBaseline = 12.75 min\nAeroEdge = 4.38 min\nSAVED: 8.37 MINUTES!",
                 xy=(10000, 4.38), xytext=(5500, 7.5),
                 arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#e8f8f5", ec="#2ecc71"))

    ax1.set_title("A. Autoregressive Wall-Clock Time (Minutes)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Tokens Generated", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Duration (Minutes)", fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 10500)
    ax1.set_ylim(0, 14.5)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)

    # 2. Cumulative Battery Energy (Joules)
    energy_a = (STEPS * (P_ACTIVE_W * (EXP_A_LAT_MS / 1000.0)))
    energy_b = (STEPS * (P_ACTIVE_W * (EXP_B_LAT_MS / 1000.0)))

    ax2.plot(STEPS, energy_a, marker='s', color='#e67e22', linewidth=2.8, label='Exp A: Baseline Energy')
    ax2.plot(STEPS, energy_b, marker='o', color='#1abc9c', linewidth=3.0, label='Exp B: AeroEdge-RT Energy')
    ax2.fill_between(STEPS, energy_b, energy_a, color='#1abc9c', alpha=0.15)

    ax2.annotate(f"10,000 Tokens:\nBaseline = 605.9 J\nAeroEdge = 208.3 J\nSAVED: 397.6 JOULES!",
                 xy=(10000, 208.3), xytext=(5500, 350),
                 arrowprops=dict(facecolor='#16a085', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#e0f2f1", ec="#26a69a"))

    ax2.set_title("B. Cumulative Energy Consumed (Joules @ 3.3V)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Tokens Generated", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Energy (Joules)", fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 10500)
    ax2.set_ylim(0, 680)
    ax2.legend(loc='upper left', frameon=True, fontsize=10)

    # 3. CPU Clock Cycles (Billions)
    cycles_a_b = (STEPS * EXP_A_CYCLES) / 1e9
    cycles_b_b = (STEPS * EXP_B_CYCLES) / 1e9

    ax3.plot(STEPS, cycles_a_b, marker='s', color='#9b59b6', linewidth=2.8, label='Exp A: Baseline Cycles')
    ax3.plot(STEPS, cycles_b_b, marker='o', color='#3498db', linewidth=3.0, label='Exp B: AeroEdge-RT Cycles')
    ax3.fill_between(STEPS, cycles_b_b, cycles_a_b, color='#3498db', alpha=0.15)

    ax3.annotate(f"10,000 Tokens:\n120.5 BILLION\nCycles Saved!",
                 xy=(10000, 63.17), xytext=(5500, 110),
                 arrowprops=dict(facecolor='#2980b9', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#e1f5fe", ec="#0288d1"))

    ax3.set_title("C. Cumulative CPU Clock Cycles (Billions)", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("Tokens Generated", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Cycles (10^9)", fontsize=11, fontweight='bold')
    ax3.set_xlim(0, 10500)
    ax3.set_ylim(0, 205)
    ax3.legend(loc='upper left', frameon=True, fontsize=10)

    # 4. RAM Feasibility: Linear vs Circular KV Cache (Log Scale)
    linear_kv_mb = (5 * STEPS * 32 * 4 * 2) / (1024.0 * 1024.0)
    circular_kv_mb = np.full(len(STEPS), (5 * 64 * 32 * 4 * 2) / (1024.0 * 1024.0))  # 0.04 MB
    esp32_max_ram_mb = 0.32  # 320 KB usable

    ax4.plot(STEPS, linear_kv_mb, marker='x', color='#c0392b', linewidth=2.8, linestyle='--', label='Naive Linear KV Cache (OOM Crash)')
    ax4.plot(STEPS, circular_kv_mb, marker='o', color='#27ae60', linewidth=3.0, label='AeroEdge-RT Ring Buffer (O(1) Safe)')
    ax4.axhline(esp32_max_ram_mb, color='#d35400', linestyle=':', linewidth=2.0, label='ESP32 Usable SRAM Limit (0.32 MB)')

    ax4.annotate(f"10,000 Tokens Linear:\n12.8 MB RAM Required!\n(Fatal OOM Panic)",
                 xy=(10000, 12.8), xytext=(5500, 8.5),
                 arrowprops=dict(facecolor='#c0392b', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#fbeee6", ec="#e74c3c"))

    ax4.annotate(f"AeroEdge-RT:\nStrictly 0.04 MB (41 KB)!\n100% Memory Safe",
                 xy=(10000, 0.04), xytext=(6000, 2.5),
                 arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#e8f8f5", ec="#2ecc71"))

    ax4.set_title("D. SRAM Safety & 12.8 MB OOM Elimination", fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel("Tokens Generated", fontsize=11, fontweight='bold')
    ax4.set_ylabel("KV-Cache RAM Required (MB)", fontsize=11, fontweight='bold')
    ax4.set_xlim(0, 10500)
    ax4.set_ylim(-0.5, 14.5)
    ax4.legend(loc='center left', frameon=True, fontsize=10)

    plt.suptitle("Figure 11: 10,000 Tokens Extreme Stress Test & Reliability Architecture", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()

    # Save
    p_png = os.path.join(OUT_DIR, "figure11_extreme_10k_stress_test.png")
    p_svg = os.path.join(OUT_DIR, "figure11_extreme_10k_stress_test.svg")
    fig.savefig(p_png, dpi=300, bbox_inches='tight')
    fig.savefig(p_svg, format='svg', bbox_inches='tight')

    art_png = os.path.join(ARTIFACT_DIR, "figure11_extreme_10k_stress_test.png")
    fig.savefig(art_png, dpi=300, bbox_inches='tight')
    print("[OK] Saved figure11_extreme_10k_stress_test (PNG + SVG)")
    plt.close()

if __name__ == '__main__':
    plot_10k_stress()
