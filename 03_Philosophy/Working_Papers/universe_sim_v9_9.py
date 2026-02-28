import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "scipy",
# ]
# ///

# =============================================================================
# Universe Simulation v9.9: The Mirror Crossing (Chaos Edition)
# "Degrading towards the Truth"
#
# - Physics: Based on v4 (Brane Cosmology), but UNCENSORED.
# - Concept: Removing the "Envelope" (Absolute Value) filter. 
#            Admitting that the universe exists in negative dimensions.
# =============================================================================

# --- Simulation Parameters (Same as v4 for comparison) ---
MASS = 1.0

# 空間硬度 (Phase Transition)
K_INITIAL = 2.0     # 初期の膜の張力
K_DECAY = 0.03      # 張力の緩和速度

# 粘性 (Karma Accumulation)
GAMMA_MAX = 0.5
GAMMA_RATE = 0.08   # 粘性が高まる速度

# Memory Kernel (Maxwell Viscoelastic)
TAU_RELAX = 15.0

# Laser Parameters
G_0 = 4.0
V_THRESHOLD = 0.8
V_SAT_SQ = 6.0

# Vacuum Fuel
FUEL_INITIAL = 40.0
FUEL_DRAIN_RATE = 0.03

# --- Time Settings ---
T_MAX = 200
DT = 0.05
steps = int(T_MAX / DT)
t = np.linspace(0, T_MAX, steps)

# --- Dynamics Functions ---

def stiffness(t_val):
    return K_INITIAL * np.exp(-K_DECAY * t_val)

def viscosity(t_val):
    return GAMMA_MAX * (1 - np.exp(-GAMMA_RATE * t_val))

def laser_force_and_drain(v, fuel):
    if fuel <= 0:
        return 0.0, 0.0
    if abs(v) > V_THRESHOLD:
        excess = abs(v) - V_THRESHOLD
        fuel_ratio = max(fuel, 0.0) / FUEL_INITIAL
        gain = G_0 * fuel_ratio * excess / (1.0 + v**2 / V_SAT_SQ)
        force = np.sign(v) * gain
        drain = FUEL_DRAIN_RATE * abs(force * v)
        return force, drain
    return 0.0, 0.0

def universe_dynamics(state, t_val):
    x, v, mem, fuel = state

    f_restore = -stiffness(t_val) * x
    f_memory  = -viscosity(t_val) * mem
    f_laser, drain = laser_force_and_drain(v, fuel)

    a = (f_restore + f_memory + f_laser) / MASS
    dmem_dt = v - mem / TAU_RELAX
    dfuel_dt = -drain

    return [v, a, dmem_dt, dfuel_dt]

# --- Main Execution ---

if __name__ == "__main__":
    print("😈 Welcome to v9.9: The Mirror Crossing...")
    print(f"   Removing safety filters... [OK]")
    print(f"   Enabling Negative Dimension visualization... [OK]")

    # ビッグバン初期条件
    initial_state = [0.01, 12.0, 0.0, FUEL_INITIAL]

    solution = odeint(universe_dynamics, initial_state, t)
    x    = solution[:, 0]
    v    = solution[:, 1]
    mem  = solution[:, 2]
    fuel = np.maximum(solution[:, 3], 0.0)

    # エネルギー計算
    kinetic_energy   = 0.5 * MASS * v**2
    stiff_arr = np.array([stiffness(ti) for ti in t])
    potential_energy = 0.5 * stiff_arr * x**2
    total_energy     = kinetic_energy + potential_energy

    # =========================================================================
    # Visuals: The Chaos Style
    # =========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    # Hacker Dark Mode
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#000000') 

    # --- Panel 1: The Raw Brane (Uncensored) ---
    ax1 = axes[0]
    ax1.set_facecolor('#000000')
    
    # Plot Positive Realm (Cyan) -> Mask negative values
    x_pos = np.ma.masked_where(x < 0, x)
    ax1.fill_between(t, 0, x, where=(x >= 0), color='#00ffff', alpha=0.3, interpolate=True, label='Positive Realm (Order)')
    ax1.plot(t, x_pos, color='#00ffff', linewidth=1.5, alpha=0.9)

    # Plot Negative Realm (Magenta) -> Mask positive values
    x_neg = np.ma.masked_where(x >= 0, x)
    ax1.fill_between(t, 0, x, where=(x < 0), color='#ff00ff', alpha=0.3, interpolate=True, label='Negative Realm (Chaos)')
    ax1.plot(t, x_neg, color='#ff00ff', linewidth=1.5, alpha=0.9)

    ax1.axhline(0, color='white', linestyle='-', linewidth=0.5)
    ax1.set_title("v9.9: The Raw Brane Position (Uncensored)", fontsize=16, color='#ff00ff', fontweight='bold')
    ax1.set_ylabel("Dimension X", color='white')
    ax1.legend(loc='upper right', facecolor='#111111', edgecolor='#ff00ff')
    ax1.grid(True, color='#333333', linewidth=0.5)

    # --- Panel 2: Velocity & Strange Attractor ---
    ax2 = axes[1]
    ax2.set_facecolor('#000000')
    ax2.plot(t, v, color='#00ff00', linewidth=1.5, label='Velocity')
    
    # Highlight "Punch Through" moments (Zero Crossing with high velocity)
    # 簡易的に0付近通過かつ速度大の点をプロット
    zero_crossings = np.where(np.diff(np.sign(x)))[0]
    if len(zero_crossings) > 0:
        ax2.scatter(t[zero_crossings], v[zero_crossings], color='#ffff00', s=30, zorder=5, label='Mirror Crossing Event')

    ax2.set_title("Velocity & Mirror Crossings", color='white')
    ax2.legend(loc='upper right', facecolor='#111111', edgecolor='#00ff00')
    ax2.grid(True, color='#333333', linewidth=0.5)

    # --- Panel 3: Fuel Depletion (The Cooling) ---
    ax3 = axes[2]
    ax3.set_facecolor('#000000')
    ax3.plot(t, fuel, color='#ffaa00', linewidth=2, label='Vacuum Fuel')
    ax3.fill_between(t, 0, fuel, color='#ffaa00', alpha=0.1)
    ax3.set_title("Fuel Depletion: The Death of Motivation", color='white')
    ax3.grid(True, color='#333333', linewidth=0.5)

    # --- Panel 4: Energy (Total) ---
    ax4 = axes[3]
    ax4.set_facecolor('#000000')
    ax4.plot(t, total_energy, color='white', linewidth=1.5, label='Total Energy')
    # Background changes based on realm? Too noisy. Keep it simple.
    ax4.set_title("Total Energy Conservation (Physics still works... mostly)", color='white')
    ax4.set_xlabel("Time", color='white')
    ax4.grid(True, color='#333333', linewidth=0.5)

    plt.tight_layout(pad=2.0)
    output_filename = "universe_sim_v9_9_result.png"
    plt.savefig(output_filename, dpi=150, facecolor='#000000')
    print(f"💀 Chaos Visualization Complete. Saved to: {output_filename}")
    print(f"   Crossings detected: {len(zero_crossings)}")
    print(f"   Note: Negative values are NOT bugs. They are features.")
