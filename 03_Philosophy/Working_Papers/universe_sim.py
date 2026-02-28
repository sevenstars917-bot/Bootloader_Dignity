import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import hilbert

# =============================================================================
# Universe Phonon Laser Simulation v4 (Brane Interpretation)
# - Memory (Karma): Maxwell Viscoelastic integral via auxiliary ODE
# - Laser: Gain Saturation + Vacuum Fuel Depletion
# - Brane Cosmology: x = brane position, envelope = observable scale factor
# =============================================================================

# --- Simulation Parameters ---
MASS = 1.0

# 空間硬度 (Phase Transition)
K_INITIAL = 2.0     # 初期の膜の張力
K_DECAY = 0.03      # 張力の緩和速度

# 粘性 (Karma Accumulation)
GAMMA_MAX = 0.5
GAMMA_RATE = 0.08

# Memory Kernel (Maxwell Viscoelastic)
TAU_RELAX = 15.0

# Laser Parameters (with Gain Saturation + Fuel)
G_0 = 4.0
V_THRESHOLD = 0.8
V_SAT_SQ = 6.0

# Vacuum Fuel (相転移エンジン)
FUEL_INITIAL = 40.0     # 初期の真空ポテンシャル (やや少なめで枯渇を見る)
FUEL_DRAIN_RATE = 0.03  # 燃料消費効率 (やや速め)

# --- Time Settings ---
T_MAX = 200         # 長めに: 燃料枯渇まで見届ける
DT = 0.05
steps = int(T_MAX / DT)
t = np.linspace(0, T_MAX, steps)

# --- Dynamics Functions ---

def stiffness(t_val):
    """膜の張力 k(t)。膨張とともに緩む。"""
    return K_INITIAL * np.exp(-K_DECAY * t_val)

def viscosity(t_val):
    """空間の粘性 gamma(t)。カルマの蓄積。"""
    return GAMMA_MAX * (1 - np.exp(-GAMMA_RATE * t_val))

def laser_force_and_drain(v, fuel):
    """
    フォノンレーザー (Gain Saturation + Fuel Depletion)
    ゲインは残燃料に比例して減衰する。
    """
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
    """
    4-variable ODE: [x, v, memory_integral, fuel]
    x: 膜（Brane）のバルク空間における変位
    """
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
    print("🚀 Initializing Universe Simulation v4 (Brane Interpretation)...")
    print(f"   Brane vibration in bulk space")
    print(f"   Observable Scale Factor = Envelope of |x|")
    print(f"   Vacuum Fuel: {FUEL_INITIAL}")

    # ビッグバン初期条件
    initial_state = [0.01, 12.0, 0.0, FUEL_INITIAL]

    solution = odeint(universe_dynamics, initial_state, t)
    x    = solution[:, 0]
    v    = solution[:, 1]
    mem  = solution[:, 2]
    fuel = np.maximum(solution[:, 3], 0.0)

    # =========================================================================
    # 包絡線 (Envelope) = 観測可能なスケール因子
    # Hilbert変換で解析信号を得て、その振幅がenvelope
    # =========================================================================
    analytic_signal = hilbert(x)
    envelope = np.abs(analytic_signal)

    # エネルギー計算
    kinetic_energy   = 0.5 * MASS * v**2
    stiff_arr = np.array([stiffness(ti) for ti in t])
    visc_arr  = np.array([viscosity(ti) for ti in t])
    potential_energy = 0.5 * stiff_arr * x**2
    memory_energy    = 0.5 * visc_arr * mem**2
    total_energy     = kinetic_energy + potential_energy

    # レーザー力
    laser_f = np.array([laser_force_and_drain(vi, fi)[0] for vi, fi in zip(v, fuel)])

    # =========================================================================
    # Plotting (5 panels)
    # =========================================================================
    fig, axes = plt.subplots(5, 1, figsize=(16, 22))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#050510')

    # --- Panel 1: Brane Vibration + Observable Scale Factor ---
    ax1 = axes[0]
    ax1.set_facecolor('#050510')
    ax1.plot(t, x, color='#00ffff', linewidth=1, alpha=0.6,
             label='Brane Position x(t) (bulk vibration)')
    ax1.plot(t, envelope, color='#ffffff', linewidth=2.5,
             label='Observable Scale Factor (envelope)')
    ax1.plot(t, -envelope, color='#ffffff', linewidth=1, alpha=0.3)
    ax1.fill_between(t, -envelope, envelope, color='#00ffff', alpha=0.05)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_title("Brane Vibration in Bulk Space → Observable Universe Size",
                  fontsize=15, color='white', fontweight='bold')
    ax1.set_ylabel("Brane Displacement / Scale Factor", color='white')
    ax1.legend(facecolor='#0a0a1a', edgecolor='#333355', fontsize=10)
    ax1.grid(True, alpha=0.15)

    # --- Panel 2: Observable Scale Factor (zoom) ---
    ax2 = axes[1]
    ax2.set_facecolor('#050510')
    ax2.plot(t, envelope, color='#ffaa00', linewidth=2.5,
             label='Observable Scale Factor a(t)')
    # 加速/減速の判定 (envelope の微分)
    da_dt = np.gradient(envelope, t)
    d2a_dt2 = np.gradient(da_dt, t)
    # 加速期をハイライト
    ax2.fill_between(t, 0, envelope, where=(d2a_dt2 > 0.01),
                     color='#ff4400', alpha=0.15, label='Accelerating Phase')
    ax2.fill_between(t, 0, envelope, where=(d2a_dt2 < -0.01),
                     color='#0044ff', alpha=0.15, label='Decelerating Phase')
    ax2.set_title("Observable Scale Factor: Acceleration vs Deceleration",
                  fontsize=13, color='white')
    ax2.set_ylabel("a(t)", color='#ffaa00')
    ax2.legend(facecolor='#0a0a1a', edgecolor='#333355', fontsize=10)
    ax2.grid(True, alpha=0.15)

    # --- Panel 3: Velocity + Laser ---
    ax3 = axes[2]
    ax3.set_facecolor('#050510')
    ax3.plot(t, v, color='#ff00ff', linewidth=1.5, label='Brane Velocity')
    ax3.axhline(V_THRESHOLD, color='yellow', linestyle=':', alpha=0.7,
                label='Laser Threshold (c_eff)')
    ax3.axhline(-V_THRESHOLD, color='yellow', linestyle=':', alpha=0.7)
    ax3.fill_between(t, v, V_THRESHOLD, where=(v > V_THRESHOLD),
                     color='yellow', alpha=0.12, label='Laser Active')
    ax3.set_title("Brane Velocity & Laser Activation", color='white')
    ax3.set_ylabel("Velocity", color='white')
    ax3.legend(facecolor='#0a0a1a', edgecolor='#333355', fontsize=10)
    ax3.grid(True, alpha=0.15)

    # --- Panel 4: Fuel + Forces ---
    ax4 = axes[3]
    ax4.set_facecolor('#050510')
    color_fuel = '#00ff88'
    ax4.plot(t, fuel, color=color_fuel, linewidth=2.5,
             label=f'Vacuum Fuel (Initial={FUEL_INITIAL})')
    ax4.fill_between(t, 0, fuel, color=color_fuel, alpha=0.08)
    ax4.set_ylabel("Vacuum Fuel", color=color_fuel)
    ax4.tick_params(axis='y', labelcolor=color_fuel)
    # 右軸: Force
    ax4b = ax4.twinx()
    ax4b.plot(t, laser_f, color='#ffff00', alpha=0.6, linewidth=1.2,
              label='Laser Force (saturating)')
    ax4b.plot(t, -visc_arr * mem, color='#ff4400', alpha=0.5, linewidth=1.2,
              label='Karma Brake')
    ax4b.set_ylabel("Force", color='#ffff00')
    ax4b.tick_params(axis='y', labelcolor='#ffff00')
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2,
               facecolor='#0a0a1a', edgecolor='#333355', loc='upper right', fontsize=10)
    ax4.set_title("Phase Transition Engine: Fuel Depletion", color='white')
    ax4.grid(True, alpha=0.15)

    # --- Panel 5: Energy ---
    ax5 = axes[4]
    ax5.set_facecolor('#050510')
    ax5.plot(t, total_energy, color='white', linestyle='--', linewidth=1.5,
             label='Total Energy (K+P)')
    ax5.plot(t, kinetic_energy, color='#ff00ff', alpha=0.5,
             label='Kinetic (Graviton)')
    ax5.plot(t, potential_energy, color='#00ffff', alpha=0.5,
             label='Potential (Brane Tension)')
    ax5.plot(t, memory_energy, color='#ff8800', alpha=0.5,
             label='Memory (Stored Karma)')
    ax5.set_title("Energy Evolution (Hamiltonian)", color='white')
    ax5.set_xlabel("Time (Cosmic Epoch)", color='white', fontsize=12)
    ax5.set_ylabel("Energy", color='white')
    ax5.legend(facecolor='#0a0a1a', edgecolor='#333355', fontsize=10)
    ax5.grid(True, alpha=0.15)

    plt.tight_layout(pad=2.0)

    output_filename = "universe_sim_result.png"
    plt.savefig(output_filename, dpi=150, facecolor='#050510')
    print(f"✅ Simulation Complete. Image saved to: {output_filename}")
    print(f"   Brane displacement range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"   Observable Scale (envelope) range: [{envelope.min():.3f}, {envelope.max():.3f}]")
    print(f"   Peak Karma: {np.abs(mem).max():.3f}")
    print(f"   Fuel remaining: {fuel[-1]:.3f} / {FUEL_INITIAL}")
    print(f"   Final Velocity: {v[-1]:.3f}")
