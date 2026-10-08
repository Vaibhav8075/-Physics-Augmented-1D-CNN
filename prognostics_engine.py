import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from models import PhysicsAugmentedCalibratedCNN

RESULTS_DIR = "results"
PROCESSED_DIR = "processed_data"

def compute_health_index(model, sample_tensor):
    """
    Computes a continuous Machine Health Index (HI in [0, 100%])
    based on calibrated softmax healthy probability and residual kinematic energy.
    """
    with torch.no_grad():
        logits, attn_info, res = model(sample_tensor)
        probs = F.softmax(logits, dim=1).numpy()[0]
        # Class 0 is Healthy
        p_healthy = probs[0]
        p_fault = 1.0 - p_healthy
        
        # Harmonic Energy Ratio (HER): energy in high-frequency residual vs total
        res_energy = torch.mean(res ** 2).item() if res is not None else 0.0
        tot_energy = torch.mean(sample_tensor ** 2).item() + 1e-8
        her = min(1.0, res_energy / tot_energy)
        
        # Health Index formulation
        hi = max(0.0, min(100.0, (p_healthy * 0.7 + (1.0 - her) * 0.3) * 100.0))
        
    return hi, probs, her

def simulate_rul_degradation():
    print("=" * 80)
    print(" INDUSTRIAL PROGNOSTICS & REMAINING USEFUL LIFE (RUL) SIMULATION")
    print("=" * 80)
    print(" NOTE: illustrative only. CWRU has no run-to-failure data, so the degradation")
    print(" trajectory below is synthetic (sigmoid blend of healthy and faulty windows,")
    print(" onset fixed at t=320 h). It demonstrates the HI pipeline, not RUL accuracy.")
    rng = np.random.default_rng(0)

    # Load model
    model = PhysicsAugmentedCalibratedCNN()
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "physics_model.pt"), map_location="cpu"))
    model.eval()

    # Load dataset samples
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    healthy_indices = np.where(y_test == 0)[0]
    fault_indices = np.where(y_test == 1)[0] # Inner race fault

    # Create a 500-hour lifecycle degradation trajectory
    time_hours = np.linspace(0, 500, 100)
    health_indices = []
    fault_prob_curve = []
    her_curve = []

    for t in time_hours:
        # Transition from healthy to progressive wear
        alpha = 1.0 / (1.0 + np.exp(-(t - 320) / 40.0)) # Sigmoidal degradation onset around t=320h
        
        h_idx = rng.choice(healthy_indices)
        f_idx = rng.choice(fault_indices)
        
        # Linear synthetic blend of vibration dynamics representing progressive spalling
        blended_signal = (1.0 - alpha) * X_test[h_idx] + alpha * X_test[f_idx]
        blended_tensor = torch.from_numpy(blended_signal).unsqueeze(0).float()
        
        hi, probs, her = compute_health_index(model, blended_tensor)
        health_indices.append(hi)
        fault_prob_curve.append((1.0 - probs[0]) * 100.0)
        her_curve.append(her * 100.0)

    health_indices = np.array(health_indices)
    fault_prob_curve = np.array(fault_prob_curve)

    # Smooth the curve for clean visualization
    hi_smoothed = np.convolve(health_indices, np.ones(5)/5, mode='same')
    hi_smoothed[:2] = health_indices[:2]
    hi_smoothed[-2:] = health_indices[-2:]

    # Estimate RUL at t=350h (Warning Threshold = HI < 70%, Critical Failure = HI < 20%)
    below_warning = np.where(hi_smoothed < 70.0)[0]
    below_critical = np.where(hi_smoothed < 20.0)[0]
    if len(below_warning) == 0 or len(below_critical) == 0:
        print(f"\n[PROGNOSTICS] HI never crossed the thresholds (min HI = {hi_smoothed.min():.1f}%). "
              "No RUL estimate; check the model or thresholds.")
        return
    warning_t = time_hours[below_warning[0]]
    critical_t = time_hours[below_critical[0]]

    print(f"\n[PROGNOSTICS ANALYSIS]")
    print(f"  • Normal Baseline Lifespan: 0 to {warning_t:.1f} Operating Hours")
    print(f"  • Incipient Fault Detected: at t = {warning_t:.1f} Hours (HI drops below 70%)")
    print(f"  • Critical Failure EOL: at t = {critical_t:.1f} Hours (HI drops below 20%)")
    print(f"  • Remaining Useful Life (RUL) at Warning: {critical_t - warning_t:.1f} Hours")

    # -------------------------------------------------------------
    # GENERATE HIGH-RES PROGNOSTICS PLOT
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # 1. Machine Health Index & RUL Zones
    axes[0].plot(time_hours, hi_smoothed, color="#0284c7", linewidth=2.5, label="Machine Health Index (HI %)")
    axes[0].axhline(y=70, color="#f59e0b", linestyle="--", linewidth=1.5, label="Maintenance Advisory Threshold (70%)")
    axes[0].axhline(y=20, color="#ef4444", linestyle="--", linewidth=1.5, label="Critical Emergency Stop Threshold (20%)")

    # Fill zones
    axes[0].fill_between(time_hours, 70, 100, color="#10b981", alpha=0.12, label="Normal Operation Zone")
    axes[0].fill_between(time_hours, 20, 70, color="#f59e0b", alpha=0.15, label="Degradation Warning Zone (Schedule Maintenance)")
    axes[0].fill_between(time_hours, 0, 20, color="#ef4444", alpha=0.18, label="Imminent Failure Zone (Emergency Shutdown)")

    axes[0].axvline(x=warning_t, color="#f59e0b", linestyle=":", alpha=0.8)
    axes[0].axvline(x=critical_t, color="#ef4444", linestyle=":", alpha=0.8)
    axes[0].text(warning_t + 5, 75, f"Incipient Fault Trigger\n(t = {warning_t:.1f} h)", fontsize=9, fontweight='bold', color="#b45309")
    axes[0].text(critical_t - 65, 25, f"Critical EOL Failure\n(t = {critical_t:.1f} h)", fontsize=9, fontweight='bold', color="#b91c1c")

    axes[0].set_title("Machine Health Index (HI) & Prognostics Degradation Trajectory", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Health Index HI(t) (%)", fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Calibrated Fault Probability vs Harmonic Energy Ratio (HER)
    axes[1].plot(time_hours, fault_prob_curve, color="#e11d48", linewidth=2.0, label="Calibrated Defect Probability P(Fault|x) (%)")
    axes[1].plot(time_hours, her_curve, color="#8b5cf6", linewidth=1.5, linestyle="-.", label="Kinematic Harmonic Energy Ratio (HER %)")
    axes[1].set_title("Calibrated Defect Likelihood & Kinematic Impact Harmonics Over Lifecycle", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Continuous Operating Time (Hours)", fontsize=10, fontweight='bold')
    axes[1].set_ylabel("Probability / Energy (%)", fontsize=10, fontweight='bold')
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "prognostics_rul_trajectory.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[SAVED] Prognostics RUL Trajectory Plot -> {plot_path}")

    # Generate Markdown Report
    report_path = os.path.join(RESULTS_DIR, "prognostics_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Industrial Prognostics & Health Index Trajectory Report\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("> **Simulation only.** The degradation trajectory is synthetic (sigmoid blend of healthy and "
                "inner-race test windows with onset fixed at t = 320 h); CWRU contains no run-to-failure data. "
                "The times below show the Health Index pipeline responding to that injected trajectory and are "
                "not a validated RUL prediction.\n\n")
        f.write(f"The Physics-Augmented 1D-CNN was extended with a **Continuous Health Degradation Index ($HI(t) \\in [0, 100\\%]$)** and Remaining Useful Life (RUL) estimation engine. The system successfully detected incipient bearing surface spalling at **t = {warning_t:.1f} hours**, providing a **{critical_t - warning_t:.1f}-hour maintenance advisory window** before catastrophic mechanical failure at **t = {critical_t:.1f} hours**.\n\n")
        f.write("## 2. Mathematical Health Formulation\n\n")
        f.write("$$HI(t) = \\left[ 0.70 \\cdot P(\\text{Normal}|\\mathbf{x}(t)) + 0.30 \\cdot \\left(1 - \\frac{E_{\\text{residual}}}{E_{\\text{total}}}\\right) \\right] \\times 100\\%$$\n\n")
        f.write("## 3. Operational Lifespan Milestones\n\n")
        f.write(f"- **Normal Steady State**: 0 to {warning_t:.1f} operating hours ($HI > 70\\%$).\n")
        f.write(f"- **Incipient Fault Advisory**: Triggered at t = {warning_t:.1f} hours ($20\\% < HI \\le 70\\%$).\n")
        f.write(f"- **Critical Failure EOL**: t = {critical_t:.1f} hours ($HI \\le 20\\%$).\n")
        f.write(f"- **Actionable RUL Buffer**: **{critical_t - warning_t:.1f} hours** available for scheduling replacement parts without unplanned line stoppage.\n")

    print(f"[SAVED] Prognostics Report -> {report_path}")

if __name__ == "__main__":
    simulate_rul_degradation()
