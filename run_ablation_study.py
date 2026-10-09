import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from common import set_seed, load_splits, make_loaders, train_model, calibrate_temperature, evaluate_model
from models import PhysicsAugmentedCalibratedCNN

RESULTS_DIR = config.RESULTS_DIR
METRICS = ["accuracy", "macro_f1", "ece", "nll", "false_alarm_rate"]

# Architectural ablations (each needs its own training run)
CONFIGURATIONS = [
    {"key": "M0", "name": "Full Proposed Architecture", "kw": {}},
    {"key": "M1", "name": "w/o Kinematic Residual Filter", "kw": {"use_residual_filter": False}},
    {"key": "M2", "name": "w/o Dual-Stream Conv", "kw": {"use_dual_stream": False}},
    {"key": "M3", "name": "w/o Sensor-Temporal Attention", "kw": {"use_attention": False}},
]
# M4 reuses each M0 network and simply skips temperature scaling. Temperature
# cannot change the argmax, so M4 only differs from M0 in ECE / NLL.
M4 = {"key": "M4", "name": "w/o Temperature Calibration"}


def run_ablation_study(seeds, epochs):
    print("=" * 80)
    print(f" ABLATION STUDY ({len(seeds)} seeds: {seeds})")
    print("=" * 80)
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    per_seed = {c["key"]: [] for c in CONFIGURATIONS + [M4]}
    for seed in seeds:
        for cfg in CONFIGURATIONS:
            print(f"\n[seed {seed}] {cfg['key']}: {cfg['name']}")
            set_seed(seed)
            train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, seed)
            model = train_model(PhysicsAugmentedCalibratedCNN(**cfg["kw"]), train_loader, val_loader,
                                y_train, epochs=epochs, verbose=False, seed=seed)
            model = calibrate_temperature(model, X_val, y_val, verbose=False)
            res = evaluate_model(model, X_test, y_test)
            per_seed[cfg["key"]].append({k: float(res[k]) for k in METRICS})
            print("   " + " | ".join(f"{k}: {res[k]:.2f}" for k in METRICS))
            if cfg["key"] == "M0":
                res_uncal = evaluate_model(model, X_test, y_test, calibrated=False)
                per_seed["M4"].append({k: float(res_uncal[k]) for k in METRICS})

    summary = {}
    for c in CONFIGURATIONS + [M4]:
        runs = per_seed[c["key"]]
        row = {"name": c["name"]}
        for k in METRICS:
            vals = np.array([r[k] for r in runs])
            ref = np.array([r[k] for r in per_seed["M0"]])
            row[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
            if c["key"] != "M0":
                row[k]["delta_vs_M0"] = float((vals - ref).mean())
                row[k]["p_value"] = float(ttest_rel(vals, ref).pvalue) if len(vals) > 1 and np.any(vals != ref) else 1.0
        summary[c["key"]] = row

    with open(os.path.join(RESULTS_DIR, "ablation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "epochs": epochs, "summary": summary, "per_seed": per_seed}, f, indent=2)

    print_table(summary, len(seeds))
    plot_chart(summary)
    write_report(summary, seeds)


def fmt(m):
    return f"{m['mean']:.2f} ± {m['std']:.2f}"


def print_table(summary, n):
    print("\n" + "=" * 110)
    print(f"{'Configuration':<36} | {'Accuracy':<15} | {'Macro-F1':<15} | {'dF1 (p)':<16} | {'ECE':<14} | {'FAR':<14}")
    print("=" * 110)
    for key, r in summary.items():
        d = r["macro_f1"]
        dstr = "---" if key == "M0" else f"{d['delta_vs_M0']:+.2f} (p={d['p_value']:.2f})"
        print(f"{key + ': ' + r['name']:<36} | {fmt(r['accuracy']):<15} | {fmt(r['macro_f1']):<15} | "
              f"{dstr:<16} | {fmt(r['ece']):<14} | {fmt(r['false_alarm_rate']):<14}")
    print("=" * 110)
    print(f"(mean ± std over {n} seeds; p = paired t-test vs M0 across seeds)")


def plot_chart(summary):
    keys = list(summary)
    labels = ["Full\nModel", "w/o Residual\nFilter", "w/o Dual\nStream", "w/o\nAttention", "w/o Temp\nCalibration"]
    colors = ["#10b981", "#ef4444", "#f97316", "#eab308", "#8b5cf6"]
    panels = [("macro_f1", "Test Macro-F1 on Unseen 3 HP Load (%)\n[Higher is Better]"),
              ("ece", "Expected Calibration Error (%)\n[Lower is Better]"),
              ("false_alarm_rate", "False Alarm Rate on Healthy Windows (%)\n[Lower is Better]")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, title) in zip(axes, panels):
        means = [summary[k][metric]["mean"] for k in keys]
        stds = [summary[k][metric]["std"] for k in keys]
        bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, width=0.55)
        ax.set_title(title, fontweight="bold", fontsize=11)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{m:.1f}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ablation_study_chart.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"\n[SAVED] Ablation Study Chart -> {path}")


def describe(row, metric, higher_is_better, alpha=0.05):
    """Plain-language effect of REMOVING a component, driven by the measured numbers."""
    d, p = row[metric]["delta_vs_M0"], row[metric]["p_value"]
    if p >= alpha:
        return f"no statistically significant change in {metric} ({d:+.2f}, p={p:.2f})"
    helped = (d < 0) if higher_is_better else (d > 0)
    verdict = "the component helps" if helped else "the model is better WITHOUT this component"
    return f"{metric} changes by {d:+.2f} (p={p:.3f}) — {verdict}"


def write_report(summary, seeds):
    test = config.IMPAIRMENTS["test"]
    path = os.path.join(RESULTS_DIR, "ablation_study_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Ablation Study: Component Impact\n\n")
        f.write(f"Trained on loads {config.TRAIN_LOADS} HP, calibrated/selected on load {config.VAL_LOADS} HP, "
                f"tested on unseen load {config.TEST_LOADS} HP with {test['snr_db']:.0f} dB plant noise. "
                f"Each configuration was trained with {len(seeds)} seeds ({seeds}); values are mean ± std and "
                f"p-values are paired t-tests against M0 across seeds.\n\n")
        f.write("| Configuration | Accuracy (%) | Macro-F1 (%) | Δ Macro-F1 (p) | ECE (%) | NLL | FAR (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for key, r in summary.items():
            d = "---" if key == "M0" else f"{r['macro_f1']['delta_vs_M0']:+.2f} (p={r['macro_f1']['p_value']:.2f})"
            f.write(f"| {key}: {r['name']} | {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} | {d} | "
                    f"{fmt(r['ece'])} | {fmt(r['nll'])} | {fmt(r['false_alarm_rate'])} |\n")
        f.write("\n## Findings (generated from the numbers above)\n\n")
        for key in ["M1", "M2", "M3"]:
            f.write(f"- **Removing {summary[key]['name'][4:]}**: {describe(summary[key], 'macro_f1', True)}; "
                    f"ECE: {describe(summary[key], 'ece', False)}.\n")
        f.write(f"- **Removing temperature calibration** (same networks, T=1): accuracy is unchanged by "
                f"construction; ECE: {describe(summary['M4'], 'ece', False)}; "
                f"NLL: {describe(summary['M4'], 'nll', False)}.\n")
    print(f"[SAVED] Ablation Study Report -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=config.DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = parser.parse_args()
    run_ablation_study(args.seeds, args.epochs)
