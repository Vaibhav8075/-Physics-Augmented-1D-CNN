"""
Leave-one-load-out cross-load benchmark with a noise sweep.

For each held-out motor load (0-3 HP):
  * train on the other three loads (all fault sizes: 0.007", 0.014", 0.021"),
  * fit temperature on a time-held-out tail of the training recordings,
  * test on the held-out load, clean and at every SNR in config.BENCHMARK_SNR_DB.
Every configuration is trained with several seeds; models are compared with a
paired Wilcoxon signed-rank test over all (fold, seed) pairs.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

import config
from common import (set_seed, make_loaders, train_model, calibrate_temperature, predict_logits,
                    classification_metrics, add_real_world_impairments)
from models import Baseline1DCNN, PhysicsAugmentedCalibratedCNN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = config.RESULTS_DIR
OUT_JSON = os.path.join(RESULTS_DIR, "benchmark_metrics.json")
REFERENCE = "physics"
MODEL_SPECS = {
    "baseline": ("Standard 1D-CNN", lambda: Baseline1DCNN()),
    "physics": ("PAC-1DCNN (full)", lambda: PhysicsAugmentedCalibratedCNN()),
    "no_residual": ("PAC w/o Residual Filter", lambda: PhysicsAugmentedCalibratedCNN(use_residual_filter=False)),
    "no_dual": ("PAC w/o Dual-Stream", lambda: PhysicsAugmentedCalibratedCNN(use_dual_stream=False)),
    "no_attention": ("PAC w/o Attention", lambda: PhysicsAugmentedCalibratedCNN(use_attention=False)),
}
SUMMARY_METRICS = ["accuracy", "macro_f1", "false_alarm_rate", "missed_fault_rate", "ece", "nll"]


def snr_label(snr):
    return "clean" if snr is None else f"{snr:g}dB"


def load_per_load():
    data = {}
    for load in config.ALL_LOADS:
        path = os.path.join(config.PROCESSED_DIR, f"load{load}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing - run preprocess_data.py first.")
        data[load] = dict(np.load(path))
    return data


def impair_fixed(X, snr_db, drift_prob, impulse_prob, seed):
    rng = np.random.default_rng(seed)
    return np.stack([add_real_world_impairments(x, snr_db=snr_db, drift_prob=drift_prob,
                                                impulse_prob=impulse_prob, rng=rng) for x in X]).astype(np.float32)


def build_fold(data, test_load):
    """Train/calibration split by time within each training recording, plus the noise-swept test sets."""
    train_loads = [l for l in config.ALL_LOADS if l != test_load]
    X = np.concatenate([data[l]["X"] for l in train_loads])
    y = np.concatenate([data[l]["y"] for l in train_loads])
    pos = np.concatenate([data[l]["pos"] for l in train_loads])

    cut = 1.0 - config.CALIB_FRACTION
    gap = 0.01  # >= 4 windows (one window length at stride 256) for every recording here
    train_mask, calib_mask = pos < cut - gap, pos >= cut
    val_params = config.IMPAIRMENTS["val"]
    X_calib = impair_fixed(X[calib_mask], val_params["snr_db"], val_params["drift_prob"],
                           val_params["impulse_prob"], config.VAL_NOISE_SEED + test_load)

    test = data[test_load]
    test_params = config.IMPAIRMENTS["test"]
    test_sets = {}
    for i, snr in enumerate(config.BENCHMARK_SNR_DB):
        if snr is None:
            test_sets[snr_label(snr)] = test["X"]
        else:
            test_sets[snr_label(snr)] = impair_fixed(test["X"], snr, test_params["drift_prob"],
                                                     test_params["impulse_prob"],
                                                     config.BENCHMARK_NOISE_SEED + 100 * test_load + i)
    return {
        "X_train": X[train_mask], "y_train": y[train_mask],
        "X_calib": X_calib, "y_calib": y[calib_mask],
        "test_sets": test_sets, "y_test": test["y"], "size_test": test["size"],
    }


def evaluate(model, X, y, sizes, calibrated):
    probs = F.softmax(predict_logits(model, X, calibrated=calibrated), dim=1).numpy()
    m = classification_metrics(probs, y)
    preds = probs.argmax(1)
    m["accuracy_by_size"] = {int(s): float((preds[sizes == s] == y[sizes == s]).mean() * 100.0)
                             for s in np.unique(sizes)}
    m["confusion"] = np.bincount(y * 4 + preds, minlength=16).reshape(4, 4).tolist()
    return m


def run(seeds, epochs, model_keys):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = load_per_load()
    results = {"protocol": {
        "loads": config.ALL_LOADS, "fault_sizes_mils": [7, 14, 21], "seeds": seeds, "epochs": epochs,
        "snr_db": [snr_label(s) for s in config.BENCHMARK_SNR_DB],
        "train_impairments": config.IMPAIRMENTS["train"], "calib_impairments": config.IMPAIRMENTS["val"],
        "test_drift_prob": config.IMPAIRMENTS["test"]["drift_prob"],
        "test_impulse_prob": config.IMPAIRMENTS["test"]["impulse_prob"],
        "calib_fraction": config.CALIB_FRACTION, "model_selection": "final epoch (no validation-based selection)",
    }, "runs": []}

    total = len(config.ALL_LOADS) * len(seeds) * len(model_keys)
    done, t_start = 0, time.time()
    for test_load in config.ALL_LOADS:
        fold = build_fold(data, test_load)
        print(f"\n=== Fold: test load {test_load} HP | train {len(fold['y_train'])} | "
              f"calib {len(fold['y_calib'])} | test {len(fold['y_test'])} windows ===")
        for seed in seeds:
            for key in model_keys:
                set_seed(seed)
                train_loader, _ = make_loaders(fold["X_train"], fold["y_train"],
                                               fold["X_calib"], fold["y_calib"], seed)
                model = train_model(MODEL_SPECS[key][1](), train_loader, None, fold["y_train"],
                                    epochs=epochs, verbose=False, seed=seed)
                calib_acc = float((predict_logits(model, fold["X_calib"], calibrated=False).argmax(1).numpy()
                                   == fold["y_calib"]).mean() * 100.0)
                model = calibrate_temperature(model, fold["X_calib"], fold["y_calib"], verbose=False)

                run_rec = {"test_load": test_load, "seed": seed, "model": key,
                           "temperature": model.temperature.item(), "calib_accuracy": calib_acc, "test": {}}
                for name, X_test in fold["test_sets"].items():
                    run_rec["test"][name] = {
                        "calibrated": evaluate(model, X_test, fold["y_test"], fold["size_test"], True),
                        "uncalibrated": evaluate(model, X_test, fold["y_test"], fold["size_test"], False),
                    }
                results["runs"].append(run_rec)
                with open(OUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(results, f)

                done += 1
                eta = (time.time() - t_start) / done * (total - done) / 60
                cal = run_rec["test"]
                print(f"[{done}/{total}] load {test_load} seed {seed} {key:<13} T={run_rec['temperature']:.3f} "
                      f"calib acc {calib_acc:.1f}% | macro-F1 clean {cal['clean']['calibrated']['macro_f1']:.1f} "
                      f"/ 0dB {cal['0dB']['calibrated']['macro_f1']:.1f} | ETA {eta:.0f} min", flush=True)

    summarize(results, model_keys)


def collect(results, model, snr, metric, calibrated=True, load=None):
    """Values over (fold, seed) pairs, sorted so models pair up for the Wilcoxon test."""
    rows = sorted((r for r in results["runs"] if r["model"] == model and (load is None or r["test_load"] == load)),
                  key=lambda r: (r["test_load"], r["seed"]))
    return np.array([r["test"][snr]["calibrated" if calibrated else "uncalibrated"][metric] for r in rows])


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_test(a, b):
    diff = a - b
    if len(diff) < 2 or np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def summarize(results, model_keys):
    snrs = results["protocol"]["snr_db"]
    summary = {}
    for key in model_keys:
        summary[key] = {}
        for snr in snrs:
            entry = {}
            for metric in SUMMARY_METRICS:
                vals = collect(results, key, snr, metric)
                entry[metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
                if key != REFERENCE:
                    ref = collect(results, REFERENCE, snr, metric)
                    entry[metric]["delta_vs_physics"] = float((vals - ref).mean())
                    entry[metric]["p_value"] = paired_test(vals, ref)
            entry["ece_uncalibrated"] = float(collect(results, key, snr, "ece", calibrated=False).mean())
            entry["macro_f1_by_load"] = {str(l): float(collect(results, key, snr, "macro_f1", load=l).mean())
                                         for l in config.ALL_LOADS}
            summary[key][snr] = entry
    # Holm correction per metric over every (model vs PAC) x SNR comparison
    for metric in SUMMARY_METRICS:
        cells = [summary[k][s][metric] for k in model_keys if k != REFERENCE for s in snrs]
        for cell, p_adj in zip(cells, holm([c["p_value"] for c in cells])):
            cell["p_holm"] = float(p_adj)
    results["summary"] = summary
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)

    print_summary(summary, model_keys, snrs)
    plot(summary, model_keys, snrs)
    write_report(results, summary, model_keys, snrs)


def fmt(m):
    return f"{m['mean']:.2f} ± {m['std']:.2f}"


def print_summary(summary, model_keys, snrs):
    print("\nMacro-F1 (%) on the held-out load, mean ± std over all (fold, seed) runs:")
    print(f"{'Model':<26}" + "".join(f"{s:>17}" for s in snrs))
    for key in model_keys:
        print(f"{MODEL_SPECS[key][0]:<26}" + "".join(f"{fmt(summary[key][s]['macro_f1']):>17}" for s in snrs))


def plot(summary, model_keys, snrs):
    colors = {"baseline": "#64748b", "physics": "#059669", "no_residual": "#ef4444",
              "no_dual": "#f97316", "no_attention": "#eab308"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(len(snrs))
    for ax, metric, title in [(axes[0], "macro_f1", "Macro-F1 (%) vs noise [higher is better]"),
                              (axes[1], "false_alarm_rate", "False alarm rate (%) vs noise [lower is better]"),
                              (axes[2], "ece", "ECE (%) vs noise [lower is better]")]:
        for key in model_keys:
            mean = np.array([summary[key][s][metric]["mean"] for s in snrs])
            std = np.array([summary[key][s][metric]["std"] for s in snrs])
            ax.plot(x, mean, marker="o", color=colors[key], label=MODEL_SPECS[key][0],
                    linewidth=2.5 if key in ("baseline", "physics") else 1.2)
            ax.fill_between(x, mean - std, mean + std, color=colors[key], alpha=0.12)
        ax.set_xticks(x, snrs)
        ax.set_xlabel("Test condition (SNR)")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "benchmark_noise_sweep.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVED] {path}")


def write_report(results, summary, model_keys, snrs):
    proto = results["protocol"]
    n_runs = len(config.ALL_LOADS) * len(proto["seeds"])
    path = os.path.join(RESULTS_DIR, "benchmark_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Leave-One-Load-Out Benchmark with Noise Sweep\n\n")
        f.write(f"Each of the {len(config.ALL_LOADS)} loads is held out in turn; models train on the other three "
                f"loads with all fault sizes (0.007\", 0.014\", 0.021\"), {proto['epochs']} epochs, final epoch kept. "
                f"Temperature is fitted on the last {proto['calib_fraction']:.0%} (in time) of each training recording. "
                f"Test windows get drift (p={proto['test_drift_prob']}) and spikes (p={proto['test_impulse_prob']}) "
                f"plus colored noise at each SNR; \"clean\" has no impairments. Values are mean ± std over "
                f"{n_runs} runs ({len(config.ALL_LOADS)} folds × {len(proto['seeds'])} seeds). p-values: paired "
                f"Wilcoxon signed-rank test against the full PAC-1DCNN over the same runs; \"Holm\" is the "
                f"p-value Holm-adjusted over all model × SNR comparisons in the table.\n\n")

        f.write("## Macro-F1 (%)\n\n| Model | " + " | ".join(snrs) + " |\n|" + " :--- |" * (len(snrs) + 1) + "\n")
        for key in model_keys:
            cells = []
            for s in snrs:
                m = summary[key][s]["macro_f1"]
                cell = fmt(m)
                if key != REFERENCE:
                    cell += f" ({m['delta_vs_physics']:+.2f}, p={m['p_value']:.3f}, Holm {m['p_holm']:.3f})"
                cells.append(cell)
            f.write(f"| {MODEL_SPECS[key][0]} | " + " | ".join(cells) + " |\n")

        f.write("\n## Macro-F1 (%) by held-out load (clean / 0 dB)\n\n| Model | "
                + " | ".join(f"{l} HP" for l in config.ALL_LOADS) + " |\n|" + " :--- |" * (len(config.ALL_LOADS) + 1) + "\n")
        for key in model_keys:
            cells = [f"{summary[key]['clean']['macro_f1_by_load'][str(l)]:.1f} / "
                     f"{summary[key]['0dB']['macro_f1_by_load'][str(l)]:.1f}" for l in config.ALL_LOADS]
            f.write(f"| {MODEL_SPECS[key][0]} | " + " | ".join(cells) + " |\n")

        f.write("\n## False alarm rate and calibration at 0 dB\n\n"
                "| Model | False alarms (%) | Missed faults (%) | ECE calibrated (%) | ECE uncalibrated (%) | Mean T | Calib-set accuracy (%) |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for key in model_keys:
            e = summary[key]["0dB"]
            runs = [r for r in results["runs"] if r["model"] == key]
            f.write(f"| {MODEL_SPECS[key][0]} | {fmt(e['false_alarm_rate'])} | {fmt(e['missed_fault_rate'])} | "
                    f"{fmt(e['ece'])} | {e['ece_uncalibrated']:.2f} | {np.mean([r['temperature'] for r in runs]):.3f} | "
                    f"{np.mean([r['calib_accuracy'] for r in runs]):.2f} |\n")

        f.write("\n## Findings (generated from the numbers above)\n\n")
        for s in snrs:
            if "baseline" in model_keys:
                m = summary["baseline"][s]["macro_f1"]
                if m["p_holm"] < 0.05:
                    better = "PAC-1DCNN" if m["delta_vs_physics"] < 0 else "the baseline"
                    f.write(f"- **{s}**: {better} is significantly better after Holm correction "
                            f"(baseline − PAC = {m['delta_vs_physics']:+.2f} macro-F1, p={m['p_value']:.3f}, "
                            f"Holm {m['p_holm']:.3f}).\n")
                else:
                    f.write(f"- **{s}**: no significant difference between baseline and PAC-1DCNN after Holm "
                            f"correction (baseline − PAC = {m['delta_vs_physics']:+.2f}, p={m['p_value']:.3f}, "
                            f"Holm {m['p_holm']:.3f}).\n")
        for key in [k for k in model_keys if k.startswith("no_")]:
            sig = [s for s in snrs if summary[key][s]["macro_f1"]["p_holm"] < 0.05]
            if not sig:
                f.write(f"- **{MODEL_SPECS[key][0]}**: no significant macro-F1 change at any noise level "
                        f"after Holm correction.\n")
            else:
                parts = [f"{s}: {summary[key][s]['macro_f1']['delta_vs_physics']:+.2f}" for s in sig]
                f.write(f"- **{MODEL_SPECS[key][0]}**: significant change vs full model at "
                        f"{', '.join(parts)} (negative = component helps).\n")
    print(f"[SAVED] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=config.DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    parser.add_argument("--summarize-only", action="store_true", help="rebuild report from benchmark_metrics.json")
    args = parser.parse_args()
    if args.summarize_only:
        with open(OUT_JSON, encoding="utf-8") as fh:
            saved = json.load(fh)
        summarize(saved, [k for k in MODEL_SPECS if any(r["model"] == k for r in saved["runs"])])
    else:
        run(args.seeds, args.epochs, args.models)
