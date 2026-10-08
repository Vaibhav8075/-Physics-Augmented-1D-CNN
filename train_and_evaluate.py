import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from common import (DEVICE, set_seed, load_splits, make_loaders, train_model,
                    calibrate_temperature, evaluate_model, measure_latency_ms)
from models import Baseline1DCNN, PhysicsAugmentedCalibratedCNN

CLASS_NAMES = config.CLASS_NAMES
RESULTS_DIR = config.RESULTS_DIR
METRIC_KEYS = ["accuracy", "macro_f1", "false_alarm_rate", "missed_fault_rate", "ece", "nll"]
MODELS = {"baseline": Baseline1DCNN, "physics": PhysicsAugmentedCalibratedCNN}


def summarize(runs):
    """Mean/std over seeds for each metric."""
    return {k: {"mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0}
            for k in runs[0]}


def strip(metrics):
    return {k: float(metrics[k]) for k in METRIC_KEYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=config.DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    print(f"Train {X_train.shape[0]} | Val {X_val.shape[0]} | Test {X_test.shape[0]} windows "
          f"(test SNR {config.IMPAIRMENTS['test']['snr_db']} dB, unseen load {config.TEST_LOADS})")

    runs = {name: {"val": [], "test": [], "test_uncalibrated": []} for name in MODELS}
    temperatures = {name: [] for name in MODELS}
    figure_results = {}

    for seed in args.seeds:
        for name, cls in MODELS.items():
            print("\n" + "=" * 70)
            print(f" SEED {seed} | TRAINING {name.upper()}")
            print("=" * 70)
            set_seed(seed)
            train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, seed)
            model = train_model(cls(), train_loader, val_loader, y_train, epochs=args.epochs)
            model = calibrate_temperature(model, X_val, y_val)
            temperatures[name].append(model.temperature.item())

            runs[name]["val"].append(strip(evaluate_model(model, X_val, y_val)))
            test_res = evaluate_model(model, X_test, y_test)
            runs[name]["test"].append(strip(test_res))
            runs[name]["test_uncalibrated"].append(strip(evaluate_model(model, X_test, y_test, calibrated=False)))

            if seed == args.seeds[0]:
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{name}_model.pt"))
                figure_results[name] = (model, test_res)

    latency = {name: measure_latency_ms(figure_results[name][0], device="cpu") for name in MODELS}

    metrics = {
        "protocol": {
            "train_loads": config.TRAIN_LOADS, "val_loads": config.VAL_LOADS, "test_loads": config.TEST_LOADS,
            "impairments": config.IMPAIRMENTS, "seeds": args.seeds, "epochs": args.epochs,
            "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
        },
        "models": {
            name: {
                "val": summarize(runs[name]["val"]),
                "test": summarize(runs[name]["test"]),
                "test_uncalibrated": summarize(runs[name]["test_uncalibrated"]),
                "temperature": summarize([{"T": t} for t in temperatures[name]])["T"],
                "per_seed_test": runs[name]["test"],
                "cpu_latency_batch1": latency[name],
            } for name in MODELS
        },
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print_table(metrics)
    generate_figures(figure_results, X_test, y_test)


def print_table(metrics):
    fmt = lambda m: f"{m['mean']:6.2f} ± {m['std']:5.2f}"
    n = len(metrics["protocol"]["seeds"])
    print(f"\nRESULTS (mean ± std over {n} seeds, calibrated, CPU latency at batch size 1):")
    print("-" * 78)
    print(f"{'Split':<6} | {'Metric':<22} | {'Baseline 1D-CNN':<17} | {'PAC-1DCNN':<17}")
    print("-" * 78)
    for split in ["val", "test"]:
        for k in METRIC_KEYS:
            b, p = metrics["models"]["baseline"][split][k], metrics["models"]["physics"][split][k]
            print(f"{split:<6} | {k:<22} | {fmt(b):<17} | {fmt(p):<17}")
        print("-" * 78)
    for name in MODELS:
        lat = metrics["models"][name]["cpu_latency_batch1"]
        print(f"{name:<9} latency: {lat['mean_ms']:.3f} ms mean | {lat['p95_ms']:.3f} ms p95")


@torch.no_grad()
def generate_figures(figure_results, X_test, y_test):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    base_res, phys_res = figure_results["baseline"][1], figure_results["physics"][1]

    # 1. Confusion matrices (first seed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, res, cmap, title in [(axes[0], base_res, "Reds", "(A) Standard 1D-CNN"),
                                 (axes[1], phys_res, "Greens", "(B) Proposed PAC-1DCNN")]:
        cm = confusion_matrix(y_test, res['preds'], normalize='true', labels=range(4)) * 100
        sns.heatmap(cm, annot=True, fmt=".1f", cmap=cmap, ax=ax, xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES, cbar=False)
        ax.set_title(f"{title} (Unseen 3 HP Load + Plant Noise)\nAccuracy: {res['accuracy']:.1f}% | "
                     f"Macro-F1: {res['macro_f1']:.1f}% | False Alarms: {res['false_alarm_rate']:.1f}%",
                     fontsize=11, fontweight='bold')
        ax.set_ylabel("True Condition", fontweight='bold')
        ax.set_xlabel("Predicted Condition", fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "patent_figure_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[SAVED] Confusion Matrix -> {cm_path}")

    # 2. Explainability saliency for one inner-race test window
    model = figure_results["physics"][0].eval()
    sample_idx = np.where(y_test == 1)[0][10]
    x = torch.from_numpy(X_test[sample_idx:sample_idx + 1]).to(DEVICE)
    _, attn_info, residual = model(x)
    raw_signal = X_test[sample_idx, 0]
    residual_signal = residual[0, 0].cpu().numpy()
    t_attn = attn_info["temporal_attention"][0, 0].cpu().numpy()
    t_attn_full = np.interp(np.linspace(0, 1, len(raw_signal)), np.linspace(0, 1, len(t_attn)), t_attn)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    axes[0].plot(raw_signal, color='#1f77b4', linewidth=1.2, label='Impaired vibration window (noise, drift, spikes)')
    axes[0].set_title(f"Explainability (Target Fault: {CLASS_NAMES[y_test[sample_idx]]} @ 3 HP Load)",
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Normalized amplitude", fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[1].plot(residual_signal, color='#d62728', linewidth=1.2, label='Kinematic residual r(t) = x(t) - Smooth(x(t))')
    axes[1].set_ylabel("Residual", fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[2].plot(t_attn_full, color='#2ca02c', linewidth=2.0, label='Learned temporal attention')
    axes[2].fill_between(range(len(t_attn_full)), 0, t_attn_full, color='#2ca02c', alpha=0.25)
    axes[2].set_ylabel("Attention Weight", fontweight='bold')
    axes[2].set_xlabel(f"Time Samples ({config.WINDOW_SIZE} points @ {config.SAMPLING_RATE_HZ // 1000} kHz)",
                       fontweight='bold')
    axes[2].legend(loc='upper right')
    plt.tight_layout()
    saliency_path = os.path.join(RESULTS_DIR, "patent_figure_explainability.png")
    plt.savefig(saliency_path, dpi=300)
    plt.close()
    print(f"[SAVED] Explainability Saliency -> {saliency_path}")


if __name__ == "__main__":
    main()
