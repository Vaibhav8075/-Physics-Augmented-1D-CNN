"""
Robustness study under a fault-size (bearing-wise) split.

Why: in CWRU each faulty bearing is recorded at all four loads, so splitting by
load still puts the same physical bearing in train and test (Abburi et al. 2023;
Vieira et al., MSSP 2026). Here every fold holds out one defect size, i.e. a
different set of physical bearings, following the fault-condition hold-out of
Rosa et al. (arXiv 2407.14625). CWRU has a single healthy bearing, so the
Normal class cannot be separated by bearing; it is split by time (first 80% of
each recording to train, last 20% to test). This residual leakage affects the
Normal class only and is reported as a limitation.

Questions tested:
  Q1  Does the fixed residual (high-pass) filter improve low-SNR robustness,
      at two model capacities (baseline vs baseline+filter; PAC w/o filter vs PAC)?
  Q2  Does any gain depend on the noise spectrum? Test noise is low-pass
      (the original impairment), white, or high-pass coloured.
  Q3  Does it survive training on mixed noise spectra instead of low-pass only?
  Q4  How do the models compare with a WDCNN-style CNN and an envelope-spectrum
      Random Forest (classical baseline)?
Paired Wilcoxon tests over (fold, seed) pairs, Holm-corrected within each
training regime over all pre-specified contrasts and test conditions.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier

import config
from common import (set_seed, make_loaders, train_model, predict_logits, classification_metrics,
                    add_real_world_impairments, NOISE_SPECTRA)
from models import Baseline1DCNN, WideKernelCNN, PhysicsAugmentedCalibratedCNN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_JSON = os.path.join(config.RESULTS_DIR, "robustness_metrics.json")
FAULT_SIZES = [7, 14, 21]
TEST_SNR_DB = [10.0, 5.0, 0.0, -5.0]
NORMAL_TRAIN_FRACTION = 0.8
REGIMES = {
    "lowpass": {**config.IMPAIRMENTS["train"], "spectrum": "lowpass"},
    "mixed": {**config.IMPAIRMENTS["train"], "spectrum": "mixed"},
}
MODEL_SPECS = {
    "baseline": ("Standard 1D-CNN", lambda: Baseline1DCNN()),
    "baseline_hp": ("Standard 1D-CNN + fixed filter", lambda: Baseline1DCNN(highpass_input=True)),
    "no_residual": ("PAC w/o fixed filter", lambda: PhysicsAugmentedCalibratedCNN(use_residual_filter=False)),
    "physics": ("PAC-1DCNN (full)", lambda: PhysicsAugmentedCalibratedCNN()),
    "wdcnn": ("WDCNN-style wide-kernel CNN", lambda: WideKernelCNN()),
    "rf": ("Envelope spectrum + Random Forest", None),
}
# (a, b, question): effect = mean(a - b); positive = a better
CONTRASTS = [
    ("baseline_hp", "baseline", "Q1 filter at baseline capacity"),
    ("physics", "no_residual", "Q1 filter at PAC capacity"),
    ("physics", "baseline", "PAC vs standard CNN"),
    ("physics", "wdcnn", "PAC vs WDCNN-style"),
    ("physics", "rf", "PAC vs classical RF"),
]
METRICS = ["macro_f1", "accuracy", "false_alarm_rate", "missed_fault_rate", "ece"]


# -------------------------------------------------------------
# Data
# -------------------------------------------------------------
def load_all():
    parts = [dict(np.load(os.path.join(config.PROCESSED_DIR, f"load{l}.npz"))) for l in config.ALL_LOADS]
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "size", "pos")}
    data["load"] = np.concatenate([np.full(len(p["y"]), l) for l, p in zip(config.ALL_LOADS, parts)])
    return data


def fold_masks(data, held_out_size):
    normal = data["size"] == 0
    gap = 0.01  # >= one window length at stride 256 for every Normal recording
    train = (~normal & (data["size"] != held_out_size)) | (normal & (data["pos"] < NORMAL_TRAIN_FRACTION - gap))
    test = (~normal & (data["size"] == held_out_size)) | (normal & (data["pos"] >= NORMAL_TRAIN_FRACTION))
    return train, test


def test_conditions():
    conds = [("clean", None)]
    for spec in NOISE_SPECTRA:
        for snr in TEST_SNR_DB:
            conds.append((f"{spec}@{snr:g}dB", dict(snr_db=snr, drift_prob=0.0, impulse_prob=0.0, spectrum=spec)))
    t = config.IMPAIRMENTS["test"]
    conds.append(("drift+spikes", dict(snr_db=None, drift_prob=t["drift_prob"], impulse_prob=t["impulse_prob"])))
    return conds


def impair_fixed(X, seed, spectrum="lowpass", **params):
    """Fixed-seed impairments; spectrum="mixed" draws one spectrum per window."""
    rng = np.random.default_rng(seed)
    specs = rng.choice(NOISE_SPECTRA, size=len(X)) if spectrum == "mixed" else [spectrum] * len(X)
    return np.stack([add_real_world_impairments(x, rng=rng, spectrum=s, **params)
                     for x, s in zip(X, specs)]).astype(np.float32)


# -------------------------------------------------------------
# Classical baseline: band spectrum + envelope spectrum features
# -------------------------------------------------------------
def spectral_features(X):
    """Per channel: log band energies of the spectrum (64 bands, 0-6 kHz) and of the
    Hilbert envelope spectrum (bins 1-52, ~12-610 Hz at 12 kHz / 1024 samples)."""
    win = np.hanning(X.shape[-1]).astype(np.float32)
    spec = np.abs(np.fft.rfft(X * win, axis=-1))[..., :512]
    bands = np.log1p(spec.reshape(*spec.shape[:-1], 64, 8).mean(-1))
    env = np.abs(scipy.signal.hilbert(X, axis=-1))
    env -= env.mean(-1, keepdims=True)
    env_spec = np.log1p(np.abs(np.fft.rfft(env * win, axis=-1))[..., 1:53])
    return np.concatenate([bands, env_spec], axis=-1).reshape(len(X), -1).astype(np.float32)


def train_rf(X_train, y_train, impairments, seed, copies=2):
    feats, labels = [], []
    for c in range(copies):
        Xi = impair_fixed(X_train, seed * 100 + c, **impairments)
        feats.append(spectral_features(Xi))
        labels.append(y_train)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed)
    return rf.fit(np.concatenate(feats), np.concatenate(labels))


# -------------------------------------------------------------
# Experiment
# -------------------------------------------------------------
def score(probs, y):
    m = classification_metrics(probs, y)
    preds = probs.argmax(1)
    m["recall_per_class"] = [float((preds[y == c] == c).mean() * 100.0) for c in range(4)]
    return {k: m[k] for k in METRICS + ["recall_per_class"]}


def run(seeds, epochs, regimes, model_keys, folds=FAULT_SIZES, resume=False):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    data = load_all()
    conds = test_conditions()
    results = {"protocol": {
        "split": "fault-size hold-out (one of 0.007/0.014/0.021 in. per fold, all loads); "
                 f"Normal split by time at {NORMAL_TRAIN_FRACTION:.0%} with a 1% gap",
        "seeds": seeds, "epochs": epochs, "regimes": {r: REGIMES[r] for r in regimes},
        "test_conditions": [c for c, _ in conds], "model_selection": "final epoch",
        "calibration": "none (uncalibrated softmax; ECE reported)",
        "test_noise_seed_base": config.BENCHMARK_NOISE_SEED,
    }, "runs": []}

    if resume and os.path.exists(OUT_JSON):
        # Keep finished folds from an interrupted run, but only if they used the same protocol
        with open(OUT_JSON, encoding="utf-8") as fh:
            saved = json.load(fh)
        if saved["protocol"] != results["protocol"]:
            raise SystemExit(f"--resume: protocol in {OUT_JSON} differs from this run; refusing to merge.")
        results["runs"] = [r for r in saved["runs"] if r["held_out_size"] not in folds]
        kept = sorted({r["held_out_size"] for r in results["runs"]})
        print(f"Resuming: keeping {len(results['runs'])} saved runs from folds {kept}, running folds {folds}")

    nn_keys = [k for k in model_keys if k != "rf"]
    total = len(folds) * len(seeds) * len(regimes) * len(model_keys)
    done, t0 = 0, time.time()
    for held_out in folds:
        tr, te = fold_masks(data, held_out)
        X_tr, y_tr, X_te, y_te = data["X"][tr], data["y"][tr], data["X"][te], data["y"][te]
        print(f"\n=== Fold: hold out {held_out / 1000:.3f} in. | train {len(y_tr)} "
              f"{np.bincount(y_tr, minlength=4).tolist()} | test {len(y_te)} {np.bincount(y_te, minlength=4).tolist()} ===",
              flush=True)

        trained = {}  # (regime, seed, model) -> model on device, or RF
        for regime in regimes:
            for seed in seeds:
                for key in nn_keys:
                    set_seed(seed)
                    loader, _ = make_loaders(X_tr, y_tr, X_tr[:1], y_tr[:1], seed)
                    trained[(regime, seed, key)] = train_model(MODEL_SPECS[key][1](), loader, None, y_tr,
                                                               epochs=epochs, verbose=False,
                                                               impairments=REGIMES[regime], seed=seed)
                    done += 1
                    eta = (time.time() - t0) / done * (total - done) / 60
                    print(f"[{done}/{total}] {held_out:02d} {regime:<7} seed {seed} {key:<12} trained | "
                          f"ETA {eta:.0f} min", flush=True)
                if "rf" in model_keys:
                    trained[(regime, seed, "rf")] = train_rf(X_tr, y_tr, REGIMES[regime], seed)
                    done += 1

        records = {k: {"held_out_size": held_out, "regime": k[0], "seed": k[1], "model": k[2], "test": {}}
                   for k in trained}
        for ci, (name, params) in enumerate(conds):
            X_c = X_te if params is None else impair_fixed(X_te, config.BENCHMARK_NOISE_SEED + 100 * held_out + ci,
                                                           **params)
            feats = spectral_features(X_c) if "rf" in model_keys else None
            for k, model in trained.items():
                if k[2] == "rf":
                    probs = model.predict_proba(feats)
                else:
                    probs = F.softmax(predict_logits(model, X_c, calibrated=False), dim=1).numpy()
                records[k]["test"][name] = score(probs, y_te)
            print(f"  evaluated {name:<16} | macro-F1 " + " ".join(
                f"{m}={np.mean([records[(r, s, m)]['test'][name]['macro_f1'] for r in regimes for s in seeds]):.1f}"
                for m in model_keys), flush=True)
            del X_c, feats
        results["runs"].extend(records.values())
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f)
        del trained
        torch.cuda.empty_cache()

    summarize(results)


# -------------------------------------------------------------
# Statistics & reporting
# -------------------------------------------------------------
def values(results, regime, model, cond, metric="macro_f1"):
    rows = sorted((r for r in results["runs"] if r["regime"] == regime and r["model"] == model),
                  key=lambda r: (r["held_out_size"], r["seed"]))
    return np.array([r["test"][cond][metric] for r in rows])


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def summarize(results):
    regimes = list(results["protocol"]["regimes"])
    conds = results["protocol"]["test_conditions"]
    models = [k for k in MODEL_SPECS if any(r["model"] == k for r in results["runs"])]
    summary = {"means": {}, "contrasts": {}}
    for regime in regimes:
        summary["means"][regime] = {
            m: {c: {met: {"mean": float(values(results, regime, m, c, met).mean()),
                          "std": float(values(results, regime, m, c, met).std(ddof=1))}
                    for met in METRICS} for c in conds} for m in models}
        rows = []
        for a, b, label in CONTRASTS:
            if a not in models or b not in models:
                continue
            for c in conds:
                va, vb = values(results, regime, a, c), values(results, regime, b, c)
                d = va - vb
                p = 1.0 if np.allclose(d, 0) else float(wilcoxon(va, vb).pvalue)
                rows.append({"a": a, "b": b, "question": label, "condition": c, "delta": float(d.mean()),
                             "a_wins": int((d > 0).sum()), "n": len(d), "p": p})
        for row, p_adj in zip(rows, holm(np.array([r["p"] for r in rows]))):
            row["p_holm"] = float(p_adj)
        summary["contrasts"][regime] = rows
    results["summary"] = summary
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    plot(summary, regimes, models, conds)
    write_report(results, summary, regimes, models, conds)


def plot(summary, regimes, models, conds):
    colors = {"baseline": "#64748b", "baseline_hp": "#38bdf8", "no_residual": "#ef4444",
              "physics": "#059669", "wdcnn": "#a855f7", "rf": "#f59e0b"}
    fig, axes = plt.subplots(len(regimes), len(NOISE_SPECTRA), figsize=(16, 4.3 * len(regimes)), squeeze=False)
    x_labels = ["clean"] + [f"{s:g}" for s in TEST_SNR_DB]
    for i, regime in enumerate(regimes):
        for j, spec in enumerate(NOISE_SPECTRA):
            ax = axes[i, j]
            cs = ["clean"] + [f"{spec}@{s:g}dB" for s in TEST_SNR_DB]
            for m in models:
                mean = np.array([summary["means"][regime][m][c]["macro_f1"]["mean"] for c in cs])
                std = np.array([summary["means"][regime][m][c]["macro_f1"]["std"] for c in cs])
                ax.plot(range(len(cs)), mean, marker="o", color=colors[m], label=MODEL_SPECS[m][0],
                        linewidth=2.4 if m == "physics" else 1.4)
                ax.fill_between(range(len(cs)), mean - std, mean + std, color=colors[m], alpha=0.10)
            ax.set_xticks(range(len(cs)), x_labels)
            ax.set_ylim(0, 102)
            ax.set_xlabel("Test SNR (dB)")
            ax.set_ylabel("Macro-F1 (%)")
            ax.set_title(f"train: {regime} noise | test: {spec} noise", fontweight="bold", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(fontsize=7.5, loc="lower left")
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "robustness_noise_spectra.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"[SAVED] {path}")


def write_report(results, summary, regimes, models, conds):
    proto = results["protocol"]
    n = len(FAULT_SIZES) * len(proto["seeds"])
    path = os.path.join(config.RESULTS_DIR, "robustness_report.md")
    fmt = lambda e: f"{e['mean']:.1f} ± {e['std']:.1f}"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Robustness Study: Fault-Size Split, Noise Spectra, Stronger Baselines\n\n")
        f.write(f"Split: {proto['split']}. {proto['epochs']} epochs, final epoch kept, no calibration. "
                f"Values are mean ± std of macro-F1 (%) over {n} runs ({len(FAULT_SIZES)} folds × "
                f"{len(proto['seeds'])} seeds). Noise conditions add only coloured noise (no drift or spikes); "
                f"`drift+spikes` adds only drift and spikes at the test probabilities. "
                f"Contrasts use a paired Wilcoxon signed-rank test over the {n} (fold, seed) pairs, "
                f"Holm-corrected over all contrasts × conditions within each training regime.\n\n"
                f"**Limitation:** CWRU has one healthy bearing, so Normal windows in train and test come from the "
                f"same recordings (split by time). WDCNN-style layer sizes were not checked against the "
                f"original paper's table.\n\n")
        for regime in regimes:
            f.write(f"## Training noise: {regime}\n\n| Model | " + " | ".join(conds) + " |\n|"
                    + " :--- |" * (len(conds) + 1) + "\n")
            for m in models:
                f.write(f"| {MODEL_SPECS[m][0]} | "
                        + " | ".join(fmt(summary["means"][regime][m][c]["macro_f1"]) for c in conds) + " |\n")
            f.write("\n### Contrasts (macro-F1 difference a − b; significant after Holm correction in bold)\n\n"
                    "| Contrast | " + " | ".join(conds) + " |\n|" + " :--- |" * (len(conds) + 1) + "\n")
            rows = summary["contrasts"][regime]
            for a, b, label in CONTRASTS:
                rs = {r["condition"]: r for r in rows if r["a"] == a and r["b"] == b}
                if not rs:
                    continue
                cells = []
                for c in conds:
                    r = rs[c]
                    cell = f"{r['delta']:+.1f} ({r['a_wins']}/{r['n']})"
                    cells.append(f"**{cell}**" if r["p_holm"] < 0.05 else cell)
                f.write(f"| {label}: {a} − {b} | " + " | ".join(cells) + " |\n")
            f.write("\nCell = mean difference (runs where a > b / total).\n\n")
            f.write("### Per-class recall at clean / lowpass@0dB (PAC-1DCNN)\n\n")
            for c in ("clean", "lowpass@0dB"):
                rec = np.mean([r["test"][c]["recall_per_class"] for r in results["runs"]
                               if r["regime"] == regime and r["model"] == "physics"], axis=0)
                f.write(f"- {c}: " + ", ".join(f"{name} {v:.1f}%" for name, v in zip(config.CLASS_NAMES, rec)) + "\n")
            f.write("\n")
    print(f"[SAVED] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=config.DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    parser.add_argument("--folds", type=int, nargs="+", default=FAULT_SIZES, choices=FAULT_SIZES,
                        help="held-out fault sizes (mils) to run")
    parser.add_argument("--resume", action="store_true",
                        help="keep saved runs for folds not listed in --folds (same protocol only)")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        with open(OUT_JSON, encoding="utf-8") as fh:
            summarize(json.load(fh))
    else:
        run(args.seeds, args.epochs, args.regimes, args.models, args.folds, args.resume)
