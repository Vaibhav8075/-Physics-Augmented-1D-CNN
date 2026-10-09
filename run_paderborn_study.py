"""
Robustness study on the Paderborn University bearing data (Lessmeier et al., 2016).

Unlike CWRU, every bearing here is a separate physical specimen, so train and
test can be separated by bearing for all classes, including healthy. Two
protocols from Lessmeier et al. (PHM Society European Conference 2016) are used:

  a2r      Table 8: train on artificially damaged bearings, test on bearings with
           real (accelerated-lifetime) damage. One split; statistics over seeds.
  real_cv  Table 10: real-damage bearings only; 3 bearings per class train, the
           other 2 test, for all C(5,3) = 10 combinations.

Classes: healthy / inner ring / outer ring. Vibration channel only, decimated to
16 kHz (preprocess_paderborn.py), 2048-sample windows (128 ms), each window
z-normalised. Operating setting N15_M07_F10 only by default, as in the paper.
Test conditions, training regimes, models and statistics follow
run_robustness_study.py (fault-size study on CWRU).
"""
import os
import sys
import json
import time
import argparse
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config
from common import set_seed, make_loaders, train_model, predict_logits, NOISE_SPECTRA
from models import Baseline1DCNN, WideKernelCNN, PhysicsAugmentedCalibratedCNN
from preprocess_paderborn import PROCESSED_DIR, FS
from run_robustness_study import (REGIMES, METRICS, TEST_SNR_DB, test_conditions, impair_fixed,
                                  spectral_features, train_rf, score, holm)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WINDOW = 2048
TRAIN_STRIDE = 1024   # half-overlapping windows for training
TEST_STRIDE = 2048    # non-overlapping windows for testing
CLASS_NAMES = ["Healthy", "Inner ring", "Outer ring"]

# Lessmeier et al. 2016, Table 8 (artificial -> real) and Table 10 (real damage, 5 per class)
A2R_TRAIN = ["K002", "KA01", "KA05", "KA07", "KI01", "KI05", "KI07"]
A2R_TEST = ["K001", "KA04", "KA15", "KA16", "KA22", "KA30", "KI14", "KI16", "KI17", "KI18", "KI21"]
REAL_CV = {0: ["K001", "K002", "K003", "K004", "K005"],
           1: ["KI04", "KI14", "KI16", "KI18", "KI21"],
           2: ["KA04", "KA15", "KA16", "KA22", "KA30"]}
# Reference results reported by Lessmeier et al. (vibration features, setting N15_M07_F10,
# accuracy per 4-s measurement): Table 9 (a2r) and Table 11 (real_cv)
REFERENCE = {"a2r": "ensemble 75.0%, RF 64.1% (Table 9)", "real_cv": "CART / RF / ensemble 98.5% (Table 11)"}

MODEL_SPECS = {
    "baseline": ("Standard 1D-CNN", lambda: Baseline1DCNN(1, 3)),
    "baseline_hp": ("Standard 1D-CNN + fixed filter", lambda: Baseline1DCNN(1, 3, highpass_input=True)),
    "no_residual": ("PAC w/o fixed filter", lambda: PhysicsAugmentedCalibratedCNN(1, 3, use_residual_filter=False)),
    "physics": ("PAC-1DCNN (full)", lambda: PhysicsAugmentedCalibratedCNN(1, 3)),
    "wdcnn": ("WDCNN-style wide-kernel CNN", lambda: WideKernelCNN(1, 3)),
    "rf": ("Envelope spectrum + Random Forest", None),
}
CONTRASTS = [
    ("baseline_hp", "baseline", "Q1 filter at baseline capacity"),
    ("physics", "no_residual", "Q1 filter at PAC capacity"),
    ("physics", "baseline", "PAC vs standard CNN"),
    ("physics", "wdcnn", "PAC vs WDCNN-style"),
    ("physics", "rf", "PAC vs classical RF"),
]


def label_of(bearing):
    return 0 if bearing.startswith("K0") else (1 if bearing.startswith("KI") else 2)


def folds_for(protocol):
    if protocol == "a2r":
        return [("a2r", A2R_TRAIN, A2R_TEST)]
    folds = []
    for idx in combinations(range(5), 3):
        train = [REAL_CV[c][i] for c in REAL_CV for i in idx]
        test = [REAL_CV[c][i] for c in REAL_CV for i in range(5) if i not in idx]
        folds.append(("train " + "".join(str(i + 1) for i in idx), train, test))
    return folds


def windows(bearings, settings, stride):
    """Z-normalised windows (N, 1, WINDOW), labels, and a measurement id per window."""
    X, y, meas_id = [], [], []
    for b in bearings:
        d = np.load(os.path.join(PROCESSED_DIR, f"{b}.npz"))
        assert int(d["fs"]) == FS
        for sig, setting, m in zip(d["signals"], d["settings"], d["meas"]):
            if setting not in settings:
                continue
            starts = range(0, len(sig) - WINDOW + 1, stride)
            w = np.stack([sig[s:s + WINDOW] for s in starts])
            w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-8)
            X.append(w[:, None, :].astype(np.float32))
            y.extend([label_of(b)] * len(w))
            meas_id.extend([f"{b}/{setting}/{m}"] * len(w))
    return np.concatenate(X), np.asarray(y, dtype=np.int64), np.asarray(meas_id)


def measurement_accuracy(probs, y, meas_id):
    """Accuracy per 4-s measurement (mean window probabilities), comparable in unit to Lessmeier et al."""
    ids, inv = np.unique(meas_id, return_inverse=True)
    mean_probs = np.zeros((len(ids), probs.shape[1]))
    np.add.at(mean_probs, inv, probs)
    labels = np.zeros(len(ids), dtype=np.int64)
    labels[inv] = y
    return float((mean_probs.argmax(1) == labels).mean() * 100.0)


def full_score(probs, y, meas_id):
    m = score(probs, y)
    m["measurement_accuracy"] = measurement_accuracy(probs, y, meas_id)
    return m


def protocol_header(protocol, seeds, epochs, regimes, settings):
    return {"protocol": protocol, "seeds": seeds, "epochs": epochs, "regimes": {r: REGIMES[r] for r in regimes},
            "settings": settings, "fs_hz": FS, "window": WINDOW, "train_stride": TRAIN_STRIDE,
            "test_stride": TEST_STRIDE, "test_conditions": [c for c, _ in test_conditions()],
            "model_selection": "final epoch", "calibration": "none (uncalibrated softmax; ECE reported)",
            "test_noise_seed_base": config.BENCHMARK_NOISE_SEED}


def run_protocol(protocol, seeds, epochs, regimes, model_keys, settings, resume):
    out_json = os.path.join(config.RESULTS_DIR, f"paderborn_{protocol}_metrics.json")
    header = protocol_header(protocol, seeds, epochs, regimes, settings)
    results = {"header": header, "runs": []}
    if resume and os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as fh:
            saved = json.load(fh)
        if saved["header"] != header:
            raise SystemExit(f"--resume: settings in {out_json} differ from this run; refusing to merge.")
        results["runs"] = saved["runs"]
    done_folds = {r["fold"] for r in results["runs"]}

    conds = test_conditions()
    nn_keys = [k for k in model_keys if k != "rf"]
    folds = folds_for(protocol)
    t0, n_done, n_total = time.time(), 0, sum(f[0] not in done_folds for f in folds) * len(seeds) * len(regimes)
    for fold_index, (fold_name, train_b, test_b) in enumerate(folds):
        if fold_name in done_folds:
            print(f"[{protocol}] fold {fold_name}: already done, skipped")
            continue
        X_tr, y_tr, _ = windows(train_b, settings, TRAIN_STRIDE)
        X_te, y_te, meas_te = windows(test_b, settings, TEST_STRIDE)
        print(f"\n=== [{protocol}] fold {fold_name} | train {train_b} -> {len(y_tr)} windows "
              f"{np.bincount(y_tr, minlength=3).tolist()} | test {test_b} -> {len(y_te)} windows "
              f"{np.bincount(y_te, minlength=3).tolist()} ===", flush=True)

        trained = {}
        for regime in regimes:
            for seed in seeds:
                for key in nn_keys:
                    set_seed(seed)
                    loader, _ = make_loaders(X_tr, y_tr, X_tr[:1], y_tr[:1], seed)
                    trained[(regime, seed, key)] = train_model(MODEL_SPECS[key][1](), loader, None, y_tr,
                                                               epochs=epochs, verbose=False,
                                                               impairments=REGIMES[regime], seed=seed)
                n_done += 1
                eta = (time.time() - t0) / n_done * (n_total - n_done) / 60
                print(f"  trained {regime} seed {seed} ({n_done}/{n_total}) | ETA {eta:.0f} min", flush=True)

        records = {(r, s, k): {"fold": fold_name, "regime": r, "seed": s, "model": k, "test": {}}
                   for r in regimes for s in seeds for k in model_keys}
        rf_feats = {}
        for ci, (name, params) in enumerate(conds):
            # test noise depends only on (fold, condition): identical for every model and regime
            X_c = X_te if params is None else impair_fixed(X_te, config.BENCHMARK_NOISE_SEED + 1000 * fold_index + ci,
                                                           **params)
            for k, model in trained.items():
                probs = F.softmax(predict_logits(model, X_c, calibrated=False), dim=1).numpy()
                records[k]["test"][name] = full_score(probs, y_te, meas_te)
            if "rf" in model_keys:
                rf_feats[name] = spectral_features(X_c, fs=FS)
            del X_c
        del trained
        torch.cuda.empty_cache()

        if "rf" in model_keys:  # one forest in memory at a time
            for regime in regimes:
                for seed in seeds:
                    rf = train_rf(X_tr, y_tr, REGIMES[regime], seed, fs=FS)
                    for name in rf_feats:
                        records[(regime, seed, "rf")]["test"][name] = full_score(rf.predict_proba(rf_feats[name]),
                                                                                  y_te, meas_te)
                    del rf
        for name in ("clean", "lowpass@0dB", "white@0dB"):
            print(f"  {name:<12} macro-F1 " + " ".join(
                f"{k}={np.mean([records[(r, s, k)]['test'][name]['macro_f1'] for r in regimes for s in seeds]):.1f}"
                for k in model_keys), flush=True)

        results["runs"].extend(records.values())
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f)
        del X_tr, X_te, rf_feats
    return results


# -------------------------------------------------------------
# Statistics & reporting
# -------------------------------------------------------------
def values(results, regime, model, cond, metric="macro_f1"):
    rows = sorted((r for r in results["runs"] if r["regime"] == regime and r["model"] == model),
                  key=lambda r: (r["fold"], r["seed"]))
    return np.array([r["test"][cond][metric] for r in rows])


def summarize(results):
    from scipy.stats import wilcoxon
    h = results["header"]
    models = [k for k in MODEL_SPECS if any(r["model"] == k for r in results["runs"])]
    summary = {"means": {}, "contrasts": {}}
    for regime in h["regimes"]:
        summary["means"][regime] = {
            m: {c: {met: {"mean": float(values(results, regime, m, c, met).mean()),
                          "std": float(values(results, regime, m, c, met).std(ddof=1))}
                    for met in METRICS + ["measurement_accuracy"]} for c in h["test_conditions"]} for m in models}
        rows = []
        for a, b, label in CONTRASTS:
            if a not in models or b not in models:
                continue
            for c in h["test_conditions"]:
                va, vb = values(results, regime, a, c), values(results, regime, b, c)
                d = va - vb
                p = 1.0 if np.allclose(d, 0) else float(wilcoxon(va, vb).pvalue)
                rows.append({"a": a, "b": b, "question": label, "condition": c, "delta": float(d.mean()),
                             "a_wins": int((d > 0).sum()), "n": len(d), "p": p})
        for row, p_adj in zip(rows, holm(np.array([r["p"] for r in rows]))):
            row["p_holm"] = float(p_adj)
        summary["contrasts"][regime] = rows
    results["summary"] = summary
    return results, models


def plot(all_results):
    colors = {"baseline": "#64748b", "baseline_hp": "#38bdf8", "no_residual": "#ef4444",
              "physics": "#059669", "wdcnn": "#a855f7", "rf": "#f59e0b"}
    panels = [(p, reg) for p, (res, _) in all_results.items() for reg in res["header"]["regimes"]]
    fig, axes = plt.subplots(len(panels), len(NOISE_SPECTRA), figsize=(16, 4.2 * len(panels)), squeeze=False)
    x_labels = ["clean"] + [f"{s:g}" for s in TEST_SNR_DB]
    for i, (protocol, regime) in enumerate(panels):
        res, models = all_results[protocol]
        for j, spec in enumerate(NOISE_SPECTRA):
            ax = axes[i, j]
            cs = ["clean"] + [f"{spec}@{s:g}dB" for s in TEST_SNR_DB]
            for m in models:
                mean = np.array([res["summary"]["means"][regime][m][c]["macro_f1"]["mean"] for c in cs])
                std = np.array([res["summary"]["means"][regime][m][c]["macro_f1"]["std"] for c in cs])
                ax.plot(range(len(cs)), mean, marker="o", color=colors[m], label=MODEL_SPECS[m][0],
                        linewidth=2.4 if m == "physics" else 1.4)
                ax.fill_between(range(len(cs)), mean - std, mean + std, color=colors[m], alpha=0.10)
            ax.set_xticks(range(len(cs)), x_labels)
            ax.set_ylim(0, 102)
            ax.set_xlabel("Test SNR (dB)")
            ax.set_ylabel("Macro-F1 (%)")
            ax.set_title(f"{protocol} | train: {regime} | test: {spec} noise", fontweight="bold", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(fontsize=7.5, loc="lower left")
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "paderborn_noise_spectra.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"[SAVED] {path}")


def write_report(all_results):
    path = os.path.join(config.RESULTS_DIR, "paderborn_report.md")
    fmt = lambda e: f"{e['mean']:.1f} ± {e['std']:.1f}"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Paderborn Bearing Study: Unseen Physical Bearings, Noise Spectra, Stronger Baselines\n\n"
                "Data: Paderborn University KAt Bearing DataCenter (Lessmeier et al., 2016; CC BY-NC 4.0). "
                f"Vibration channel decimated 64 kHz -> {FS // 1000} kHz, {WINDOW}-sample windows, classes "
                "healthy / inner ring / outer ring. Every test bearing is a physical specimen never seen in "
                "training (healthy bearings included). Noise conditions add only coloured noise; "
                "`drift+spikes` adds only drift and spikes. Paired Wilcoxon signed-rank tests over (fold, seed) "
                "pairs, Holm-corrected over all contrasts × conditions within each protocol and training regime. "
                "`meas. acc.` = accuracy per 4-s measurement (mean of window probabilities).\n\n")
        for protocol, (res, models) in all_results.items():
            h = res["header"]
            conds = h["test_conditions"]
            n_pairs = len(values(res, list(h["regimes"])[0], models[0], "clean"))
            f.write(f"## Protocol `{protocol}`\n\n{len(folds_for(protocol))} fold(s) × {len(h['seeds'])} seeds = "
                    f"{n_pairs} runs per model; settings {h['settings']}; {h['epochs']} epochs. "
                    f"Reference (Lessmeier et al., different features and classifiers): {REFERENCE[protocol]}.\n\n")
            for regime in h["regimes"]:
                means = res["summary"]["means"][regime]
                f.write(f"### Training noise: {regime}\n\n| Model | meas. acc. (clean) | "
                        + " | ".join(conds) + " |\n|" + " :--- |" * (len(conds) + 2) + "\n")
                for m in models:
                    f.write(f"| {MODEL_SPECS[m][0]} | {fmt(means[m]['clean']['measurement_accuracy'])} | "
                            + " | ".join(fmt(means[m][c]["macro_f1"]) for c in conds) + " |\n")
                f.write("\nMacro-F1 (%) per window unless stated.\n\n**Contrasts** (a − b macro-F1; bold = "
                        "significant after Holm):\n\n| Contrast | " + " | ".join(conds) + " |\n|"
                        + " :--- |" * (len(conds) + 1) + "\n")
                rows = res["summary"]["contrasts"][regime]
                for a, b, label in CONTRASTS:
                    rs = {r["condition"]: r for r in rows if r["a"] == a and r["b"] == b}
                    if not rs:
                        continue
                    cells = [(f"**{rs[c]['delta']:+.1f} ({rs[c]['a_wins']}/{rs[c]['n']})**" if rs[c]["p_holm"] < 0.05
                              else f"{rs[c]['delta']:+.1f} ({rs[c]['a_wins']}/{rs[c]['n']})") for c in conds]
                    f.write(f"| {label}: {a} − {b} | " + " | ".join(cells) + " |\n")
                f.write("\nPer-class recall, clean (healthy / inner / outer): "
                        + "; ".join(f"{MODEL_SPECS[m][0]} " + "/".join(
                            f"{v:.0f}" for v in np.mean([r["test"]["clean"]["recall_per_class"] for r in res["runs"]
                                                          if r["regime"] == regime and r["model"] == m], axis=0))
                                    for m in models) + "\n\n")
    print(f"[SAVED] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", nargs="+", default=["real_cv", "a2r"], choices=["real_cv", "a2r"])
    parser.add_argument("--seeds-cv", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds-a2r", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    parser.add_argument("--settings", nargs="+", default=["N15_M07_F10"])
    parser.add_argument("--resume", action="store_true", help="skip folds already saved with identical settings")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    all_results = {}
    for protocol in args.protocols:
        if args.summarize_only:
            with open(os.path.join(config.RESULTS_DIR, f"paderborn_{protocol}_metrics.json"), encoding="utf-8") as fh:
                res = json.load(fh)
        else:
            seeds = args.seeds_cv if protocol == "real_cv" else args.seeds_a2r
            res = run_protocol(protocol, seeds, args.epochs, args.regimes, args.models, args.settings, args.resume)
        res, models = summarize(res)
        with open(os.path.join(config.RESULTS_DIR, f"paderborn_{protocol}_metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        all_results[protocol] = (res, models)
    plot(all_results)
    write_report(all_results)
