"""
Fold-level view of the paired comparisons.

The main tests pair (fold, seed) runs, but runs that share a fold share its test data and,
across folds, overlapping training data, so they are not independent (Dietterich 1998;
Nadeau & Bengio 2003). This script averages each model's score over seeds within a fold
and reports, for every contrast and test condition:
  folds_agree   number of folds whose mean difference has the sign of the overall mean
                difference (P1: 4 folds, P2: 3, P4: 10; P3 has a single split);
  p_nb          P4 only: the corrected resampled t-test of Nadeau & Bengio (2003) over the
                10 fold means, t = mean(d) / sqrt((1/J + n_test/n_train) * var(d)), J - 1 df,
                with n_test/n_train = 6/9 counted in bearings (the independent units), and its
                Holm adjustment within each training regime and contrast family.
Writes results/fold_level_stats.json.  Usage: python fold_level_stats.py
"""
import json
import os

import numpy as np
from scipy import stats

import config
from run_robustness_study import CONTRASTS, holm

P1_SNRS = ["clean", "10dB", "5dB", "0dB", "-5dB", "-10dB"]
P1_MODELS = ["baseline", "no_residual", "no_dual", "no_attention"]
P4_TEST_TRAIN_RATIO = 6 / 9  # two of five bearings per class tested, three trained


def load(name):
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fold_means(runs, fold_key, model, score):
    """{fold: mean over seeds of score(run)} for one model."""
    by_fold = {}
    for r in runs:
        if r["model"] == model:
            by_fold.setdefault(r[fold_key], []).append(score(r))
    return {f: float(np.mean(v)) for f, v in sorted(by_fold.items())}


def summarize_difference(a, b, ratio=None):
    folds = sorted(a)
    d = np.array([a[f] - b[f] for f in folds])
    out = {"delta": float(d.mean()), "fold_deltas": d.round(3).tolist(),
           "folds_agree": int(np.sum(np.sign(d) == np.sign(d.mean()))) if d.mean() != 0 else 0,
           "n_folds": len(d)}
    if ratio is not None:
        sd = d.std(ddof=1)
        t = d.mean() / np.sqrt((1 / len(d) + ratio) * sd ** 2) if sd > 0 else 0.0
        out["p_nb"] = float(2 * stats.t.sf(abs(t), df=len(d) - 1)) if sd > 0 else 1.0
    return out


def p1(bench):
    rows = []
    for m in P1_MODELS:
        for snr in P1_SNRS:
            score = lambda r: r["test"][snr]["calibrated"]["macro_f1"]
            row = summarize_difference(fold_means(bench["runs"], "test_load", "physics", score),
                                       fold_means(bench["runs"], "test_load", m, score))
            rows.append({"a": "physics", "b": m, "condition": snr, **row})
    return rows


def contrast_table(res, fold_key, ratio=None):
    out = {}
    regimes = res["protocol"]["regimes"] if "protocol" in res else res["header"]["regimes"]
    conds = res["protocol"]["test_conditions"] if "protocol" in res else res["header"]["test_conditions"]
    models = {r["model"] for r in res["runs"]}
    for regime in regimes:
        runs = [r for r in res["runs"] if r["regime"] == regime]
        rows = []
        for a, b, label, family in CONTRASTS:
            if a not in models or b not in models:
                continue
            for c in conds:
                score = lambda r: r["test"][c]["macro_f1"]
                row = summarize_difference(fold_means(runs, fold_key, a, score),
                                           fold_means(runs, fold_key, b, score), ratio)
                rows.append({"a": a, "b": b, "question": label, "family": family, "condition": c, **row})
        if ratio is not None:
            for family in sorted({r["family"] for r in rows}):
                fam = [r for r in rows if r["family"] == family]
                for row, p_adj in zip(fam, holm(np.array([r["p_nb"] for r in fam]))):
                    row["p_nb_holm"] = float(p_adj)
        out[regime] = rows
    return out


def main():
    report = {}
    bench = load("benchmark_metrics.json")
    if bench:
        report["p1_lolo"] = p1(bench)
    rob = load("robustness_metrics.json")
    if rob:
        report["p2_fault_size"] = contrast_table(rob, "held_out_size")
    cv = load("paderborn_real_cv_metrics.json")
    if cv:
        report["p4_real_cv"] = contrast_table(cv, "fold", ratio=P4_TEST_TRAIN_RATIO)
    path = os.path.join(config.RESULTS_DIR, "fold_level_stats.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    if "p4_real_cv" in report:
        for regime, rows in report["p4_real_cv"].items():
            sig = [r for r in rows if r["p_nb_holm"] < 0.05]
            print(f"P4 {regime}: {len(sig)} of {len(rows)} contrasts significant at fold level (corrected t-test, Holm)"
                  + "".join(f"\n  {r['a']} - {r['b']} @ {r['condition']}: {r['delta']:+.1f}, "
                            f"{r['folds_agree']}/{r['n_folds']} folds, p_holm={r['p_nb_holm']:.3f}" for r in sig))
    print(f"[SAVED] {path}")


if __name__ == "__main__":
    main()
