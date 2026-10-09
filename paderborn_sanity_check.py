"""
Control experiment for the Paderborn study: separates pipeline errors from genuine
generalisation failure.

  leaky      windows of the SAME bearings split at random into train/test halves
             (bearing identity leaks); near-perfect scores show that preprocessing,
             labels and features are correct.
  disjoint   the 10 real-damage bearing-wise folds (run_paderborn_study.real_cv),
             clean training, so the gap to "leaky" is attributable to unseen bearings.

  chance     macro-F1 of a uniformly random classifier and accuracy of a constant
             majority-class classifier on each protocol's test set, as reference levels.

Envelope-spectrum random forest only (CPU). Writes results/paderborn_sanity.json.
"""
import os
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import config
from run_paderborn_study import windows, folds_for, measurement_accuracy, label_of, TEST_STRIDE, FS, REAL_CV
from run_robustness_study import spectral_features

SETTINGS = ["N15_M07_F10"]


def forest(seed=0):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed)


def chance_levels(protocol, draws=500):
    """Window macro-F1 of uniform random guessing (Monte Carlo mean over test folds) and the
    measurement accuracy of always predicting the most frequent test class."""
    f1s, majority = [], []
    rng = np.random.default_rng(0)
    for _, _, test_b in folds_for(protocol):
        _, y, meas = windows(test_b, SETTINGS, TEST_STRIDE)
        f1s.append(np.mean([f1_score(y, rng.integers(0, 3, len(y)), average="macro") * 100
                            for _ in range(draws)]))
        meas_labels = np.array([y[meas == m][0] for m in np.unique(meas)])
        majority.append(np.bincount(meas_labels, minlength=3).max() / len(meas_labels) * 100)
    return {"random_macro_f1": float(np.mean(f1s)), "majority_measurement_accuracy": float(np.mean(majority)),
            "random_measurement_accuracy": 100.0 / 3}


def main():
    out = {"settings": SETTINGS, "model": "envelope-spectrum random forest, 300 trees, clean training"}
    out["chance"] = {p: chance_levels(p) for p in ("real_cv", "a2r")}
    for p, c in out["chance"].items():
        print(f"chance    {p:<8} random macro-F1 {c['random_macro_f1']:.1f} | "
              f"majority-class measurement accuracy {c['majority_measurement_accuracy']:.1f}")
    bearings = sorted(sum(REAL_CV.values(), []))
    X, y, meas = windows(bearings, SETTINGS, TEST_STRIDE)
    feats = spectral_features(X, fs=FS)

    # leaky: random halves of the windows of all 15 real-damage bearings, 5 repetitions
    leaky = []
    for rep in range(5):
        idx = np.random.default_rng(rep).permutation(len(y))
        tr, te = idx[: len(y) // 2], idx[len(y) // 2:]
        proba = forest(rep).fit(feats[tr], y[tr]).predict_proba(feats[te])
        leaky.append({"macro_f1": f1_score(y[te], proba.argmax(1), average="macro") * 100,
                      "measurement_accuracy": measurement_accuracy(proba, y[te], meas[te])})
    out["leaky"] = leaky

    # disjoint: the 10 bearing-wise folds
    disjoint = []
    for name, train_b, test_b in folds_for("real_cv"):
        tr = np.isin(np.array([m.split("/")[0] for m in meas]), train_b)
        te = ~tr
        rf = forest(0).fit(feats[tr], y[tr])
        proba = rf.predict_proba(feats[te])
        per_bearing = {}
        for b in test_b:
            sel = np.array([m.split("/")[0] == b for m in meas[te]])
            per_bearing[b] = {"true": label_of(b), "predicted_counts": np.bincount(proba[sel].argmax(1), minlength=3).tolist()}
        disjoint.append({"fold": name, "macro_f1": f1_score(y[te], proba.argmax(1), average="macro") * 100,
                         "measurement_accuracy": measurement_accuracy(proba, y[te], meas[te]), "per_bearing": per_bearing})
    out["disjoint"] = disjoint

    for key in ("leaky", "disjoint"):
        f1 = [r["macro_f1"] for r in out[key]]
        acc = [r["measurement_accuracy"] for r in out[key]]
        out[f"{key}_summary"] = {"macro_f1_mean": float(np.mean(f1)), "macro_f1_min": float(np.min(f1)),
                                 "macro_f1_max": float(np.max(f1)), "measurement_accuracy_mean": float(np.mean(acc)),
                                 "measurement_accuracy_min": float(np.min(acc)), "measurement_accuracy_max": float(np.max(acc))}
        print(f"{key:<9} macro-F1 {np.mean(f1):.1f} ({np.min(f1):.1f}-{np.max(f1):.1f}) | "
              f"measurement accuracy {np.mean(acc):.1f} ({np.min(acc):.1f}-{np.max(acc):.1f})")
    with open(os.path.join(config.RESULTS_DIR, "paderborn_sanity.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
