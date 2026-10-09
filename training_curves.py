"""
Convergence check for the shared training recipe (15 epochs, final epoch kept, no tuning).

Trains every network for 30 epochs on one CWRU fault-size fold (0.014 in. held out) and one
Paderborn real-damage fold (the first), seed 0, mixed training noise, and records after each
epoch the mean training loss and the clean test macro-F1. The test scores are a diagnostic of
the recipe only; they were not used to choose it, and the studies keep the 15th epoch.

Writes results/training_curves.json and results/training_curves.png.
Usage: python training_curves.py [--epochs 30]
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import f1_score

import config
from common import set_seed, make_loaders, train_model, predict_logits
import run_robustness_study as cwru
import run_paderborn_study as pu

NETS = ["baseline", "baseline_hp", "baseline_lhp", "no_residual", "physics", "no_dual", "wdcnn"]


def curves(specs, X_tr, y_tr, X_te, y_te, epochs, regime="mixed", seed=0):
    out = {}
    for key in NETS:
        log = {"loss": [], "test_macro_f1": []}

        def record(epoch, model, loss):
            preds = F.softmax(predict_logits(model, X_te, calibrated=False), dim=1).numpy().argmax(1)
            log["loss"].append(float(loss))
            log["test_macro_f1"].append(float(f1_score(y_te, preds, average="macro") * 100.0))

        set_seed(seed)
        loader, _ = make_loaders(X_tr, y_tr, X_tr[:1], y_tr[:1], seed)
        train_model(specs[key][1](), loader, None, y_tr, epochs=epochs, verbose=False,
                    impairments=cwru.REGIMES[regime], seed=seed, on_epoch_end=record)
        out[key] = log
        f1 = log["test_macro_f1"]
        print(f"  {key:<13} macro-F1 epoch 15: {f1[14]:.1f} | epoch {epochs}: {f1[-1]:.1f} | "
              f"max {max(f1):.1f} (epoch {int(np.argmax(f1)) + 1}) | loss {log['loss'][14]:.3f} -> {log['loss'][-1]:.3f}",
              flush=True)
    return out


def main(epochs):
    report = {"epochs": epochs, "regime": "mixed", "seed": 0}
    data = cwru.load_all()
    tr, te = cwru.fold_masks(data, 14)
    print("CWRU P2, 0.014 in. held out")
    report["cwru_p2_fold14"] = curves(cwru.MODEL_SPECS, data["X"][tr], data["y"][tr], data["X"][te], data["y"][te],
                                      epochs)
    name, train_b, test_b = pu.folds_for("real_cv")[0]
    X_tr, y_tr, _ = pu.windows(train_b, ["N15_M07_F10"], pu.TRAIN_STRIDE)
    X_te, y_te, _ = pu.windows(test_b, ["N15_M07_F10"], pu.TEST_STRIDE)
    print(f"Paderborn P4, fold {name}")
    report["pu_p4_fold1"] = curves(pu.MODEL_SPECS, X_tr, y_tr, X_te, y_te, epochs)
    with open(os.path.join(config.RESULTS_DIR, "training_curves.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for col, (key, title) in enumerate((("cwru_p2_fold14", "CWRU P2 (0.014 in. held out)"),
                                        ("pu_p4_fold1", "Paderborn P4 (fold 1)"))):
        for m, log in report[key].items():
            x = np.arange(1, epochs + 1)
            axes[0, col].plot(x, log["loss"], label=m)
            axes[1, col].plot(x, log["test_macro_f1"], label=m)
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("training loss")
        axes[1, col].set_ylabel("clean test macro-F1 (%)")
        axes[1, col].set_xlabel("epoch")
        for ax in axes[:, col]:
            ax.axvline(config.EPOCHS, color="k", linestyle=":", linewidth=1)
            ax.grid(True, alpha=0.4)
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    print(f"[SAVED] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    main(parser.parse_args().epochs)
