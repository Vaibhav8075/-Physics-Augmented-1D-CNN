"""
Live training demo: the same networks, the same code, two CWRU splits.

  load split       train on loads 0-2 HP, test on 3 HP (the same physical bearings appear in both)
  fault-size split train on 0.007 and 0.021 in. defects, test on the unseen 0.014 in. bearings

Prints the training loss and clean test macro-F1 after every epoch, then per-class recall.
Nothing is saved: the results/ files and demo checkpoints are not touched. One seed and a few
epochs only, so the numbers vary from run to run and are not the paper's results (5 seeds,
15 epochs, all folds; see results/benchmark_report.md and results/robustness_report.md).

    python live_demo.py                    # standard CNN and PAC-1DCNN, 5 epochs
    python live_demo.py --models physics --epochs 3
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
from common import set_seed, make_loaders, train_model, predict_logits, classification_metrics, DEVICE
from run_robustness_study import load_all, fold_masks, MODEL_SPECS, REGIMES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SHORT = ["Normal", "Inner", "Ball", "Outer"]


def splits(data):
    load_test = data["load"] == 3
    size_train, size_test = fold_masks(data, 14)
    return {"load split (test 3 HP)": (~load_test, load_test),
            "fault-size split (test 0.014 in.)": (size_train, size_test)}


def run(model_keys, epochs, seed):
    data = load_all()
    print(f"Device: {DEVICE} | {len(data['y'])} CWRU windows | seed {seed} | {epochs} epochs | "
          "training noise: mixed spectra, 15 dB")
    summary = []
    for split_name, (tr, te) in splits(data).items():
        X_tr, y_tr, X_te, y_te = data["X"][tr], data["y"][tr], data["X"][te], data["y"][te]
        print(f"\n=== {split_name}: train {len(y_tr)} windows, test {len(y_te)} windows "
              f"(per class {np.bincount(y_te, minlength=4).tolist()}) ===")
        for key in model_keys:
            name = MODEL_SPECS[key][0]
            print(f"--- {name}")

            def report(epoch, model, loss):
                probs = F.softmax(predict_logits(model, X_te, calibrated=False), dim=1).numpy()
                print(f"  epoch {epoch + 1:2d} | train loss {loss:.3f} | "
                      f"test macro-F1 {classification_metrics(probs, y_te)['macro_f1']:5.1f}%", flush=True)

            set_seed(seed)
            loader, _ = make_loaders(X_tr, y_tr, X_tr[:1], y_tr[:1], seed)
            t0 = time.time()
            model = train_model(MODEL_SPECS[key][1](), loader, None, y_tr, epochs=epochs, verbose=False,
                                impairments=REGIMES["mixed"], seed=seed, on_epoch_end=report)
            pred = predict_logits(model, X_te, calibrated=False).argmax(1).numpy()
            recall = [100 * np.mean(pred[y_te == c] == c) for c in range(4)]
            confusion = np.array([[np.sum((y_te == t) & (pred == p)) for p in range(4)] for t in range(4)])
            f1 = classification_metrics(np.eye(4)[pred], y_te)["macro_f1"]
            print(f"  trained in {time.time() - t0:.0f} s | recall " +
                  "  ".join(f"{s} {r:.0f}%" for s, r in zip(SHORT, recall)))
            print("  confusion (rows = true, columns = predicted):  " + "  ".join(f"{s:>6}" for s in SHORT))
            for s, row in zip(SHORT, confusion):
                print(f"  {s:>45}  " + "  ".join(f"{v:>6}" for v in row))
            summary.append((split_name, name, f1))
            del model
            torch.cuda.empty_cache()

    print("\n=== Clean test macro-F1 after the last epoch ===")
    for split_name, name, f1 in summary:
        print(f"  {split_name:<36} {name:<20} {f1:5.1f}%")
    print("Single seed and short training: indicative only. Paper results: results/*_report.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["baseline", "physics"],
                        choices=[k for k, v in MODEL_SPECS.items() if v[1] is not None])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.models, args.epochs, args.seed)
