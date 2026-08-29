import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from models import Baseline1DCNN, PhysicsAugmentedCalibratedCNN

PROCESSED_DIR = "processed_data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal (Healthy)", "Inner Race Fault", "Ball Fault", "Outer Race Fault"]

def compute_ece(probs, labels, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece * 100.0

def train_model(model, train_loader, val_loader, epochs=15, lr=0.001):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits, _, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                logits, _, _ = model(X_val)
                preds = torch.argmax(logits, dim=1)
                v_correct += (preds == y_val).sum().item()
                v_total += y_val.size(0)
        val_acc = v_correct / v_total

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    model.load_state_dict(best_weights)
    return model

def calibrate_temperature(model, val_loader):
    """Tunes temperature parameter on validation logits to minimize NLL."""
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(DEVICE)
            logits, _, _ = model(X_val, return_calibrated=False)
            logits_list.append(logits)
            labels_list.append(y_val.to(DEVICE))
            
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    temp_param = nn.Parameter(torch.ones(1, device=DEVICE) * 1.5)
    optimizer = optim.LBFGS([temp_param], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(logits / torch.clamp(temp_param, min=0.05), labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    opt_temp = torch.clamp(temp_param, min=0.05).item()
    print(f"[CALIBRATION] Optimized Temperature T = {opt_temp:.4f}")
    model.temperature.data = torch.tensor([opt_temp], device=DEVICE)
    return model

def evaluate_model(model, X_eval, y_test):
    model.eval()
    X_tensor = torch.from_numpy(X_eval).to(DEVICE)
    
    start_time = time.time()
    with torch.no_grad():
        logits, attn_info, res = model(X_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    inference_time_ms = ((time.time() - start_time) / len(X_eval)) * 1000.0

    acc = accuracy_score(y_test, preds) * 100.0
    ece = compute_ece(probs, y_test)
    
    # False Alarm Rate (False Positive Rate on Normal Class 0)
    normal_indices = np.where(y_test == 0)[0]
    false_alarms = np.sum(preds[normal_indices] != 0)
    false_alarm_rate = (false_alarms / len(normal_indices)) * 100.0 if len(normal_indices) > 0 else 0.0

    return {
        "accuracy": acc,
        "false_alarm_rate": false_alarm_rate,
        "ece": ece,
        "inference_ms": inference_time_ms,
        "preds": preds,
        "probs": probs,
        "attn_info": attn_info,
        "residual": res
    }

def main():
    print("Loading Real-World Industrial Dataset (Cross-Load Domain Shift)...")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"), allow_pickle=True)
    X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"), allow_pickle=True)
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"), allow_pickle=True)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=64, shuffle=False)

    print("\n" + "="*70)
    print(" 1. TRAINING BASELINE 1D-CNN (Standard Data-Driven Model)")
    print("="*70)
    baseline_model = Baseline1DCNN()
    baseline_model = train_model(baseline_model, train_loader, val_loader, epochs=15, lr=0.001)
    torch.save(baseline_model.state_dict(), os.path.join(RESULTS_DIR, "baseline_model.pt"))

    print("\n" + "="*70)
    print(" 2. TRAINING PHYSICS-AUGMENTED CALIBRATED CNN (Patent Architecture)")
    print("="*70)
    physics_model = PhysicsAugmentedCalibratedCNN()
    physics_model = train_model(physics_model, train_loader, val_loader, epochs=15, lr=0.001)
    physics_model = calibrate_temperature(physics_model, val_loader)
    torch.save(physics_model.state_dict(), os.path.join(RESULTS_DIR, "physics_model.pt"))

    print("\n" + "="*70)
    print(" 3. BENCHMARKING ON REAL-WORLD DOMAIN SHIFT & UNSEEN LOAD 3 HP")
    print("="*70)
    # Validation Evaluation (Load 2 HP)
    base_val = evaluate_model(baseline_model, X_val, y_val)
    phys_val = evaluate_model(physics_model, X_val, y_val)

    # Hard Unseen Test Evaluation (Load 3 HP + Harsh Factory Variance)
    base_test = evaluate_model(baseline_model, X_test, y_test)
    phys_test = evaluate_model(physics_model, X_test, y_test)

    print("\nREAL-WORLD BENCHMARK RESULTS (CROSS-LOAD DOMAIN GENERALIZATION):")
    print("-" * 80)
    print(f"{'Operating Regime':<25} | {'Metric':<22} | {'Standard 1D-CNN':<15} | {'Patent Architecture':<18}")
    print("-" * 80)
    print(f"{'Intermediate Load (2 HP)':<25} | {'Accuracy':<22} | {base_val['accuracy']:.2f}%{'':<9} | {phys_val['accuracy']:.2f}%")
    print(f"{'Intermediate Load (2 HP)':<25} | {'False Alarm Rate':<22} | {base_val['false_alarm_rate']:.2f}%{'':<9} | {phys_val['false_alarm_rate']:.2f}%")
    print(f"{'Intermediate Load (2 HP)':<25} | {'Calibration Error(ECE)':<22} | {base_val['ece']:.2f}%{'':<9} | {phys_val['ece']:.2f}%")
    print("-" * 80)
    print(f"{'Unseen Load 3 HP + Noise':<25} | {'Accuracy':<22} | {base_test['accuracy']:.2f}%{'':<9} | {phys_test['accuracy']:.2f}%")
    print(f"{'Unseen Load 3 HP + Noise':<25} | {'False Alarm Rate':<22} | {base_test['false_alarm_rate']:.2f}%{'':<9} | {phys_test['false_alarm_rate']:.2f}%")
    print(f"{'Unseen Load 3 HP + Noise':<25} | {'Calibration Error(ECE)':<22} | {base_test['ece']:.2f}%{'':<9} | {phys_test['ece']:.2f}%")
    print("-" * 80)
    print(f"Edge Latency per Window: Baseline = {base_test['inference_ms']:.3f} ms | Patent Architecture = {phys_test['inference_ms']:.3f} ms")

    generate_figures(base_test, phys_test, X_test, y_test)

def generate_figures(base_res, phys_res, X_test, y_test):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Confusion Matrix Comparison under Real-World Domain Shift
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cm_base = confusion_matrix(y_test, base_res['preds'], normalize='true') * 100
    cm_phys = confusion_matrix(y_test, phys_res['preds'], normalize='true') * 100

    sns.heatmap(cm_base, annot=True, fmt=".1f", cmap="Reds", ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    axes[0].set_title(f"(A) Standard 1D-CNN (Unseen 3 HP Load + Plant Noise)\nAccuracy: {base_res['accuracy']:.1f}% | False Alarms: {base_res['false_alarm_rate']:.1f}%", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("True Condition", fontweight='bold')
    axes[0].set_xlabel("Predicted Condition", fontweight='bold')

    sns.heatmap(cm_phys, annot=True, fmt=".1f", cmap="Greens", ax=axes[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    axes[1].set_title(f"(B) Proposed Physics-Augmented Architecture (Patent)\nAccuracy: {phys_res['accuracy']:.1f}% | False Alarms: {phys_res['false_alarm_rate']:.1f}%", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("True Condition", fontweight='bold')
    axes[1].set_xlabel("Predicted Condition", fontweight='bold')

    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "patent_figure_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[SAVED] Real-World Confusion Matrix Drawing -> {cm_path}")

    # 2. Explainability Saliency Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    sample_idx = np.where(y_test == 1)[0][10] # Inner race fault sample
    raw_signal = X_test[sample_idx, 0, :]
    
    if phys_res['residual'] is not None:
        residual_signal = phys_res['residual'][sample_idx, 0, :].cpu().numpy()
    else:
        residual_signal = raw_signal

    t_attn = phys_res['attn_info']['temporal_attention'][sample_idx, 0, :].cpu().numpy()
    t_attn_full = np.interp(np.linspace(0, 1, len(raw_signal)), np.linspace(0, 1, len(t_attn)), t_attn)

    axes[0].plot(raw_signal, color='#1f77b4', linewidth=1.2, label='Industrial Vibration Feed (With Thermal Drift & Spikes)')
    axes[0].set_title(f"FIG. 4: Patent Real-World Explainability (Target Fault: {CLASS_NAMES[y_test[sample_idx]]} @ 3 HP Load)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Vibration (g)", fontweight='bold')
    axes[0].legend(loc='upper right')

    axes[1].plot(residual_signal, color='#d62728', linewidth=1.2, label='Kinematic Residual Filter Output r(t) [Drift Eliminated]')
    axes[1].set_ylabel("Residual Amplitude", fontweight='bold')
    axes[1].legend(loc='upper right')

    axes[2].plot(t_attn_full, color='#2ca02c', linewidth=2.0, label='Learned Temporal Attention Saliency (Localized Fault Bursts)')
    axes[2].fill_between(range(len(t_attn_full)), 0, t_attn_full, color='#2ca02c', alpha=0.25)
    axes[2].set_ylabel("Attention Weight", fontweight='bold')
    axes[2].set_xlabel("Time Samples (Sliding Window: 1024 points @ 12 kHz)", fontweight='bold')
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    saliency_path = os.path.join(RESULTS_DIR, "patent_figure_explainability.png")
    plt.savefig(saliency_path, dpi=300)
    plt.close()
    print(f"[SAVED] Real-World Explainability Saliency Drawing -> {saliency_path}")

if __name__ == "__main__":
    main()
