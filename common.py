"""Shared data, training, calibration and evaluation utilities."""
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------
# Industrial impairment simulation
# -------------------------------------------------------------
NOISE_SPECTRA = ("lowpass", "white", "highpass")


def noise_kernel(spectrum):
    """
    FIR shaping kernel applied to white noise (odd length, symmetric except lowpass):
      lowpass  - 15-tap decaying exponential (the original impairment; energy mostly below ~1 kHz at 12 kHz)
      white    - identity (flat spectrum)
      highpass - second difference [-0.5, 1, -0.5] (energy rises towards Nyquist, i.e. into the band
                 a moving-average residual filter passes)
    """
    if spectrum == "lowpass":
        return np.exp(-np.linspace(0, 3, 15)).astype(np.float32)
    if spectrum == "white":
        return np.ones(1, dtype=np.float32)
    if spectrum == "highpass":
        return np.array([-0.5, 1.0, -0.5], dtype=np.float32)
    raise ValueError(f"unknown noise spectrum {spectrum!r}; expected one of {NOISE_SPECTRA}")


def add_real_world_impairments(signal, snr_db=8.0, drift_prob=0.8, impulse_prob=0.3, rng=None,
                               spectrum="lowpass"):
    """
    Simulates a factory environment on a (C, L) window:
    1. Coloured background noise at the given SNR (spectrum: see noise_kernel; snr_db=None skips it)
    2. Non-linear sensor baseline drift
    3. Transient electromagnetic shock spikes
    """
    rng = np.random.default_rng() if rng is None else rng
    augmented = signal.astype(np.float32).copy()
    L = signal.shape[-1]

    white = rng.standard_normal(signal.shape).astype(np.float32)
    if snr_db is None:
        white[:] = 0.0  # keep RNG consumption identical so drift/spikes match the noisy conditions
        snr_db = 0.0
    kernel = noise_kernel(spectrum)
    colored_noise = np.stack([np.convolve(white[c], kernel, mode="same") for c in range(signal.shape[0])])

    signal_power = np.mean(signal ** 2) + 1e-8
    noise_power = np.mean(colored_noise ** 2) + 1e-8
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    augmented += colored_noise * np.sqrt(target_noise_power / noise_power)

    if rng.random() < drift_prob:
        t = np.linspace(0, 2 * np.pi * rng.uniform(0.5, 2.0), L)
        drift = np.sin(t) * rng.uniform(0.3, 0.8) + np.linspace(0, rng.uniform(-0.5, 0.5), L)
        augmented += drift.astype(np.float32)

    if rng.random() < impulse_prob:
        spike_locs = rng.choice(L, size=rng.integers(2, 6), replace=False)
        for loc in spike_locs:
            augmented[:, loc:min(loc + 3, L)] += rng.choice([-1, 1]) * rng.uniform(2.0, 4.0)

    return augmented


def impair_array(X_clean, split, seed):
    """Deterministically impairs a whole split (used for val/test)."""
    rng = np.random.default_rng(seed)
    params = config.IMPAIRMENTS[split]
    return np.stack([add_real_world_impairments(x, rng=rng, **params) for x in X_clean]).astype(np.float32)


def _shape_noise(white, spectrum):
    """np.convolve(..., mode="same") with an odd kernel == conv1d with the flipped kernel, padding k//2."""
    k = torch.from_numpy(noise_kernel(spectrum)[::-1].copy()).to(white.device).view(1, 1, -1)
    return F.conv1d(white, k, padding=k.shape[-1] // 2)


def impair_batch(x, snr_db, drift_prob, impulse_prob, generator=None, spectrum="lowpass"):
    """
    Batched torch version of add_real_world_impairments (same distribution),
    applied on the GPU to every training batch so impairments are fresh each epoch.
    x: (B, C, L) tensor. spectrum may also be "mixed": each window draws one of NOISE_SPECTRA.
    """
    B, C, L = x.shape
    dev, g = x.device, generator
    rand = lambda *shape: torch.rand(*shape, device=dev, generator=g)

    # 1. Colored noise (power is normalised per window below, so kernel gain does not matter)
    white = torch.randn(B * C, 1, L, device=dev, generator=g)
    if spectrum == "mixed":
        choice = torch.randint(0, len(NOISE_SPECTRA), (B, 1, 1), device=dev, generator=g)
        colored = torch.zeros(B, C, L, device=dev)
        for i, s in enumerate(NOISE_SPECTRA):
            colored = colored + _shape_noise(white, s).view(B, C, L) * (choice == i)
    else:
        colored = _shape_noise(white, spectrum).view(B, C, L)
    signal_power = x.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-8
    noise_power = colored.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-8
    target_power = signal_power / (10 ** (snr_db / 10.0))
    out = x + colored * torch.sqrt(target_power / noise_power)

    # 2. Baseline drift (same on all channels)
    t = torch.linspace(0, 1, L, device=dev).view(1, 1, L)
    cycles = 0.5 + 1.5 * rand(B, 1, 1)
    amp = 0.3 + 0.5 * rand(B, 1, 1)
    ramp_end = -0.5 + rand(B, 1, 1)
    drift = torch.sin(2 * np.pi * cycles * t) * amp + ramp_end * t
    out = out + drift * (rand(B, 1, 1) < drift_prob)

    # 3. 2-5 spikes of +/-U(2, 4), 3 samples wide, on all channels
    n_spikes = torch.randint(2, 6, (B, 1), device=dev, generator=g)
    active = (torch.arange(5, device=dev).view(1, 5) < n_spikes) & (rand(B, 1) < impulse_prob)
    locs = torch.randint(0, L, (B, 5), device=dev, generator=g)
    signs = torch.randint(0, 2, (B, 5), device=dev, generator=g) * 2 - 1
    vals = signs * (2.0 + 2.0 * rand(B, 5)) * active
    spikes = torch.zeros(B, L + 2, device=dev)
    for offset in range(3):
        spikes.scatter_add_(1, locs + offset, vals)
    return out + spikes[:, :L].unsqueeze(1)


def load_splits():
    """Returns clean train windows plus fixed-seed impaired val/test windows."""
    p = lambda name: os.path.join(config.PROCESSED_DIR, name)
    X_train = np.load(p("X_train_clean.npy"))
    y_train = np.load(p("y_train.npy"))
    X_val = np.load(p("X_val.npy"))
    y_val = np.load(p("y_val.npy"))
    X_test = np.load(p("X_test.npy"))
    y_test = np.load(p("y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loaders(X_train, y_train, X_val, y_val, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    # Clean windows; train_model adds fresh impairments to every batch
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=config.BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=256, shuffle=False)
    return train_loader, val_loader


# -------------------------------------------------------------
# Training & calibration
# -------------------------------------------------------------
def class_weights(y, num_classes=4):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, X, calibrated=True, batch_size=256):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
        logits, _, _ = model(xb, return_calibrated=calibrated)
        out.append(logits.cpu())
    return torch.cat(out)


def train_model(model, train_loader, val_loader, y_train, epochs=config.EPOCHS, lr=config.LR, verbose=True,
                impairments=config.IMPAIRMENTS["train"], seed=0):
    """
    Trains with class-weighted CE on clean windows plus fresh per-batch impairments.
    With a val_loader, keeps the epoch with the best validation macro-F1;
    with val_loader=None, returns the final epoch.
    """
    model = model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train).to(DEVICE))
    noise_gen = torch.Generator(device=DEVICE)
    noise_gen.manual_seed(seed)

    if val_loader is not None:
        X_val = val_loader.dataset.tensors[0].numpy()
        y_val = val_loader.dataset.tensors[1].numpy()

    best_f1, best_weights = -1.0, None
    for epoch in range(epochs):
        model.train()
        total_loss, total = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            if impairments is not None:
                X_batch = impair_batch(X_batch, generator=noise_gen, **impairments)
            optimizer.zero_grad()
            logits, _, _ = model(X_batch, return_calibrated=False)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            total += X_batch.size(0)

        if val_loader is None:
            if verbose and ((epoch + 1) % 3 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {total_loss/total:.4f}")
            continue

        val_preds = predict_logits(model, X_val, calibrated=False).argmax(1).numpy()
        val_f1 = f1_score(y_val, val_preds, average="macro")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and ((epoch + 1) % 3 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {total_loss/total:.4f} | "
                  f"Val Acc: {accuracy_score(y_val, val_preds)*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}%")

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def calibrate_temperature(model, X_val, y_val, verbose=True):
    """Post-hoc temperature scaling: fits log(T) on validation NLL with the network frozen."""
    logits = predict_logits(model, X_val, calibrated=False).to(DEVICE)
    labels = torch.from_numpy(y_val).to(DEVICE)

    log_t = nn.Parameter(torch.zeros(1, device=DEVICE))
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    nll_before = F.cross_entropy(logits, labels).item()
    optimizer.step(closure)
    temperature = log_t.exp().item()
    nll_after = F.cross_entropy(logits / temperature, labels).item()
    if verbose:
        print(f"[CALIBRATION] T = {temperature:.4f} | Val NLL {nll_before:.4f} -> {nll_after:.4f}")
    model.temperature.fill_(temperature)
    return model


# -------------------------------------------------------------
# Metrics
# -------------------------------------------------------------
def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error in percent."""
    confidences = probs.max(axis=1)
    accuracies = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            ece += abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * in_bin.mean()
    return ece * 100.0


def classification_metrics(probs, labels):
    preds = probs.argmax(axis=1)
    normal = labels == 0
    return {
        "accuracy": accuracy_score(labels, preds) * 100.0,
        "macro_f1": f1_score(labels, preds, average="macro") * 100.0,
        "false_alarm_rate": float((preds[normal] != 0).mean() * 100.0) if normal.any() else 0.0,
        "missed_fault_rate": float((preds[~normal] == 0).mean() * 100.0) if (~normal).any() else 0.0,
        "ece": compute_ece(probs, labels),
        "nll": float(F.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-12)), torch.from_numpy(labels)).item()),
    }


def evaluate_model(model, X, y, calibrated=True):
    probs = F.softmax(predict_logits(model, X, calibrated=calibrated), dim=1).numpy()
    metrics = classification_metrics(probs, y)
    metrics["preds"] = probs.argmax(axis=1)
    metrics["probs"] = probs
    return metrics


@torch.no_grad()
def measure_latency_ms(model, n_runs=500, warmup=50, device="cpu"):
    """Per-window (batch=1) latency with warm-up and device sync."""
    model = model.to(device).eval()
    x = torch.randn(1, 2, config.WINDOW_SIZE, device=device)
    sync = torch.cuda.synchronize if device != "cpu" and torch.cuda.is_available() else (lambda: None)
    for _ in range(warmup):
        model(x)
    sync()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    model.to(DEVICE)
    return {"mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95))}
