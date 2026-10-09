"""
Checks the CWRU windows for preprocessing artefacts that identify the healthy class.

The healthy recordings are decimated from 48 kHz; the fault recordings are 12 kHz originals.
Two artefacts could follow: (1) the decimation filter rolls off above 4.8 kHz, so the healthy
windows lack the 4.8-6 kHz content the fault windows have; (2) the original recordings keep
their ADC quantisation steps, the decimated ones do not. For every load and both preprocessing
variants (with and without the common band limit of preprocess_data.py) this script reports how
well each artefact separates healthy from faulty windows (separability = max(AUC, 1 - AUC);
0.5 = no separation, 1.0 = perfect). The edge-band share of power is evaluated on clean windows
(with its median level in dB) and after the training noise (15 dB, low-pass or white), which is
what every model sees during training: a difference buried under that noise cannot be learned.

Writes results/cwru_shortcut_check.json.  Usage: python cwru_shortcut_check.py
"""
import glob
import json
import os

import numpy as np
import scipy.signal
from sklearn.metrics import roc_auc_score

import config
from common import add_real_world_impairments
from preprocess_data import load_recording, parse_filename

FS = config.SAMPLING_RATE_HZ
FREQS = np.fft.rfftfreq(config.WINDOW_SIZE, 1.0 / FS)
EDGE_BAND = (5200, 6000)
# Blackman-Harris taper (side lobes ~ -92 dB): without it, leakage from the strong low-frequency
# content (~ -40 dB) would hide the true level of the edge band.
TAPER = scipy.signal.windows.blackmanharris(config.WINDOW_SIZE).astype(np.float64)


def windows(signal):
    """Raw (unstandardised) and standardised drive-end windows, as in preprocess_data.window_signal."""
    starts = range(0, signal.shape[1] - config.WINDOW_SIZE + 1, config.STRIDE)
    raw = np.stack([signal[:, s:s + config.WINDOW_SIZE] for s in starts])
    std = (raw - raw.mean(-1, keepdims=True)) / (raw.std(-1, keepdims=True) + 1e-8)
    return raw[:, 0], std


def edge_fraction(X):
    """Share of drive-end window power in 5.2-6 kHz, the band the 48 kHz decimation filter removes."""
    p = np.abs(np.fft.rfft(X * TAPER, axis=-1)) ** 2
    return p[:, (FREQS >= EDGE_BAND[0]) & (FREQS < EDGE_BAND[1])].sum(-1) / p.sum(-1)


def separability(feature, healthy):
    auc = roc_auc_score(healthy, feature)
    return float(max(auc, 1.0 - auc))


def check(band_limit, rng_seed=0):
    out = {}
    files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.mat")))
    for load in config.ALL_LOADS:
        raw_de, std, labels = [], [], []
        for path in files:
            label, rec_load = parse_filename(os.path.basename(path))
            if label is None or rec_load != load:
                continue
            r, s = windows(load_recording(path, band_limit=band_limit))
            raw_de.append(r)
            std.append(s)
            labels += [label] * len(s)
        raw_de, std, healthy = np.concatenate(raw_de), np.concatenate(std), np.asarray(labels) == 0
        rng = np.random.default_rng(rng_seed)
        edge = edge_fraction(std[:, 0])
        res = {
            "edge_clean": separability(edge, healthy),
            "edge_clean_healthy_db": float(10 * np.log10(np.median(edge[healthy]))),
            "edge_clean_faulty_db": float(10 * np.log10(np.median(edge[~healthy]))),
            # fraction of distinct values in the raw window: low for quantised originals
            "distinct_values": separability(np.array([len(np.unique(w)) for w in raw_de]) / raw_de.shape[1], healthy),
        }
        for spectrum in ("lowpass", "white"):
            noisy = np.stack([add_real_world_impairments(x, snr_db=config.IMPAIRMENTS["train"]["snr_db"],
                                                         drift_prob=0.0, impulse_prob=0.0, rng=rng,
                                                         spectrum=spectrum)[0] for x in std])
            res[f"edge_{spectrum}_15dB"] = separability(edge_fraction(noisy), healthy)
        out[f"load{load}"] = res
        print(f"  band limit {'on ' if band_limit else 'off'} | load {load} | "
              + " | ".join(f"{k} {v:.3f}" for k, v in res.items()))
    return out


if __name__ == "__main__":
    report = {"edge_band_hz": EDGE_BAND, "band_limit_hz": config.BAND_LIMIT_HZ,
              "without_band_limit": check(False), "with_band_limit": check(True)}
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "cwru_shortcut_check.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[SAVED] {os.path.join(config.RESULTS_DIR, 'cwru_shortcut_check.json')}")
