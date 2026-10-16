"""
Dataset tour: what the raw CWRU and Paderborn files contain and what the models are given.

Prints an inventory of both datasets and draws, per class, the vibration signal, its amplitude
spectrum and its envelope spectrum with the bearing fault frequencies marked
(results/dataset_cwru_overview.png, results/dataset_paderborn_overview.png).

    python show_dataset.py            # print, save and show the figures
    python show_dataset.py --no-show  # print and save only
"""
import argparse
import glob
import os
import sys

import numpy as np
import scipy.io as sio
import scipy.signal
import matplotlib.pyplot as plt

import config
from download_data import recording_id
from preprocess_data import parse_filename, fault_size_mils, recording_rpm, load_recording, NATIVE_FS_HZ
from kinematic_features import CWRU_DE_ORDERS, PADERBORN_ORDERS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PB_RAW = os.path.join("paderborn_data", "raw")
PB_PROCESSED = os.path.join("paderborn_data", "processed")
PB_SETTING = "N15_M07_F10"  # 1500 rpm, 0.7 Nm, 1000 N: the setting used in the study
PB_SHAFT_HZ = 1500 / 60
PB_EXAMPLES = [("K001", "Healthy"), ("KI04", "Inner ring (real damage)"), ("KA04", "Outer ring (real damage)")]
FAULT_COLOURS = {"inner": "tab:red", "outer": "tab:blue", "ball": "tab:green"}


def envelope_spectrum(x, fs):
    """Band-pass fs/12..fs/3 (as in kinematic_features.py), Hilbert envelope, amplitude spectrum."""
    sos = scipy.signal.butter(4, [fs / 12, fs / 3], btype="bandpass", fs=fs, output="sos")
    env = np.abs(scipy.signal.hilbert(scipy.signal.sosfiltfilt(sos, x)))
    env -= env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env)
    return np.fft.rfftfreq(len(env), 1 / fs), spec


def amplitude_spectrum(x, fs):
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) / len(x)
    return np.fft.rfftfreq(len(x), 1 / fs), 20 * np.log10(spec + 1e-12)


def plot_row(axes, x, fs, shaft_hz, orders, title, env_max_hz):
    t = np.arange(int(0.1 * fs)) / fs
    axes[0].plot(t * 1000, x[:len(t)], lw=0.7)
    axes[0].set_title(title, fontsize=9, loc="left")
    axes[0].set_xlabel("time (ms)", fontsize=8)
    f, s = amplitude_spectrum(x, fs)
    axes[1].plot(f / 1000, s, lw=0.5)
    axes[1].set_xlabel("frequency (kHz)", fontsize=8)
    axes[1].set_ylabel("dB", fontsize=8)
    f, s = envelope_spectrum(x, fs)
    keep = f <= env_max_hz
    axes[2].plot(f[keep], s[keep], color="k", lw=0.7)
    for name, order in orders.items():
        for h in (1, 2, 3):
            axes[2].axvline(h * order * shaft_hz, color=FAULT_COLOURS[name], ls="--", lw=0.8,
                            alpha=1.0 if h == 1 else 0.45, label=name if h == 1 else None)
    axes[2].set_xlabel("envelope frequency (Hz)", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)


def cwru_tour():
    files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.mat")))
    print("=" * 78)
    print(f"CWRU Bearing Data Center: {len(files)} recordings in {config.DATA_DIR}/ "
          "(drive-end bearing SKF 6205, motor loads 0-3 HP)")
    print("=" * 78)
    if not files:
        print("  no recordings found; run download_data.py")
        return

    example = os.path.join(config.DATA_DIR, "IR007_0.mat")
    if os.path.exists(example):
        mat = sio.loadmat(example)
        print("Inside one file (IR007_0.mat):")
        for k in (k for k in mat if not k.startswith("__")):
            print(f"  {k:<14} shape {mat[k].shape}")
        print("  DE = drive-end accelerometer, FE = fan-end, BA = base; RPM = shaft speed")

    print(f"\n{'file':<15}{'class':<19}{'size':>6}{'load':>6}{'rec':>6}{'fs (kHz)':>10}{'seconds':>9}{'rpm':>7}")
    for path in files:
        name = os.path.basename(path)
        label, load = parse_filename(name)
        rid = recording_id(name)
        fs = NATIVE_FS_HZ.get(rid, config.SAMPLING_RATE_HZ)
        rec = sio.loadmat(path, variable_names=[f"X{rid}_DE_time"])[f"X{rid}_DE_time"]
        size = fault_size_mils(name)
        print(f"{name:<15}{config.CLASS_NAMES[label]:<19}{(f'0.{size:03d}' if size else '-'):>6}{load:>6}"
              f"{rid:>6}{fs / 1000:>10.0f}{len(rec) / fs:>9.1f}{recording_rpm(path):>7.0f}")

    print("\nPreprocessing (preprocess_data.py):")
    print("  - the healthy recordings 097-100 are 48 kHz and are decimated to 12 kHz;")
    print(f"  - every recording is low-passed to {config.BAND_LIMIT_HZ} Hz (removes two artefacts that "
          "identify the healthy class);")
    print(f"  - windows of {config.WINDOW_SIZE} samples ({config.WINDOW_SIZE / config.SAMPLING_RATE_HZ * 1000:.0f} ms) "
          f"every {config.STRIDE} samples, DE and FE channels, z-normalised per window.")
    print(f"\n{'load':<6}{'windows':>8}  " + "  ".join(f"{c:>18}" for c in config.CLASS_NAMES))
    for load in config.ALL_LOADS:
        path = os.path.join(config.PROCESSED_DIR, f"load{load}.npz")
        if os.path.exists(path):
            y = np.load(path)["y"]
            counts = np.bincount(y, minlength=len(config.CLASS_NAMES))
            print(f"{load:<6}{len(y):>8}  " + "  ".join(f"{c:>18}" for c in counts))
    print("  Every faulty bearing is recorded at all four loads, so a load split reuses the same bearings.")

    fig, axes = plt.subplots(4, 3, figsize=(13, 9.5), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    fs = config.SAMPLING_RATE_HZ
    for row, name in enumerate(["Normal_0.mat", "IR007_0.mat", "B007_0.mat", "OR007_6_0.mat"]):
        path = os.path.join(config.DATA_DIR, name)
        x = load_recording(path)[0]  # drive-end channel, 12 kHz, band-limited
        shaft = recording_rpm(path) / 60
        label, _ = parse_filename(name)
        plot_row(axes[row], x, fs, shaft, CWRU_DE_ORDERS,
                 f"{config.CLASS_NAMES[label]}: {name}, drive end, {shaft * 60:.0f} rpm", env_max_hz=600)
        axes[row, 1].axvline(config.BAND_LIMIT_HZ / 1000, color="grey", ls=":", lw=1)
    axes[0, 1].set_title(f"amplitude spectrum (dotted: {config.BAND_LIMIT_HZ / 1000:.0f} kHz band limit)",
                         fontsize=9)
    axes[0, 2].set_title("envelope spectrum; dashed: fault frequencies x1-3", fontsize=9)
    axes[0, 2].legend(fontsize=7, loc="upper right")
    fig.suptitle("CWRU: one recording per class (0 HP, 0.007 in defects), after preprocessing", fontsize=11)
    fig.tight_layout()
    out = os.path.join(config.RESULTS_DIR, "dataset_cwru_overview.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


def paderborn_tour():
    print("\n" + "=" * 78)
    print("Paderborn University KAt bearing data (Lessmeier et al. 2016; SKF 6203; CC BY-NC 4.0)")
    print("=" * 78)
    bearings = sorted(d for d in os.listdir(PB_RAW) if os.path.isdir(os.path.join(PB_RAW, d))) \
        if os.path.isdir(PB_RAW) else []
    if not bearings:
        print("  no raw data found; run download_paderborn.py")
        return
    groups = {"healthy (K0..)": [b for b in bearings if b.startswith("K0")],
              "outer ring (KA..)": [b for b in bearings if b.startswith("KA")],
              "inner ring (KI..)": [b for b in bearings if b.startswith("KI")]}
    for g, bs in groups.items():
        print(f"  {g:<18} {len(bs):>2} bearings: {' '.join(bs)}")
    n_mat = sum(len(glob.glob(os.path.join(PB_RAW, b, "*.mat"))) for b in bearings)
    print(f"  {len(bearings)} bearings, {n_mat} measurement files; each file = 4 s at 64 kHz, "
          "4 operating settings x 20 repetitions per bearing")
    print("  Real damage (accelerated life tests): KA04 KA15 KA16 KA22 KA30 KI04 KI14 KI16 KI17 KI18 KI21;")
    print("  the other KA/KI bearings have artificially made damage (run_paderborn_study.py, after Lessmeier Tables 8, 10).")
    print("  Every bearing is a different physical specimen, so bearing-disjoint splits are possible.")
    print(f"\nPreprocessing (preprocess_paderborn.py): vibration_1 channel, decimated 64 -> 16 kHz;")
    print(f"  the study uses setting {PB_SETTING} only, 2048-sample windows (128 ms), z-normalised.")

    fig, axes = plt.subplots(3, 3, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    for row, (bearing, label) in enumerate(PB_EXAMPLES):
        path = os.path.join(PB_PROCESSED, f"{bearing}.npz")
        if not os.path.exists(path):
            print(f"  {path} missing; run preprocess_paderborn.py")
            plt.close(fig)
            return
        z = np.load(path)
        idx = np.flatnonzero(z["settings"] == PB_SETTING)[0]
        plot_row(axes[row], z["signals"][idx].astype(np.float64), int(z["fs"]), PB_SHAFT_HZ, PADERBORN_ORDERS,
                 f"{label}: {bearing}, measurement {z['meas'][idx]}, {PB_SETTING}", env_max_hz=400)
    axes[0, 1].set_title("amplitude spectrum", fontsize=9)
    axes[0, 2].set_title("envelope spectrum; dashed: fault frequencies x1-3", fontsize=9)
    axes[0, 2].legend(fontsize=7, loc="upper right")
    fig.suptitle("Paderborn: one 4-s measurement per class (16 kHz, 1500 rpm)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(config.RESULTS_DIR, "dataset_paderborn_overview.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    args = parser.parse_args()
    cwru_tour()
    paderborn_tour()
    if not args.no_show:
        plt.show()
