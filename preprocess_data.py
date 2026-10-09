import os
import re
import glob
import numpy as np
import scipy.io as sio
import scipy.signal

import config
from common import add_real_world_impairments, impair_array  # noqa: F401  (re-exported for older imports)
from download_data import recording_id

LABEL_PREFIXES = {"Normal": 0, "IR": 1, "B": 2, "OR": 3}

# The normal-baseline recordings (97-100) are sampled at 48 kHz, unlike the 12 kHz drive-end fault
# recordings: read as 12 kHz, their shaft-speed line appears at a quarter of RPM/60 at every load,
# and their ~485k samples are 10 s at 48 kHz. They are decimated to 12 kHz with the same zero-phase
# order-8 Chebyshev type I anti-aliasing filter as the Paderborn data (scipy.signal.decimate).
NATIVE_FS_HZ = {"097": 48000, "098": 48000, "099": 48000, "100": 48000}

# That filter is flat to 4.8 kHz and then rolls off, while the 12 kHz fault recordings keep their
# content up to 6 kHz, so the 4.8-6 kHz band alone would separate healthy from faulty windows.
# Every recording is therefore low-passed to a common band limit (same filter family, zero phase),
# which attenuates all of them by at least 56 dB above 4.8 kHz.
BAND_LIMIT_SOS = scipy.signal.cheby1(8, 0.05, config.BAND_LIMIT_HZ / (config.SAMPLING_RATE_HZ / 2),
                                     output="sos")


def parse_filename(filename):
    """'OR007_6_3.mat' -> (label=3, load=3). Matches the fault-type prefix exactly."""
    stem = os.path.splitext(filename)[0]
    match = re.match(r"^(Normal|IR|B|OR)", stem)
    if not match:
        return None, None
    return LABEL_PREFIXES[match.group(1)], int(stem.split("_")[-1])


def fault_size_mils(filename):
    """'IR014_2.mat' -> 14, 'Normal_0.mat' -> 0 (defect diameter in thousandths of an inch)."""
    match = re.match(r"^(?:IR|B|OR)(\d{3})", filename)
    return int(match.group(1)) if match else 0


# Approximate speeds of the normal-baseline recordings whose files store no RPM value
# (CWRU Bearing Data Center, "Normal Baseline Data" page: 1772 rpm at 1 hp, 1750 rpm at 2 hp).
NORMAL_RPM_FALLBACK = {"098": 1772.0, "099": 1750.0}


def recording_rpm(filepath):
    """Shaft speed (rpm) stored in the file, or the published approximate speed if it has none."""
    rid = recording_id(os.path.basename(filepath))
    mat = sio.loadmat(filepath, variable_names=[f"X{rid}RPM"])
    key = f"X{rid}RPM"
    return float(mat[key].ravel()[0]) if key in mat else NORMAL_RPM_FALLBACK[rid]


def load_recording(filepath, band_limit=True):
    """
    Loads the DE/FE channels belonging to this file's own CWRU recording at 12 kHz.
    Some files (e.g. 99.mat) also contain channels from a neighbouring recording
    (X098_*), so the key is matched by recording ID, not by position.
    band_limit=False skips the common low-pass (only for cwru_shortcut_check.py).
    """
    mat = sio.loadmat(filepath)
    rid = recording_id(os.path.basename(filepath))
    de_key, fe_key = f"X{rid}_DE_time", f"X{rid}_FE_time"
    if de_key not in mat:
        raise KeyError(f"{filepath}: expected key {de_key}, found {[k for k in mat if 'time' in k]}")
    de = mat[de_key].ravel()
    fe = mat[fe_key].ravel() if fe_key in mat else de
    n = min(len(de), len(fe))
    signal = np.stack([de[:n], fe[:n]]).astype(np.float64)
    native_fs = NATIVE_FS_HZ.get(rid, config.SAMPLING_RATE_HZ)
    if native_fs != config.SAMPLING_RATE_HZ:
        factor = native_fs // config.SAMPLING_RATE_HZ
        signal = scipy.signal.decimate(signal, factor, ftype="iir", zero_phase=True, axis=-1)
    if band_limit:
        signal = scipy.signal.sosfiltfilt(BAND_LIMIT_SOS, signal, axis=-1)
    return signal.astype(np.float32)


def window_signal(signal):
    """Sliding windows, each channel z-normalized independently."""
    windows = []
    for start in range(0, signal.shape[1] - config.WINDOW_SIZE + 1, config.STRIDE):
        w = signal[:, start:start + config.WINDOW_SIZE]
        w = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8)
        windows.append(w)
    return windows


def create_real_world_dataset():
    print("Generating cross-load dataset (clean windows + fixed-seed impaired val/test)...")
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    split_of_load = {**{l: "train" for l in config.TRAIN_LOADS},
                     **{l: "val" for l in config.VAL_LOADS},
                     **{l: "test" for l in config.TEST_LOADS}}

    for filepath in sorted(glob.glob(os.path.join(config.DATA_DIR, "*.mat"))):
        label, load = parse_filename(os.path.basename(filepath))
        if label is None or load not in split_of_load:
            continue
        if fault_size_mils(os.path.basename(filepath)) not in (0, *config.LEGACY_FAULT_SIZES):
            continue
        windows = window_signal(load_recording(filepath))
        X_list, y_list = splits[split_of_load[load]]
        X_list.extend(windows)
        y_list.extend([label] * len(windows))

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    save = lambda name, arr: np.save(os.path.join(config.PROCESSED_DIR, name), arr)
    for split, (X_list, y_list) in splits.items():
        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.int64)
        save(f"X_{split}_clean.npy", X)
        save(f"y_{split}.npy", y)
        if split != "train":
            # Train impairments are drawn fresh every epoch (common.train_model);
            # val/test impairments are fixed so every model is scored on identical inputs.
            seed = config.VAL_NOISE_SEED if split == "val" else config.TEST_NOISE_SEED
            save(f"X_{split}.npy", impair_array(X, split, seed))
        print(f"  • {split:<5} loads {[l for l, s in split_of_load.items() if s == split]}: "
              f"{len(y)} windows, class counts {np.bincount(y, minlength=4).tolist()}")

    stale = os.path.join(config.PROCESSED_DIR, "X_train.npy")
    if os.path.exists(stale):
        os.remove(stale)  # old pre-impaired training set is no longer used


def create_per_load_dataset():
    """
    Clean windows for every recording, grouped by motor load, for the
    leave-one-load-out benchmark (run_benchmark.py). Alongside each window we
    store the fault size and the window's relative position in its recording,
    so a calibration split can be cut by time rather than by random windows
    (overlapping windows would otherwise leak between the two).
    """
    print("Generating per-load dataset (all fault sizes) for the cross-load benchmark...")
    per_load = {}
    for filepath in sorted(glob.glob(os.path.join(config.DATA_DIR, "*.mat"))):
        filename = os.path.basename(filepath)
        label, load = parse_filename(filename)
        if label is None:
            continue
        windows = window_signal(load_recording(filepath))
        n = len(windows)
        entry = per_load.setdefault(load, {"X": [], "y": [], "size": [], "pos": [], "rec": [], "rpm": []})
        entry["X"].extend(windows)
        entry["y"].extend([label] * n)
        entry["size"].extend([fault_size_mils(filename)] * n)
        entry["pos"].extend(np.arange(n) / n)
        entry["rec"].extend([recording_id(filename)] * n)
        entry["rpm"].extend([recording_rpm(filepath)] * n)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    for load, entry in sorted(per_load.items()):
        np.savez(os.path.join(config.PROCESSED_DIR, f"load{load}.npz"),
                 X=np.asarray(entry["X"], dtype=np.float32), y=np.asarray(entry["y"], dtype=np.int64),
                 size=np.asarray(entry["size"], dtype=np.int64), pos=np.asarray(entry["pos"], dtype=np.float32),
                 rec=np.asarray(entry["rec"]), rpm=np.asarray(entry["rpm"], dtype=np.float32))
        y = np.asarray(entry["y"])
        print(f"  • load {load} HP: {len(y)} windows, class counts {np.bincount(y, minlength=4).tolist()}, "
              f"fault sizes {sorted(set(entry['size']))}")


if __name__ == "__main__":
    create_real_world_dataset()
    create_per_load_dataset()
