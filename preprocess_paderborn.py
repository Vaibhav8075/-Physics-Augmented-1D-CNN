"""
Converts the Paderborn bearing data (download_paderborn.py) into compact arrays.

For every bearing: the housing vibration signal (vibration_1, 64 kHz) of each
4-second measurement is decimated by 4 to 16 kHz (8th-order Chebyshev I
anti-aliasing filter, zero phase) and stored with its operating setting and
measurement number in paderborn_data/processed/{bearing}.npz.
Windowing and normalisation happen in run_paderborn_study.py.
"""
import os
import re
import glob

import numpy as np
import scipy.io as sio
import scipy.signal

from download_paderborn import SAVE_DIR as RAW_DIR, BEARINGS

PROCESSED_DIR = os.path.join("paderborn_data", "processed")
RAW_FS = 64000
DECIMATION = 4
FS = RAW_FS // DECIMATION
SETTINGS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]  # Lessmeier et al. 2016, Table 6
FILE_RE = re.compile(r"^(N\d\d_M\d\d_F\d\d)_(K[AIB]?\d\d+)_(\d+)\.mat$")


def load_vibration(path):
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    rec = mat[[k for k in mat if not k.startswith("__")][0]]
    for y in np.atleast_1d(rec.Y):
        if y.Name == "vibration_1":
            return np.asarray(y.Data, dtype=np.float64)
    raise KeyError(f"{path}: no vibration_1 channel")


def process_bearing(bearing):
    signals, settings, meas = [], [], []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, bearing, "*.mat"))):
        match = FILE_RE.match(os.path.basename(path))
        if not match or match.group(2) != bearing:
            print(f"  [WARN] unexpected file name {os.path.basename(path)}; skipped")
            continue
        try:
            x = load_vibration(path)
        except Exception as exc:  # a few archives are known to contain unreadable files
            print(f"  [WARN] {os.path.basename(path)}: {exc}; skipped")
            continue
        signals.append(scipy.signal.decimate(x, DECIMATION, ftype="iir", zero_phase=True).astype(np.float32))
        settings.append(match.group(1))
        meas.append(int(match.group(3)))

    lengths = np.array([len(s) for s in signals])
    n = int(lengths.min())
    short = (lengths < 0.95 * np.median(lengths)).sum()
    if short:
        print(f"  [WARN] {bearing}: {short} measurement(s) shorter than 95% of the median; all cropped to {n}")
    np.savez(os.path.join(PROCESSED_DIR, f"{bearing}.npz"), signals=np.stack([s[:n] for s in signals]),
             settings=np.array(settings), meas=np.array(meas), fs=FS)
    return len(signals), n


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for bearing in BEARINGS:
        if not os.path.isdir(os.path.join(RAW_DIR, bearing)):
            print(f"[MISSING] {bearing}: run download_paderborn.py first")
            continue
        count, length = process_bearing(bearing)
        print(f"[OK] {bearing}: {count} measurements x {length} samples @ {FS} Hz", flush=True)
