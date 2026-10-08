import os
import re
import glob
import numpy as np
import scipy.io as sio

import config
from common import add_real_world_impairments, impair_array  # noqa: F401  (re-exported for older imports)
from download_data import recording_id

LABEL_PREFIXES = {"Normal": 0, "IR": 1, "B": 2, "OR": 3}


def parse_filename(filename):
    """'OR007_6_3.mat' -> (label=3, load=3). Matches the fault-type prefix exactly."""
    stem = os.path.splitext(filename)[0]
    match = re.match(r"^(Normal|IR|B|OR)", stem)
    if not match:
        return None, None
    return LABEL_PREFIXES[match.group(1)], int(stem.split("_")[-1])


def load_recording(filepath):
    """
    Loads the DE/FE channels belonging to this file's own CWRU recording.
    Some files (e.g. 99.mat) also contain channels from a neighbouring recording
    (X098_*), so the key is matched by recording ID, not by position.
    """
    mat = sio.loadmat(filepath)
    rid = recording_id(os.path.basename(filepath))
    de_key, fe_key = f"X{rid}_DE_time", f"X{rid}_FE_time"
    if de_key not in mat:
        raise KeyError(f"{filepath}: expected key {de_key}, found {[k for k in mat if 'time' in k]}")
    de = mat[de_key].ravel()
    fe = mat[fe_key].ravel() if fe_key in mat else de
    n = min(len(de), len(fe))
    return np.stack([de[:n], fe[:n]]).astype(np.float32)


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
            # Train impairments are drawn fresh every epoch (common.OnlineImpairedDataset);
            # val/test impairments are fixed so every model is scored on identical inputs.
            seed = config.VAL_NOISE_SEED if split == "val" else config.TEST_NOISE_SEED
            save(f"X_{split}.npy", impair_array(X, split, seed))
        print(f"  • {split:<5} loads {[l for l, s in split_of_load.items() if s == split]}: "
              f"{len(y)} windows, class counts {np.bincount(y, minlength=4).tolist()}")

    stale = os.path.join(config.PROCESSED_DIR, "X_train.npy")
    if os.path.exists(stale):
        os.remove(stale)  # old pre-impaired training set is no longer used


if __name__ == "__main__":
    create_real_world_dataset()
