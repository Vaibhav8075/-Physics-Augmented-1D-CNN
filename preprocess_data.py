import os
import glob
import numpy as np
import scipy.io as sio

DATA_DIR = "cwru_fault_data"
PROCESSED_DIR = "processed_data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

WINDOW_SIZE = 1024
STRIDE = 256 # More realistic overlapping window extraction

def add_real_world_impairments(signal, snr_db=8.0, drift_prob=0.8, impulse_prob=0.3):
    """
    Simulates real-world industrial factory environment:
    1. Realistic Factory Background Noise (Colored/Pink Noise)
    2. Sensor Thermal/Offset Baseline Drift (Non-linear low frequency drift)
    3. Random Electromagnetic / Mechanical Shock Spikes (Impulse Noise)
    """
    augmented = signal.copy()
    L = signal.shape[-1]
    
    # 1. Realistic Factory Noise (Pink Noise simulation via cumulative sum filter)
    white = np.random.randn(*signal.shape).astype(np.float32)
    # 1st-order IIR filter to create 1/f industrial background noise
    colored_noise = np.zeros_like(white)
    for c in range(signal.shape[0]):
        colored_noise[c] = np.convolve(white[c], np.exp(-np.linspace(0, 3, 15)), mode='same')
        
    signal_power = np.mean(signal ** 2) + 1e-8
    noise_power = np.mean(colored_noise ** 2) + 1e-8
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    colored_noise = colored_noise * np.sqrt(target_noise_power / noise_power)
    augmented += colored_noise
    
    # 2. Non-linear Sensor Baseline Drift (Thermal & Load Fluctuation)
    if np.random.rand() < drift_prob:
        t = np.linspace(0, 2 * np.pi * np.random.uniform(0.5, 2.0), L)
        drift = (np.sin(t) * np.random.uniform(0.3, 0.8) + np.linspace(0, np.random.uniform(-0.5, 0.5), L)).astype(np.float32)
        augmented += drift
        
    # 3. Transient Electromagnetic Spikes (Adjacent heavy machinery turning ON/OFF)
    if np.random.rand() < impulse_prob:
        spike_locs = np.random.choice(L, size=np.random.randint(2, 6), replace=False)
        for loc in spike_locs:
            spike_val = np.random.choice([-1, 1]) * np.random.uniform(2.0, 4.0)
            augmented[:, loc:min(loc+3, L)] += spike_val
            
    return augmented

def create_real_world_dataset():
    print("Generating Real-World Industrial Dataset with Variance & Cross-Load Domain Shift...")
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    files = glob.glob(os.path.join(DATA_DIR, "*.mat"))

    for filepath in files:
        filename = os.path.basename(filepath)
        mat = sio.loadmat(filepath)
        
        # Determine label
        if "Normal" in filename:
            label = 0
        elif "IR" in filename:
            label = 1
        elif "B" in filename:
            label = 2
        elif "OR" in filename:
            label = 3
        else:
            continue
            
        load = int(filename.split(".")[0].split("_")[-1])
        
        de_key = [k for k in mat.keys() if "DE_time" in k]
        fe_key = [k for k in mat.keys() if "FE_time" in k]
        
        if not de_key:
            continue
            
        de_signal = mat[de_key[0]].flatten()
        if fe_key:
            fe_signal = mat[fe_key[0]].flatten()
            min_len = min(len(de_signal), len(fe_signal))
            multichannel_signal = np.stack([de_signal[:min_len], fe_signal[:min_len]], axis=0)
        else:
            multichannel_signal = np.stack([de_signal, de_signal], axis=0)
            
        total_len = multichannel_signal.shape[1]
        
        # Sliding window slicing
        for start in range(0, total_len - WINDOW_SIZE + 1, STRIDE):
            window = multichannel_signal[:, start:start + WINDOW_SIZE]
            
            # Normalize
            mean = np.mean(window, axis=1, keepdims=True)
            std = np.std(window, axis=1, keepdims=True) + 1e-8
            norm_window = (window - mean) / std
            
            # ==============================================================
            # REAL-WORLD SPLIT PROTOCOL (Cross-Load Domain Shift):
            # Train on Loads 0 & 1 (Low/Medium Load: 0-1 HP @ 1797-1772 RPM)
            # Validation on Load 2 (Intermediate Load: 2 HP @ 1750 RPM)
            # Test on UNSEEN Load 3 (High Load: 3 HP @ 1730 RPM + Industrial Impairments)
            # ==============================================================
            if load in [0, 1]:
                # Train samples (with mild operational noise)
                train_sample = add_real_world_impairments(norm_window, snr_db=15.0, drift_prob=0.4, impulse_prob=0.1)
                X_train_list.append(train_sample)
                y_train_list.append(label)
            elif load == 2:
                # Validation samples
                val_sample = add_real_world_impairments(norm_window, snr_db=10.0, drift_prob=0.6, impulse_prob=0.2)
                X_val_list.append(val_sample)
                y_val_list.append(label)
            elif load == 3:
                # Hard Unseen Test samples (Severe industrial plant noise & drift)
                test_sample = add_real_world_impairments(norm_window, snr_db=6.0, drift_prob=0.85, impulse_prob=0.35)
                X_test_list.append(test_sample)
                y_test_list.append(label)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int64)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    # Shuffle train set
    train_indices = np.random.permutation(len(X_train))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print(f"\nReal-World Dataset Generated:")
    print(f"  • Training Samples (Loads 0 & 1 HP):   {X_train.shape[0]}")
    print(f"  • Validation Samples (Load 2 HP):     {X_val.shape[0]}")
    print(f"  • Unseen Test Samples (Load 3 HP + Harsh Factory Variance): {X_test.shape[0]}")

if __name__ == "__main__":
    create_real_world_dataset()
