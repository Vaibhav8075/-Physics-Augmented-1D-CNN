"""Single source of truth for experiment settings shared by every script."""

DATA_DIR = "cwru_fault_data"
PROCESSED_DIR = "processed_data"
RESULTS_DIR = "results"

CLASS_NAMES = ["Normal (Healthy)", "Inner Race Fault", "Ball Fault", "Outer Race Fault"]

SAMPLING_RATE_HZ = 12000
WINDOW_SIZE = 1024
STRIDE = 256

# Fault sizes (mils) used by the original fixed split (train_and_evaluate.py,
# run_ablation_study.py). The cross-load benchmark (run_benchmark.py) uses all sizes.
LEGACY_FAULT_SIZES = [7]
ALL_LOADS = [0, 1, 2, 3]

# Cross-load split protocol
TRAIN_LOADS = [0, 1]
VAL_LOADS = [2]
TEST_LOADS = [3]

# Industrial impairment levels per split (train noise is re-sampled every epoch)
IMPAIRMENTS = {
    "train": {"snr_db": 15.0, "drift_prob": 0.4, "impulse_prob": 0.1},
    "val": {"snr_db": 10.0, "drift_prob": 0.6, "impulse_prob": 0.2},
    "test": {"snr_db": 8.0, "drift_prob": 0.85, "impulse_prob": 0.35},
}

# Fixed seeds so val/test impairments are identical across runs and models
VAL_NOISE_SEED = 1002
TEST_NOISE_SEED = 1003

EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-3
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# Leave-one-load-out benchmark (run_benchmark.py)
BENCHMARK_SNR_DB = [None, 10.0, 5.0, 0.0, -5.0, -10.0]  # None = clean (no impairments)
BENCHMARK_NOISE_SEED = 2000
# The last CALIB_FRACTION of every training recording (in time) is held out for
# temperature calibration; one window-length gap prevents overlap with training windows.
CALIB_FRACTION = 0.2
