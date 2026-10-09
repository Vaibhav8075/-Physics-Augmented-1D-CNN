import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from common import add_real_world_impairments, compute_ece, impair_array
from models import Baseline1DCNN, PhysicsAugmentedCalibratedCNN
from preprocess_data import parse_filename

HAS_DATA = os.path.exists(os.path.join(config.PROCESSED_DIR, "X_train_clean.npy"))
HAS_RAW = os.path.isdir(config.DATA_DIR) and len(os.listdir(config.DATA_DIR)) > 0


@pytest.mark.parametrize("kw", [{}, {"use_residual_filter": False}, {"use_dual_stream": False},
                                {"use_attention": False}])
def test_model_output_shapes(kw):
    logits, attn, _ = PhysicsAugmentedCalibratedCNN(**kw)(torch.randn(3, 2, config.WINDOW_SIZE))
    assert logits.shape == (3, 4)


def test_temperature_does_not_change_predictions():
    model = PhysicsAugmentedCalibratedCNN().eval()
    x = torch.randn(8, 2, config.WINDOW_SIZE)
    with torch.no_grad():
        raw = model(x, return_calibrated=False)[0]
        model.temperature.fill_(3.7)
        cal = model(x, return_calibrated=True)[0]
    assert torch.equal(raw.argmax(1), cal.argmax(1))
    assert torch.allclose(cal, raw / 3.7)


def test_temperature_is_not_trainable():
    names = [n for n, _ in PhysicsAugmentedCalibratedCNN().named_parameters()]
    assert "temperature" not in names
    assert "temperature" not in [n for n, _ in Baseline1DCNN().named_parameters()]


def test_ece_perfect_and_overconfident():
    labels = np.array([0, 1, 2, 3])
    assert compute_ece(np.eye(4), labels) == pytest.approx(0.0)
    wrong = np.eye(4)[[1, 2, 3, 0]]  # 100% confident, always wrong
    assert compute_ece(wrong, labels) == pytest.approx(100.0)


def test_impairments_are_reproducible():
    x = np.random.default_rng(0).standard_normal((5, 2, config.WINDOW_SIZE)).astype(np.float32)
    assert np.array_equal(impair_array(x, "test", 42), impair_array(x, "test", 42))
    assert not np.array_equal(impair_array(x, "test", 42), impair_array(x, "test", 43))


def test_impairment_snr():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, config.WINDOW_SIZE)).astype(np.float32)
    noisy = add_real_world_impairments(x, snr_db=10.0, drift_prob=0.0, impulse_prob=0.0, rng=rng)
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean((noisy - x) ** 2))
    assert snr == pytest.approx(10.0, abs=0.1)


@pytest.mark.parametrize("name,label,load", [("Normal_2.mat", 0, 2), ("IR007_1.mat", 1, 1),
                                             ("B007_3.mat", 2, 3), ("OR007_6_0.mat", 3, 0)])
def test_parse_filename(name, label, load):
    assert parse_filename(name) == (label, load)


@pytest.mark.skipif(not HAS_RAW, reason="raw CWRU data not downloaded")
def test_each_file_uses_its_own_recording():
    """Regression: 99.mat also contains X098_* (the 1 HP training recording)."""
    from preprocess_data import load_recording
    normal_1 = load_recording(os.path.join(config.DATA_DIR, "Normal_1.mat"))
    normal_2 = load_recording(os.path.join(config.DATA_DIR, "Normal_2.mat"))
    n = min(normal_1.shape[1], normal_2.shape[1])
    assert not np.allclose(normal_1[:, :n], normal_2[:, :n])


@pytest.mark.skipif(not HAS_DATA, reason="run preprocess_data.py first")
def test_no_window_shared_between_splits():
    def hashes(split):
        X = np.load(os.path.join(config.PROCESSED_DIR, f"X_{split}_clean.npy"))
        return {w.tobytes() for w in X}
    train, val, test = hashes("train"), hashes("val"), hashes("test")
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_batched_impairments_match_numpy_distribution():
    """GPU training noise (impair_batch) must match the numpy val/test noise in distribution."""
    from common import impair_batch
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 2, config.WINDOW_SIZE)).astype(np.float32)
    params = config.IMPAIRMENTS["test"]

    ref = np.stack([add_real_world_impairments(x, rng=rng, **params) for x in X]) - X
    g = torch.Generator().manual_seed(0)
    out = impair_batch(torch.from_numpy(X), generator=g, **params).numpy() - X

    for name, stat in [("power", lambda d: np.mean(d ** 2)),
                       ("abs-max", lambda d: np.mean(np.abs(d).max(axis=-1))),
                       ("low-freq", lambda d: np.mean(d.mean(axis=-1) ** 2))]:
        assert stat(out) == pytest.approx(stat(ref), rel=0.1), name


def test_batched_impairment_snr_without_drift_or_spikes():
    from common import impair_batch
    x = torch.randn(64, 2, config.WINDOW_SIZE)
    noisy = impair_batch(x, snr_db=0.0, drift_prob=0.0, impulse_prob=0.0)
    snr = 10 * torch.log10(x.pow(2).mean((1, 2)) / (noisy - x).pow(2).mean((1, 2)))
    assert torch.allclose(snr, torch.zeros_like(snr), atol=0.05)


@pytest.mark.skipif(not os.path.exists(os.path.join(config.PROCESSED_DIR, "load0.npz")),
                    reason="run preprocess_data.py first")
def test_benchmark_calibration_split_does_not_overlap_training():
    """Calibration windows come from a later time span than training windows, with no shared samples."""
    data = dict(np.load(os.path.join(config.PROCESSED_DIR, "load1.npz")))
    gap_windows = config.WINDOW_SIZE // config.STRIDE  # windows this far apart share no samples
    cut = 1.0 - config.CALIB_FRACTION
    for rec in np.unique(data["rec"]):
        pos = data["pos"][data["rec"] == rec]
        idx = np.round(pos * len(pos)).astype(int)
        train_idx, calib_idx = idx[pos < cut - 0.01], idx[pos >= cut]
        assert calib_idx.min() - train_idx.max() >= gap_windows, rec


@pytest.mark.parametrize("spectrum", ["lowpass", "white", "highpass", "mixed"])
def test_batched_impairment_snr_for_every_spectrum(spectrum):
    from common import impair_batch
    x = torch.randn(64, 2, config.WINDOW_SIZE)
    noisy = impair_batch(x, snr_db=0.0, drift_prob=0.0, impulse_prob=0.0, spectrum=spectrum)
    snr = 10 * torch.log10(x.pow(2).mean((1, 2)) / (noisy - x).pow(2).mean((1, 2)))
    assert torch.allclose(snr, torch.zeros_like(snr), atol=0.05)


def test_noise_spectra_differ_in_high_frequency_share():
    """Share of noise power above 1.5 kHz: lowpass < white < highpass (numpy and torch agree)."""
    from common import impair_batch
    freqs = np.fft.rfftfreq(config.WINDOW_SIZE, d=1.0 / config.SAMPLING_RATE_HZ)
    def hf_share(d):
        p = np.abs(np.fft.rfft(d, axis=-1)) ** 2
        return p[..., freqs > 1500].sum() / p.sum()
    x = np.zeros((200, 2, config.WINDOW_SIZE), dtype=np.float32) + 1.0  # constant signal: noise isolated below
    shares = {}
    for s in ["lowpass", "white", "highpass"]:
        rng = np.random.default_rng(0)
        ref = np.stack([add_real_world_impairments(w, snr_db=0.0, drift_prob=0.0, impulse_prob=0.0,
                                                   rng=rng, spectrum=s) for w in x]) - x
        out = impair_batch(torch.from_numpy(x), 0.0, 0.0, 0.0, spectrum=s).numpy() - x
        shares[s] = hf_share(ref)
        assert hf_share(out) == pytest.approx(shares[s], abs=0.03), s
    assert shares["lowpass"] < shares["white"] < shares["highpass"]


@pytest.mark.skipif(not os.path.exists(os.path.join(config.PROCESSED_DIR, "load0.npz")),
                    reason="run preprocess_data.py first")
def test_fault_size_folds_hold_out_whole_bearings():
    from run_robustness_study import load_all, fold_masks, FAULT_SIZES
    data = load_all()
    for held_out in FAULT_SIZES:
        train, test = fold_masks(data, held_out)
        assert not (train & test).any()
        assert set(np.unique(data["size"][test])) == {0, held_out}
        assert held_out not in set(np.unique(data["size"][train]))
        assert set(np.unique(data["y"][test])) == {0, 1, 2, 3}


def test_spectral_features_unchanged_for_cwru_windows():
    """Generalised RF features must equal the original 1024-sample / 12 kHz implementation."""
    import scipy.signal
    from run_robustness_study import spectral_features
    X = np.random.default_rng(0).standard_normal((5, 2, 1024)).astype(np.float32)
    win = np.hanning(1024).astype(np.float32)
    spec = np.abs(np.fft.rfft(X * win, axis=-1))[..., :512]
    bands = np.log1p(spec.reshape(5, 2, 64, 8).mean(-1))
    env = np.abs(scipy.signal.hilbert(X, axis=-1))
    env -= env.mean(-1, keepdims=True)
    env_spec = np.log1p(np.abs(np.fft.rfft(env * win, axis=-1))[..., 1:53])
    expected = np.concatenate([bands, env_spec], axis=-1).reshape(5, -1)
    np.testing.assert_allclose(spectral_features(X), expected, rtol=1e-6)


def test_models_accept_paderborn_windows():
    from models import WideKernelCNN, PhysicsAugmentedCalibratedCNN
    x = torch.randn(4, 1, 2048)
    for model in (Baseline1DCNN(1, 3), Baseline1DCNN(1, 3, highpass_input=True), WideKernelCNN(1, 3),
                  PhysicsAugmentedCalibratedCNN(1, 3), PhysicsAugmentedCalibratedCNN(1, 3, use_residual_filter=False)):
        assert model(x)[0].shape == (4, 3)
    assert WideKernelCNN()(torch.randn(4, 2, 1024))[0].shape == (4, 4)


def test_paderborn_folds_separate_physical_bearings():
    from run_paderborn_study import folds_for, label_of
    cv = folds_for("real_cv")
    assert len(cv) == 10 and len({f[0] for f in cv}) == 10
    for _, train, test in cv + folds_for("a2r"):
        assert not set(train) & set(test)
        assert {label_of(b) for b in train} == {label_of(b) for b in test} == {0, 1, 2}
    for _, train, test in cv:
        assert np.bincount([label_of(b) for b in train]).tolist() == [3, 3, 3]
        assert np.bincount([label_of(b) for b in test]).tolist() == [2, 2, 2]


def test_measurement_accuracy_averages_window_probabilities():
    from run_paderborn_study import measurement_accuracy
    probs = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.4, 0.6, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0, 0, 1, 2])
    meas = np.array(["a", "a", "b", "c"])
    assert measurement_accuracy(probs, y, meas) == pytest.approx(100.0)  # a: mean [0.55, 0.45] -> 0
