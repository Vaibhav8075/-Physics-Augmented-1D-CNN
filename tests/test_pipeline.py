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
