"""
Bearing-kinematics features for the "Kin-RF" baseline.

A localised defect produces impacts at a rate fixed by the bearing geometry and the shaft speed
(e.g. Randall & Antoni 2011). These features measure, per channel, how strongly the envelope
spectrum peaks at the first three harmonics of each fault frequency, relative to the envelope
spectrum's median level. Unlike the 64-band / envelope-bin features of Env-RF, they look only at
frequencies that the physics of the bearing predicts, so they cannot describe other properties
of an individual recording.

Fault frequencies as multiples of the shaft frequency:
  CWRU drive-end bearing 6205-2RS JEM SKF: inner ring 5.4152, outer ring 3.5848, rolling element
  4.7135 (CWRU Bearing Data Center, "Bearing Information" page; the rolling-element value is
  twice the ball spin frequency).
  Paderborn bearing 6203: computed from the geometry in Lessmeier et al. (2016), Table 1
  (pitch circle diameter 28.55 mm, 8 rolling elements of 6.75 mm, nominal pressure angle 0).
"""
import numpy as np
import scipy.signal

CWRU_DE_ORDERS = {"inner": 5.4152, "outer": 3.5848, "ball": 4.7135}


def ball_bearing_orders(pitch_mm, ball_mm, n_balls, contact_deg=0.0):
    """Fault frequencies / shaft frequency for a stationary outer ring: BPFI, BPFO, 2 x BSF."""
    r = ball_mm / pitch_mm * np.cos(np.deg2rad(contact_deg))
    return {"inner": n_balls / 2 * (1 + r),
            "outer": n_balls / 2 * (1 - r),
            "ball": pitch_mm / ball_mm * (1 - r ** 2)}  # 2 x BSF: a ball defect strikes both races


PADERBORN_ORDERS = ball_bearing_orders(28.55, 6.75, 8)
HARMONICS = 3
TOLERANCE = 0.03  # +-3 % around each target frequency (slip, speed variation), at least one FFT bin


def kinematic_features(X, shaft_hz, orders, fs, band=None, pad=8):
    """
    X: (N, C, L) windows; shaft_hz: scalar or (N,) shaft frequency per window.
    The signal is band-passed (default fs/12 to fs/3, a fixed untuned band in the upper part of
    the spectrum, where impact-excited resonances dominate), its Hilbert envelope is taken
    (mean removed, Hann window, zero-padded by `pad` for peak interpolation), and for every fault
    order and harmonic h the peak magnitude within +-TOLERANCE of h * order * shaft_hz is divided
    by the median envelope-spectrum magnitude below 1.2 x the highest target frequency.
    Returns (N, C * len(orders) * HARMONICS) log ratios. Windows are processed in chunks of
    CHUNK so that memory stays bounded (a zero-padded spectrum of every window at once would
    need several GB for a CWRU training set).
    """
    N = len(X)
    shaft = np.broadcast_to(np.asarray(shaft_hz, dtype=np.float64), (N,))
    out = np.empty((N, X.shape[1] * len(orders) * HARMONICS), dtype=np.float32)
    for s in range(0, N, CHUNK):
        out[s:s + CHUNK] = _chunk_features(X[s:s + CHUNK], shaft[s:s + CHUNK], orders, fs, band, pad)
    return out


CHUNK = 256


def _chunk_features(X, shaft, orders, fs, band, pad):
    N, C, L = X.shape
    lo, hi = band or (fs / 12.0, fs / 3.0)
    sos = scipy.signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xb = scipy.signal.sosfiltfilt(sos, X.astype(np.float64), axis=-1)
    env = np.abs(scipy.signal.hilbert(xb, axis=-1))
    env -= env.mean(-1, keepdims=True)
    spec = np.abs(np.fft.rfft(env * np.hanning(L), n=pad * L, axis=-1))
    freqs = np.fft.rfftfreq(pad * L, 1.0 / fs)
    half_bin = fs / L  # one native bin on either side at least

    names = list(orders)
    feats = np.empty((N, C, len(names), HARMONICS))
    for i in range(N):
        targets = {(k, h): (h + 1) * orders[k] * shaft[i] for k in names for h in range(HARMONICS)}
        floor_mask = (freqs > 0) & (freqs < 1.2 * max(targets.values()))
        floor = np.median(spec[i][:, floor_mask], axis=-1) + 1e-12
        for (k, h), f0 in targets.items():
            w = max(TOLERANCE * f0, half_bin)
            mask = (freqs >= f0 - w) & (freqs <= f0 + w)
            feats[i, :, names.index(k), h] = np.log(spec[i][:, mask].max(-1) / floor)
    return feats.reshape(N, -1).astype(np.float32)
