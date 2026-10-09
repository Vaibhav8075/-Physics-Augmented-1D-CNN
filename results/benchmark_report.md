# Leave-One-Load-Out Benchmark with Noise Sweep

Each of the 4 loads is held out in turn; models train on the other three loads with all fault sizes (0.007", 0.014", 0.021"), 15 epochs, final epoch kept. Temperature is fitted on the last 20% (in time) of each training recording. Test windows get drift (p=0.85) and spikes (p=0.35) plus colored noise at each SNR; "clean" has no impairments. Values are mean ± std over 20 runs (4 folds × 5 seeds). p-values: paired Wilcoxon signed-rank test against the full PAC-1DCNN over the same runs; "Holm" is the p-value Holm-adjusted over all model × SNR comparisons in the table.

## Macro-F1 (%)

| Model | clean | 10dB | 5dB | 0dB | -5dB | -10dB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 100.00 ± 0.01 (-0.00, p=0.317, Holm 1.000) | 100.00 ± 0.00 (-0.00, p=0.317, Holm 1.000) | 99.89 ± 0.44 (-0.10, p=0.374, Holm 1.000) | 87.73 ± 9.03 (-1.91, p=0.261, Holm 1.000) | 29.59 ± 8.35 (-10.09, p=0.000, Holm 0.004) | 15.48 ± 6.22 (-8.04, p=0.001, Holm 0.019) |
| PAC-1DCNN (full) | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.99 ± 0.02 | 89.64 ± 8.77 | 39.67 ± 9.54 | 23.52 ± 5.62 |
| PAC w/o Residual Filter | 100.00 ± 0.00 (+0.00, p=0.317, Holm 1.000) | 100.00 ± 0.01 (-0.00, p=0.317, Holm 1.000) | 99.99 ± 0.05 (-0.00, p=0.753, Holm 1.000) | 86.35 ± 10.75 (-3.30, p=0.143, Holm 1.000) | 37.03 ± 12.55 (-2.64, p=0.216, Holm 1.000) | 17.88 ± 6.34 (-5.64, p=0.004, Holm 0.071) |
| PAC w/o Dual-Stream | 100.00 ± 0.01 (-0.00, p=0.655, Holm 1.000) | 100.00 ± 0.01 (-0.00, p=0.102, Holm 1.000) | 99.89 ± 0.32 (-0.10, p=0.345, Holm 1.000) | 98.97 ± 1.90 (+9.33, p=0.000, Holm 0.000) | 67.92 ± 22.06 (+28.25, p=0.000, Holm 0.001) | 12.24 ± 1.57 (-11.28, p=0.000, Holm 0.003) |
| PAC w/o Attention | 100.00 ± 0.00 (+0.00, p=0.317, Holm 1.000) | 99.99 ± 0.02 (-0.01, p=0.317, Holm 1.000) | 99.96 ± 0.15 (-0.03, p=0.500, Holm 1.000) | 87.86 ± 10.50 (-1.79, p=0.409, Holm 1.000) | 33.73 ± 10.85 (-5.94, p=0.008, Holm 0.150) | 21.95 ± 5.73 (-1.57, p=0.277, Holm 1.000) |

## Macro-F1 (%) by held-out load (clean / 0 dB)

| Model | 0 HP | 1 HP | 2 HP | 3 HP |
| :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 100.0 / 82.8 | 100.0 / 91.8 | 100.0 / 83.3 | 100.0 / 93.1 |
| PAC-1DCNN (full) | 100.0 / 83.8 | 100.0 / 91.1 | 100.0 / 86.3 | 100.0 / 97.3 |
| PAC w/o Residual Filter | 100.0 / 78.6 | 100.0 / 86.7 | 100.0 / 82.3 | 100.0 / 97.8 |
| PAC w/o Dual-Stream | 100.0 / 96.8 | 100.0 / 100.0 | 100.0 / 99.4 | 100.0 / 99.6 |
| PAC w/o Attention | 100.0 / 79.9 | 100.0 / 93.1 | 100.0 / 83.0 | 100.0 / 95.3 |

## False alarm rate and calibration at 0 dB

| Model | False alarms (%) | Missed faults (%) | ECE calibrated (%) | ECE uncalibrated (%) | Mean T | Calib-set accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 0.00 ± 0.00 | 8.18 ± 7.91 | 6.61 ± 6.52 | 5.01 | 0.341 | 100.00 |
| PAC-1DCNN (full) | 0.00 ± 0.00 | 6.94 ± 6.99 | 4.59 ± 5.78 | 4.53 | 0.404 | 100.00 |
| PAC w/o Residual Filter | 0.02 ± 0.10 | 9.15 ± 8.74 | 7.42 ± 7.71 | 5.73 | 0.402 | 100.00 |
| PAC w/o Dual-Stream | 0.00 ± 0.00 | 0.01 ± 0.04 | 0.95 ± 1.99 | 1.94 | 0.357 | 100.00 |
| PAC w/o Attention | 0.00 ± 0.00 | 8.34 ± 9.10 | 6.03 ± 7.84 | 5.58 | 0.398 | 100.00 |

## Findings (generated from the numbers above)

- **clean**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.00, p=0.317, Holm 1.000).
- **10dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.00, p=0.317, Holm 1.000).
- **5dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.10, p=0.374, Holm 1.000).
- **0dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -1.91, p=0.261, Holm 1.000).
- **-5dB**: PAC-1DCNN is significantly better after Holm correction (baseline − PAC = -10.09 macro-F1, p=0.000, Holm 0.004).
- **-10dB**: PAC-1DCNN is significantly better after Holm correction (baseline − PAC = -8.04 macro-F1, p=0.001, Holm 0.019).
- **PAC w/o Residual Filter**: no significant macro-F1 change at any noise level after Holm correction.
- **PAC w/o Dual-Stream**: significant change vs full model at 0dB: +9.33, -5dB: +28.25, -10dB: -11.28 (negative = component helps).
- **PAC w/o Attention**: no significant macro-F1 change at any noise level after Holm correction.
