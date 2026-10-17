# Leave-One-Load-Out Benchmark with Noise Sweep

Each of the 4 loads is held out in turn; models train on the other three loads with all fault sizes (0.007", 0.014", 0.021"), 15 epochs, final epoch kept. Temperature is fitted on the last 20% (in time) of each training recording. Test windows get drift (p=0.85) and spikes (p=0.35) plus colored noise at each SNR; "clean" has no impairments. Values are mean ± std over 20 runs (4 folds × 5 seeds). p-values: paired Wilcoxon signed-rank test against the full PAC-1DCNN over the same runs; "Holm" is the p-value Holm-adjusted over all model × SNR comparisons in the table.

## Macro-F1 (%)

| Model | clean | 10dB | 5dB | 0dB | -5dB | -10dB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 99.85 ± 0.20 (-0.04, p=0.139, Holm 1.000) | 99.69 ± 0.55 (-0.11, p=0.110, Holm 1.000) | 97.69 ± 2.92 (-1.64, p=0.001, Holm 0.023) | 56.43 ± 9.08 (-13.80, p=0.000, Holm 0.000) | 11.42 ± 6.51 (-6.54, p=0.033, Holm 0.593) | 8.59 ± 5.88 (-1.52, p=0.356, Holm 1.000) |
| PAC-1DCNN (full) | 99.90 ± 0.26 | 99.79 ± 0.59 | 99.33 ± 1.50 | 70.23 ± 9.74 | 17.96 ± 11.05 | 10.11 ± 7.18 |
| PAC w/o Residual Filter | 99.89 ± 0.24 (-0.01, p=0.508, Holm 1.000) | 99.82 ± 0.40 (+0.02, p=0.953, Holm 1.000) | 99.21 ± 1.34 (-0.12, p=0.191, Holm 1.000) | 66.25 ± 12.19 (-3.98, p=0.027, Holm 0.506) | 19.59 ± 12.35 (+1.63, p=0.701, Holm 1.000) | 10.45 ± 6.69 (+0.33, p=0.796, Holm 1.000) |
| PAC w/o Dual-Stream | 99.81 ± 0.40 (-0.09, p=0.445, Holm 1.000) | 99.78 ± 0.41 (-0.01, p=0.937, Holm 1.000) | 98.44 ± 2.54 (-0.88, p=0.053, Holm 0.895) | 80.15 ± 8.36 (+9.92, p=0.000, Holm 0.004) | 46.28 ± 17.81 (+28.32, p=0.000, Holm 0.000) | 12.80 ± 3.16 (+2.68, p=0.126, Holm 1.000) |
| PAC w/o Attention | 99.88 ± 0.22 (-0.01, p=0.109, Holm 1.000) | 99.80 ± 0.38 (+0.00, p=0.139, Holm 1.000) | 99.19 ± 1.43 (-0.14, p=0.149, Holm 1.000) | 65.66 ± 11.64 (-4.57, p=0.004, Holm 0.084) | 16.22 ± 10.22 (-1.75, p=1.000, Holm 1.000) | 8.61 ± 6.80 (-1.50, p=0.523, Holm 1.000) |

## Macro-F1 (%) by held-out load (clean / 0 dB)

| Model | 0 HP | 1 HP | 2 HP | 3 HP |
| :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 99.6 / 48.0 | 100.0 / 61.4 | 100.0 / 52.8 | 99.8 / 63.5 |
| PAC-1DCNN (full) | 99.6 / 59.8 | 100.0 / 76.5 | 100.0 / 65.7 | 100.0 / 78.9 |
| PAC w/o Residual Filter | 99.6 / 59.1 | 100.0 / 69.7 | 100.0 / 57.9 | 99.9 / 78.2 |
| PAC w/o Dual-Stream | 99.9 / 71.9 | 100.0 / 86.5 | 100.0 / 77.0 | 99.3 / 85.2 |
| PAC w/o Attention | 99.6 / 53.4 | 100.0 / 71.3 | 100.0 / 61.9 | 99.9 / 76.1 |

## False alarm rate and calibration at 0 dB

| Model | False alarms (%) | Missed faults (%) | ECE calibrated (%) | ECE uncalibrated (%) | Mean T | Calib-set accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 0.00 ± 0.00 | 44.77 ± 10.17 | 38.21 ± 9.70 | 31.18 | 0.288 | 100.00 |
| PAC-1DCNN (full) | 0.00 ± 0.00 | 29.63 ± 10.78 | 23.97 ± 10.32 | 18.02 | 0.367 | 100.00 |
| PAC w/o Residual Filter | 0.00 ± 0.00 | 32.86 ± 13.85 | 27.57 ± 12.54 | 21.59 | 0.349 | 100.00 |
| PAC w/o Dual-Stream | 0.00 ± 0.00 | 17.07 ± 8.55 | 14.46 ± 7.83 | 9.94 | 0.353 | 100.00 |
| PAC w/o Attention | 0.00 ± 0.00 | 34.72 ± 12.72 | 28.64 ± 12.10 | 21.55 | 0.319 | 100.00 |

## Findings (generated from the numbers above)

- **clean**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.04, p=0.139, Holm 1.000).
- **10dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.11, p=0.110, Holm 1.000).
- **5dB**: PAC-1DCNN is significantly better after Holm correction (baseline − PAC = -1.64 macro-F1, p=0.001, Holm 0.023).
- **0dB**: PAC-1DCNN is significantly better after Holm correction (baseline − PAC = -13.80 macro-F1, p=0.000, Holm 0.000).
- **-5dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -6.54, p=0.033, Holm 0.593).
- **-10dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -1.52, p=0.356, Holm 1.000).
- **PAC w/o Residual Filter**: no significant macro-F1 change at any noise level after Holm correction.
- **PAC w/o Dual-Stream**: significant change vs full model at 0dB: +9.92, -5dB: +28.32 (negative = component helps).
- **PAC w/o Attention**: no significant macro-F1 change at any noise level after Holm correction.
