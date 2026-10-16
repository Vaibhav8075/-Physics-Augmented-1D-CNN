# Leave-One-Load-Out Benchmark with Noise Sweep

Each of the 4 loads is held out in turn; models train on the other three loads with all fault sizes (0.007", 0.014", 0.021"), 15 epochs, final epoch kept. Temperature is fitted on the last 20% (in time) of each training recording. Test windows get drift (p=0.85) and spikes (p=0.35) plus colored noise at each SNR; "clean" has no impairments. Values are mean ± std over 20 runs (4 folds × 5 seeds). p-values: paired Wilcoxon signed-rank test against the full PAC-1DCNN over the same runs; "Holm" is the p-value Holm-adjusted over all model × SNR comparisons in the table.

## Macro-F1 (%)

| Model | clean | 10dB | 5dB | 0dB | -5dB | -10dB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 99.86 ± 0.20 (-0.00, p=0.333, Holm 1.000) | 99.69 ± 0.61 (-0.06, p=0.575, Holm 1.000) | 98.42 ± 3.27 (-0.76, p=0.075, Holm 1.000) | 62.79 ± 12.61 (-11.97, p=0.000, Holm 0.000) | 12.95 ± 9.06 (-4.31, p=0.349, Holm 1.000) | 10.69 ± 7.66 (-0.31, p=0.879, Holm 1.000) |
| PAC-1DCNN (full) | 99.86 ± 0.29 | 99.75 ± 0.56 | 99.18 ± 1.73 | 74.76 ± 12.19 | 17.25 ± 12.57 | 11.00 ± 8.58 |
| PAC w/o Residual Filter | 99.86 ± 0.37 (+0.00, p=0.314, Holm 1.000) | 99.76 ± 0.68 (+0.02, p=0.286, Holm 1.000) | 99.15 ± 1.81 (-0.02, p=0.754, Holm 1.000) | 70.66 ± 14.31 (-4.10, p=0.261, Holm 1.000) | 23.23 ± 14.20 (+5.97, p=0.008, Holm 0.166) | 13.92 ± 9.12 (+2.91, p=0.053, Holm 1.000) |
| PAC w/o Dual-Stream | 99.94 ± 0.10 (+0.08, p=0.314, Holm 1.000) | 99.95 ± 0.09 (+0.20, p=0.093, Holm 1.000) | 99.43 ± 1.20 (+0.25, p=0.347, Holm 1.000) | 89.16 ± 9.01 (+14.40, p=0.000, Holm 0.000) | 41.86 ± 16.23 (+24.60, p=0.000, Holm 0.000) | 12.46 ± 1.75 (+1.46, p=0.478, Holm 1.000) |
| PAC w/o Attention | 99.80 ± 0.49 (-0.06, p=0.398, Holm 1.000) | 99.64 ± 0.94 (-0.11, p=0.594, Holm 1.000) | 98.90 ± 2.30 (-0.27, p=0.110, Holm 1.000) | 70.47 ± 13.48 (-4.29, p=0.004, Holm 0.077) | 19.43 ± 8.93 (+2.18, p=0.355, Holm 1.000) | 13.25 ± 7.82 (+2.24, p=0.492, Holm 1.000) |

## Macro-F1 (%) by held-out load (clean / 0 dB)

| Model | 0 HP | 1 HP | 2 HP | 3 HP |
| :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 99.5 / 48.8 | 100.0 / 67.6 | 100.0 / 62.9 | 99.9 / 71.8 |
| PAC-1DCNN (full) | 99.5 / 60.7 | 100.0 / 82.7 | 100.0 / 71.2 | 99.9 / 84.4 |
| PAC w/o Residual Filter | 99.5 / 57.7 | 100.0 / 71.4 | 100.0 / 72.6 | 100.0 / 80.9 |
| PAC w/o Dual-Stream | 100.0 / 75.7 | 100.0 / 94.7 | 100.0 / 92.0 | 99.8 / 94.3 |
| PAC w/o Attention | 99.2 / 53.6 | 100.0 / 78.2 | 100.0 / 68.4 | 100.0 / 81.7 |

## False alarm rate and calibration at 0 dB

| Model | False alarms (%) | Missed faults (%) | ECE calibrated (%) | ECE uncalibrated (%) | Mean T | Calib-set accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 0.00 ± 0.00 | 37.24 ± 12.90 | 31.46 ± 13.50 | 23.71 | 0.300 | 99.96 |
| PAC-1DCNN (full) | 0.00 ± 0.00 | 23.28 ± 12.29 | 19.40 ± 11.64 | 13.43 | 0.339 | 100.00 |
| PAC w/o Residual Filter | 0.00 ± 0.00 | 28.70 ± 15.39 | 23.53 ± 14.52 | 17.34 | 0.322 | 100.00 |
| PAC w/o Dual-Stream | 0.03 ± 0.10 | 8.15 ± 7.76 | 6.64 ± 6.61 | 4.40 | 0.354 | 100.00 |
| PAC w/o Attention | 0.00 ± 0.00 | 29.02 ± 14.20 | 23.16 ± 13.35 | 16.20 | 0.333 | 99.99 |

## Findings (generated from the numbers above)

- **clean**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.00, p=0.333, Holm 1.000).
- **10dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.06, p=0.575, Holm 1.000).
- **5dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.76, p=0.075, Holm 1.000).
- **0dB**: PAC-1DCNN is significantly better after Holm correction (baseline − PAC = -11.97 macro-F1, p=0.000, Holm 0.000).
- **-5dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -4.31, p=0.349, Holm 1.000).
- **-10dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.31, p=0.879, Holm 1.000).
- **PAC w/o Residual Filter**: no significant macro-F1 change at any noise level after Holm correction.
- **PAC w/o Dual-Stream**: significant change vs full model at 0dB: +14.40, -5dB: +24.60 (negative = component helps).
- **PAC w/o Attention**: no significant macro-F1 change at any noise level after Holm correction.
