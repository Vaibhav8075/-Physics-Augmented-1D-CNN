# Leave-One-Load-Out Benchmark with Noise Sweep

Each of the 4 loads is held out in turn; models train on the other three loads with all fault sizes (0.007", 0.014", 0.021"), 15 epochs, final epoch kept. Temperature is fitted on the last 20% (in time) of each training recording. Test windows get drift (p=0.85) and spikes (p=0.35) plus colored noise at each SNR; "clean" has no impairments. Values are mean ± std over 20 runs (4 folds × 5 seeds). p-values: paired Wilcoxon signed-rank test against the full PAC-1DCNN over the same runs; "Holm" is the p-value Holm-adjusted over all model × SNR comparisons in the table.

## Macro-F1 (%)

| Model | clean | 10dB | 5dB | 0dB | -5dB | -10dB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 99.89 ± 0.16 (-0.04, p=0.017, Holm 0.299) | 99.47 ± 1.15 (-0.41, p=0.074, Holm 1.000) | 96.95 ± 6.48 (-2.43, p=0.030, Holm 0.508) | 54.90 ± 11.52 (-11.48, p=0.002, Holm 0.051) | 12.14 ± 10.33 (-3.47, p=0.368, Holm 1.000) | 7.11 ± 6.33 (-0.07, p=0.955, Holm 1.000) |
| PAC-1DCNN (full) | 99.93 ± 0.15 | 99.88 ± 0.23 | 99.38 ± 0.95 | 66.38 ± 12.53 | 15.61 ± 9.94 | 7.18 ± 6.00 |
| PAC w/o Residual Filter | 99.94 ± 0.12 (+0.01, p=0.232, Holm 1.000) | 99.91 ± 0.18 (+0.03, p=0.047, Holm 0.750) | 99.38 ± 0.98 (-0.00, p=0.532, Holm 1.000) | 66.36 ± 12.24 (-0.01, p=0.756, Holm 1.000) | 20.46 ± 10.95 (+4.85, p=0.009, Holm 0.178) | 11.01 ± 7.91 (+3.83, p=0.015, Holm 0.277) |
| PAC w/o Dual-Stream | 99.37 ± 2.02 (-0.57, p=0.213, Holm 1.000) | 99.16 ± 2.26 (-0.72, p=0.110, Holm 1.000) | 98.24 ± 4.01 (-1.14, p=0.191, Holm 1.000) | 80.37 ± 12.13 (+13.99, p=0.001, Holm 0.033) | 43.93 ± 17.32 (+28.31, p=0.000, Holm 0.000) | 15.67 ± 7.78 (+8.49, p=0.004, Holm 0.089) |
| PAC w/o Attention | 99.90 ± 0.27 (-0.03, p=0.310, Holm 1.000) | 99.80 ± 0.62 (-0.08, p=0.833, Holm 1.000) | 99.30 ± 1.49 (-0.08, p=0.594, Holm 1.000) | 64.59 ± 16.42 (-1.78, p=0.674, Holm 1.000) | 18.43 ± 12.63 (+2.82, p=0.330, Holm 1.000) | 8.90 ± 6.53 (+1.72, p=0.334, Holm 1.000) |

## Macro-F1 (%) by held-out load (clean / 0 dB)

| Model | 0 HP | 1 HP | 2 HP | 3 HP |
| :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 99.7 / 44.1 | 100.0 / 61.8 | 99.9 / 51.4 | 99.9 / 62.3 |
| PAC-1DCNN (full) | 99.7 / 56.7 | 100.0 / 71.1 | 100.0 / 64.5 | 100.0 / 73.3 |
| PAC w/o Residual Filter | 99.8 / 56.1 | 100.0 / 75.0 | 100.0 / 56.9 | 100.0 / 77.4 |
| PAC w/o Dual-Stream | 99.9 / 70.3 | 100.0 / 88.3 | 100.0 / 78.6 | 97.6 / 84.2 |
| PAC w/o Attention | 99.6 / 53.7 | 100.0 / 70.0 | 100.0 / 52.5 | 100.0 / 82.2 |

## False alarm rate and calibration at 0 dB

| Model | False alarms (%) | Missed faults (%) | ECE calibrated (%) | ECE uncalibrated (%) | Mean T | Calib-set accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 0.00 ± 0.00 | 44.75 ± 13.43 | 38.76 ± 12.66 | 31.80 | 0.316 | 99.93 |
| PAC-1DCNN (full) | 0.00 ± 0.00 | 32.64 ± 13.56 | 26.95 ± 12.55 | 20.51 | 0.347 | 100.00 |
| PAC w/o Residual Filter | 0.00 ± 0.00 | 32.31 ± 12.79 | 26.30 ± 12.17 | 19.99 | 0.367 | 100.00 |
| PAC w/o Dual-Stream | 0.00 ± 0.00 | 14.97 ± 11.20 | 12.92 ± 9.71 | 8.46 | 0.387 | 99.63 |
| PAC w/o Attention | 0.00 ± 0.00 | 35.21 ± 18.30 | 28.82 ± 17.32 | 21.45 | 0.330 | 99.99 |

## Findings (generated from the numbers above)

- **clean**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.04, p=0.017, Holm 0.299).
- **10dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.41, p=0.074, Holm 1.000).
- **5dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -2.43, p=0.030, Holm 0.508).
- **0dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -11.48, p=0.002, Holm 0.051).
- **-5dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -3.47, p=0.368, Holm 1.000).
- **-10dB**: no significant difference between baseline and PAC-1DCNN after Holm correction (baseline − PAC = -0.07, p=0.955, Holm 1.000).
- **PAC w/o Residual Filter**: no significant macro-F1 change at any noise level after Holm correction.
- **PAC w/o Dual-Stream**: significant change vs full model at 0dB: +13.99, -5dB: +28.31 (negative = component helps).
- **PAC w/o Attention**: no significant macro-F1 change at any noise level after Holm correction.
