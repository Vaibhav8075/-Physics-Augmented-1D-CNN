# Leave-One-Load-Out Benchmark with Noise Sweep

Each of the 4 loads is held out in turn; models train on the other three loads with all fault sizes (0.007", 0.014", 0.021"), 15 epochs, final epoch kept. Temperature is fitted on the last 20% (in time) of each training recording. Test windows get drift (p=0.85) and spikes (p=0.35) plus colored noise at each SNR; "clean" has no impairments. Values are mean ± std over 20 runs (4 folds × 5 seeds). p-values: paired Wilcoxon signed-rank test against the full PAC-1DCNN over the same runs.

## Macro-F1 (%)

| Model | clean | 10dB | 5dB | 0dB | -5dB | -10dB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 100.00 ± 0.02 (+0.12, p=0.593) | 99.99 ± 0.04 (+0.19, p=1.000) | 99.92 ± 0.20 (+0.17, p=0.139) | 93.36 ± 5.79 (-4.02, p=0.003) | 31.02 ± 15.53 (-16.39, p=0.003) | 9.72 ± 0.59 (-1.85, p=0.075) |
| PAC-1DCNN (full) | 99.88 ± 0.53 | 99.79 ± 0.92 | 99.74 ± 1.13 | 97.38 ± 3.40 | 47.40 ± 17.68 | 11.57 ± 5.57 |
| PAC w/o Residual Filter | 99.98 ± 0.08 (+0.10, p=0.655) | 99.98 ± 0.09 (+0.18, p=0.655) | 99.97 ± 0.08 (+0.23, p=0.624) | 96.73 ± 5.20 (-0.65, p=0.573) | 39.24 ± 13.76 (-8.17, p=0.044) | 10.58 ± 3.52 (-0.99, p=0.401) |
| PAC w/o Dual-Stream | 99.95 ± 0.21 (+0.07, p=1.000) | 99.92 ± 0.36 (+0.12, p=0.655) | 99.85 ± 0.54 (+0.11, p=0.241) | 98.50 ± 3.95 (+1.11, p=0.019) | 51.77 ± 22.16 (+4.37, p=0.452) | 13.56 ± 12.05 (+1.99, p=0.893) |
| PAC w/o Attention | 99.91 ± 0.38 (+0.03, p=1.000) | 99.89 ± 0.46 (+0.09, p=1.000) | 99.85 ± 0.48 (+0.10, p=0.866) | 97.61 ± 2.41 (+0.23, p=0.622) | 47.85 ± 13.12 (+0.44, p=0.674) | 11.50 ± 4.54 (-0.07, p=0.515) |

## Macro-F1 (%) by held-out load (clean / 0 dB)

| Model | 0 HP | 1 HP | 2 HP | 3 HP |
| :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 100.0 / 90.0 | 100.0 / 96.2 | 100.0 / 92.2 | 100.0 / 95.0 |
| PAC-1DCNN (full) | 100.0 / 98.6 | 100.0 / 99.2 | 100.0 / 96.7 | 99.5 / 95.0 |
| PAC w/o Residual Filter | 100.0 / 99.1 | 100.0 / 98.6 | 100.0 / 94.6 | 99.9 / 94.6 |
| PAC w/o Dual-Stream | 100.0 / 99.3 | 100.0 / 98.7 | 100.0 / 99.8 | 99.8 / 96.1 |
| PAC w/o Attention | 100.0 / 98.9 | 100.0 / 98.9 | 100.0 / 97.9 | 99.6 / 94.7 |

## False alarm rate and calibration at 0 dB

| Model | False alarms (%) | Missed faults (%) | ECE calibrated (%) | ECE uncalibrated (%) | Mean T | Calib-set accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 0.00 ± 0.00 | 0.47 ± 1.85 | 4.15 ± 4.36 | 2.22 | 0.399 | 100.00 |
| PAC-1DCNN (full) | 0.07 ± 0.22 | 0.01 ± 0.06 | 1.76 ± 2.75 | 1.52 | 0.414 | 100.00 |
| PAC w/o Residual Filter | 2.04 ± 9.13 | 0.04 ± 0.15 | 2.40 ± 4.57 | 2.45 | 0.442 | 100.00 |
| PAC w/o Dual-Stream | 1.23 ± 5.49 | 0.00 ± 0.00 | 1.34 ± 3.74 | 1.30 | 0.354 | 100.00 |
| PAC w/o Attention | 0.02 ± 0.11 | 0.11 ± 0.35 | 1.31 ± 1.59 | 1.68 | 0.435 | 100.00 |

## Findings (generated from the numbers above)

- **clean**: no significant difference between baseline and PAC-1DCNN (+0.12, p=0.593).
- **10dB**: no significant difference between baseline and PAC-1DCNN (+0.19, p=1.000).
- **5dB**: no significant difference between baseline and PAC-1DCNN (+0.17, p=0.139).
- **0dB**: PAC-1DCNN is significantly better (baseline − PAC = -4.02 macro-F1, p=0.003).
- **-5dB**: PAC-1DCNN is significantly better (baseline − PAC = -16.39 macro-F1, p=0.003).
- **-10dB**: no significant difference between baseline and PAC-1DCNN (-1.85, p=0.075).
- **PAC w/o Residual Filter**: significant change vs full model at -5dB: -8.17 (negative = component helps).
- **PAC w/o Dual-Stream**: significant change vs full model at 0dB: +1.11 (negative = component helps).
- **PAC w/o Attention**: no significant macro-F1 change at any noise level.
