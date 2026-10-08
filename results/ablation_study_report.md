# Ablation Study: Component Impact

Trained on loads [0, 1] HP, calibrated/selected on load [2] HP, tested on unseen load [3] HP with 8 dB plant noise. Each configuration was trained with 5 seeds ([0, 1, 2, 3, 4]); values are mean ± std and p-values are paired t-tests against M0 across seeds.

| Configuration | Accuracy (%) | Macro-F1 (%) | Δ Macro-F1 (p) | ECE (%) | NLL | FAR (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M0: Full Proposed Architecture | 85.82 ± 0.05 | 66.83 ± 0.21 | --- | 14.09 ± 0.25 | 2.03 ± 0.75 | 0.00 ± 0.00 |
| M1: w/o Kinematic Residual Filter | 86.08 ± 0.62 | 67.74 ± 2.24 | +0.91 (p=0.37) | 13.61 ± 1.31 | 1.75 ± 0.78 | 0.00 ± 0.00 |
| M2: w/o Dual-Stream Conv | 85.80 ± 0.00 | 66.74 ± 0.00 | -0.09 (p=0.37) | 14.20 ± 0.00 | 3.11 ± 1.02 | 0.00 ± 0.00 |
| M3: w/o Sensor-Temporal Attention | 85.80 ± 0.00 | 66.74 ± 0.00 | -0.09 (p=0.37) | 14.19 ± 0.01 | 2.73 ± 0.99 | 0.00 ± 0.00 |
| M4: w/o Temperature Calibration | 85.82 ± 0.05 | 66.83 ± 0.21 | +0.00 (p=1.00) | 11.95 ± 0.70 | 0.39 ± 0.11 | 0.00 ± 0.00 |

## Findings (generated from the numbers above)

- **Removing Kinematic Residual Filter**: no statistically significant change in macro_f1 (+0.91, p=0.37); ECE: no statistically significant change in ece (-0.47, p=0.37).
- **Removing Dual-Stream Conv**: no statistically significant change in macro_f1 (-0.09, p=0.37); ECE: no statistically significant change in ece (+0.11, p=0.37).
- **Removing Sensor-Temporal Attention**: no statistically significant change in macro_f1 (-0.09, p=0.37); ECE: no statistically significant change in ece (+0.11, p=0.37).
- **Removing temperature calibration** (same networks, T=1): accuracy is unchanged by construction; ECE: ece changes by -2.14 (p=0.001) — the model is better WITHOUT this component; NLL: nll changes by -1.64 (p=0.005) — the model is better WITHOUT this component.
