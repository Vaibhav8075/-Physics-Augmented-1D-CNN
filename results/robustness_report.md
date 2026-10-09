# Robustness Study: Fault-Size Split, Noise Spectra, Stronger Baselines

Split: fault-size hold-out (one of 0.007/0.014/0.021 in. per fold, all loads); Normal split by time at 80% with a 1% gap. 15 epochs, final epoch kept, no calibration. Values are mean ± std of macro-F1 (%) over 15 runs (3 folds × 5 seeds). Noise conditions add only coloured noise (no drift or spikes); `drift+spikes` adds only drift and spikes at the test probabilities. Contrasts use a paired Wilcoxon signed-rank test over the 15 (fold, seed) pairs, Holm-corrected over all contrasts × conditions within each training regime.

**Limitation:** CWRU has one healthy bearing, so Normal windows in train and test come from the same recordings (split by time). WDCNN-style layer sizes were not checked against the original paper's table.

## Training noise: lowpass

| Model | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 47.6 ± 11.8 | 47.7 ± 12.3 | 46.7 ± 14.9 | 33.4 ± 20.9 | 13.4 ± 15.4 | 47.5 ± 12.7 | 46.6 ± 14.7 | 33.4 ± 13.1 | 16.3 ± 9.2 | 49.5 ± 12.7 | 45.8 ± 15.1 | 25.6 ± 7.8 | 14.0 ± 8.8 | 47.9 ± 12.2 |
| Standard 1D-CNN + fixed filter | 48.1 ± 11.5 | 48.8 ± 12.6 | 48.7 ± 13.3 | 45.3 ± 15.9 | 29.0 ± 18.1 | 49.0 ± 13.0 | 43.9 ± 17.1 | 30.0 ± 8.9 | 18.4 ± 7.7 | 51.6 ± 12.3 | 42.6 ± 16.6 | 19.3 ± 6.0 | 13.6 ± 5.7 | 48.3 ± 11.7 |
| PAC w/o fixed filter | 46.4 ± 10.4 | 46.4 ± 11.5 | 44.3 ± 14.2 | 33.1 ± 21.9 | 20.8 ± 24.2 | 45.6 ± 10.9 | 45.7 ± 12.1 | 36.3 ± 17.0 | 20.0 ± 10.8 | 47.0 ± 11.7 | 43.3 ± 14.4 | 32.8 ± 13.7 | 17.2 ± 7.1 | 46.8 ± 11.1 |
| PAC-1DCNN (full) | 46.4 ± 11.1 | 46.6 ± 12.2 | 45.2 ± 14.9 | 35.0 ± 23.9 | 20.9 ± 23.6 | 45.9 ± 11.7 | 45.8 ± 13.6 | 35.4 ± 16.4 | 18.6 ± 9.0 | 47.8 ± 12.8 | 44.1 ± 16.3 | 28.2 ± 10.0 | 13.9 ± 5.3 | 46.7 ± 11.6 |
| WDCNN-style wide-kernel CNN | 48.4 ± 10.7 | 48.4 ± 10.7 | 48.4 ± 10.9 | 47.4 ± 13.0 | 35.6 ± 15.6 | 49.0 ± 9.9 | 50.9 ± 9.6 | 51.8 ± 11.3 | 30.7 ± 13.2 | 48.9 ± 10.1 | 50.5 ± 10.3 | 49.9 ± 11.6 | 30.2 ± 14.3 | 48.6 ± 10.5 |
| Envelope spectrum + Random Forest | 52.6 ± 14.0 | 56.5 ± 19.0 | 58.5 ± 20.7 | 56.3 ± 15.3 | 53.3 ± 8.3 | 60.8 ± 23.8 | 60.6 ± 18.1 | 50.5 ± 19.6 | 20.5 ± 9.0 | 60.4 ± 19.6 | 59.2 ± 18.3 | 48.9 ± 24.4 | 26.3 ± 10.4 | 54.5 ± 15.7 |

### Contrasts (macro-F1 difference a − b; significant after Holm correction in bold)

| Contrast | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q1 filter at baseline capacity: baseline_hp − baseline | +0.5 (9/15) | +1.1 (9/15) | +2.1 (7/15) | **+11.9 (15/15)** | **+15.6 (15/15)** | +1.5 (11/15) | -2.7 (5/15) | -3.3 (3/15) | +2.1 (7/15) | +2.1 (12/15) | -3.2 (2/15) | **-6.4 (1/15)** | -0.3 (3/15) | +0.4 (8/15) |
| Q1 filter at PAC capacity: physics − no_residual | -0.0 (4/15) | +0.2 (4/15) | +0.9 (5/15) | +1.9 (8/15) | +0.1 (6/15) | +0.3 (5/15) | +0.1 (6/15) | -0.9 (8/15) | -1.4 (9/15) | +0.7 (5/15) | +0.8 (7/15) | -4.6 (7/15) | -3.2 (4/15) | -0.1 (4/15) |
| PAC vs standard CNN: physics − baseline | -1.3 (5/15) | -1.2 (6/15) | -1.4 (7/15) | +1.7 (10/15) | +7.5 (8/15) | -1.7 (5/15) | -0.8 (4/15) | +2.0 (9/15) | +2.3 (9/15) | -1.8 (4/15) | -1.8 (6/15) | +2.6 (10/15) | -0.0 (7/15) | -1.2 (5/15) |
| PAC vs WDCNN-style: physics − wdcnn | -2.0 (4/15) | -1.9 (4/15) | -3.2 (4/15) | -12.4 (4/15) | -14.7 (4/15) | -3.1 (4/15) | -5.1 (2/15) | -16.4 (2/15) | -12.1 (3/15) | -1.1 (5/15) | -6.5 (3/15) | **-21.7 (1/15)** | -16.2 (2/15) | -1.9 (4/15) |
| PAC vs classical RF: physics − rf | **-6.3 (0/15)** | **-10.0 (0/15)** | **-13.2 (0/15)** | **-21.2 (0/15)** | **-32.4 (0/15)** | **-14.9 (0/15)** | **-14.8 (0/15)** | -15.2 (4/15) | -1.9 (7/15) | **-12.6 (0/15)** | **-15.1 (0/15)** | -20.7 (2/15) | -12.3 (3/15) | **-7.8 (0/15)** |

Cell = mean difference (runs where a > b / total).

### Per-class recall at clean / lowpass@0dB (PAC-1DCNN)

- clean: Normal (Healthy) 100.0%, Inner Race Fault 0.0%, Ball Fault 96.2%, Outer Race Fault 19.8%
- lowpass@0dB: Normal (Healthy) 100.0%, Inner Race Fault 0.0%, Ball Fault 75.9%, Outer Race Fault 33.0%

## Training noise: mixed

| Model | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 47.9 ± 12.0 | 47.5 ± 12.2 | 45.0 ± 15.2 | 32.0 ± 22.8 | 12.4 ± 14.7 | 46.9 ± 12.0 | 47.3 ± 11.9 | 34.2 ± 17.5 | 16.6 ± 9.9 | 47.3 ± 12.4 | 45.4 ± 15.1 | 33.1 ± 11.1 | 16.6 ± 8.6 | 47.9 ± 12.2 |
| Standard 1D-CNN + fixed filter | 47.2 ± 10.3 | 47.7 ± 11.3 | 48.0 ± 12.3 | 46.0 ± 15.2 | 31.0 ± 17.7 | 47.1 ± 11.4 | 45.5 ± 14.2 | 30.9 ± 16.6 | 20.3 ± 9.1 | 47.8 ± 12.2 | 43.0 ± 17.5 | 27.4 ± 7.0 | 16.6 ± 5.0 | 47.4 ± 10.6 |
| PAC w/o fixed filter | 48.3 ± 12.7 | 48.1 ± 13.1 | 45.5 ± 16.1 | 34.5 ± 23.9 | 19.5 ± 22.5 | 47.7 ± 12.8 | 47.5 ± 13.3 | 39.5 ± 19.1 | 15.8 ± 11.5 | 48.0 ± 13.3 | 46.0 ± 14.9 | 32.9 ± 13.8 | 16.6 ± 5.6 | 48.4 ± 12.8 |
| PAC-1DCNN (full) | 46.5 ± 11.0 | 46.5 ± 11.9 | 44.1 ± 14.9 | 33.3 ± 21.5 | 17.6 ± 19.8 | 45.3 ± 10.9 | 44.0 ± 11.9 | 36.3 ± 17.8 | 17.4 ± 10.9 | 46.1 ± 11.9 | 42.9 ± 13.7 | 29.8 ± 11.6 | 17.6 ± 6.6 | 46.7 ± 11.5 |
| WDCNN-style wide-kernel CNN | 49.6 ± 11.2 | 48.9 ± 11.3 | 48.1 ± 12.1 | 47.2 ± 14.2 | 33.0 ± 14.2 | 49.5 ± 10.7 | 50.8 ± 10.0 | 51.8 ± 8.7 | 28.3 ± 8.6 | 49.5 ± 10.7 | 50.6 ± 10.7 | 50.5 ± 8.5 | 29.7 ± 11.3 | 49.7 ± 11.5 |
| Envelope spectrum + Random Forest | 59.6 ± 22.0 | 59.9 ± 21.8 | 59.9 ± 20.8 | 58.2 ± 16.6 | 56.2 ± 10.2 | 62.3 ± 23.9 | 63.6 ± 20.0 | 54.7 ± 18.8 | 23.8 ± 10.7 | 61.2 ± 23.6 | 62.3 ± 21.5 | 51.4 ± 26.1 | 26.8 ± 11.6 | 59.7 ± 21.4 |

### Contrasts (macro-F1 difference a − b; significant after Holm correction in bold)

| Contrast | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q1 filter at baseline capacity: baseline_hp − baseline | -0.8 (7/15) | +0.2 (6/15) | +3.0 (8/15) | **+14.0 (15/15)** | **+18.6 (15/15)** | +0.2 (9/15) | -1.8 (4/15) | -3.3 (4/15) | +3.8 (10/15) | +0.5 (9/15) | -2.4 (4/15) | -5.7 (5/15) | -0.0 (6/15) | -0.5 (8/15) |
| Q1 filter at PAC capacity: physics − no_residual | -1.8 (3/15) | -1.6 (3/15) | -1.4 (5/15) | -1.2 (9/15) | -1.9 (6/15) | -2.3 (1/15) | -3.5 (5/15) | -3.2 (6/15) | +1.7 (10/15) | -1.9 (3/15) | -3.1 (4/15) | -3.1 (6/15) | +1.0 (8/15) | -1.7 (3/15) |
| PAC vs standard CNN: physics − baseline | -1.5 (6/15) | -1.0 (7/15) | -0.9 (10/15) | +1.3 (10/15) | +5.2 (7/15) | -1.6 (5/15) | -3.2 (5/15) | +2.0 (11/15) | +0.9 (9/15) | -1.2 (4/15) | -2.5 (9/15) | -3.2 (8/15) | +1.0 (8/15) | -1.2 (6/15) |
| PAC vs WDCNN-style: physics − wdcnn | -3.1 (5/15) | -2.4 (5/15) | -4.0 (3/15) | -13.9 (3/15) | -15.4 (3/15) | -4.2 (3/15) | -6.7 (3/15) | -15.6 (2/15) | -10.9 (4/15) | -3.4 (4/15) | -7.7 (3/15) | **-20.6 (1/15)** | -12.0 (4/15) | -3.0 (3/15) |
| PAC vs classical RF: physics − rf | **-13.1 (0/15)** | **-13.4 (0/15)** | **-15.8 (0/15)** | **-24.9 (0/15)** | **-38.6 (0/15)** | **-16.9 (0/15)** | **-19.6 (0/15)** | -18.5 (2/15) | -6.4 (5/15) | **-15.1 (0/15)** | **-19.4 (0/15)** | -21.5 (2/15) | -9.2 (4/15) | **-13.0 (0/15)** |

Cell = mean difference (runs where a > b / total).

### Per-class recall at clean / lowpass@0dB (PAC-1DCNN)

- clean: Normal (Healthy) 99.0%, Inner Race Fault 6.7%, Ball Fault 92.0%, Outer Race Fault 20.6%
- lowpass@0dB: Normal (Healthy) 98.5%, Inner Race Fault 6.7%, Ball Fault 69.2%, Outer Race Fault 26.6%

