# Robustness Study: Fault-Size Split, Noise Spectra, Stronger Baselines

Split: fault-size hold-out (one of 0.007/0.014/0.021 in. per fold, all loads); Normal split by time at 80% with a 1% gap. 15 epochs, final epoch kept, no calibration. Values are mean ± std of macro-F1 (%) over 15 runs (3 folds × 5 seeds). Noise conditions add only coloured noise (no drift or spikes); `drift+spikes` adds only drift and spikes at the test probabilities. Contrasts use a paired Wilcoxon signed-rank test over the 15 (fold, seed) pairs, Holm-corrected over all contrasts × conditions within each training regime.

**Limitation:** CWRU has one healthy bearing, so Normal windows in train and test come from the same recordings (split by time). WDCNN-style layer sizes were not checked against the original paper's table.

## Training noise: lowpass

| Model | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 46.9 ± 10.4 | 46.4 ± 9.9 | 47.9 ± 10.9 | 47.9 ± 14.8 | 27.6 ± 13.8 | 46.2 ± 9.7 | 46.9 ± 11.2 | 27.7 ± 18.6 | 13.6 ± 8.2 | 46.3 ± 10.4 | 43.3 ± 17.0 | 19.6 ± 13.1 | 15.4 ± 8.6 | 46.9 ± 10.2 |
| Standard 1D-CNN + fixed filter | 46.9 ± 10.4 | 47.3 ± 10.9 | 47.8 ± 11.6 | 49.5 ± 12.1 | 32.1 ± 15.8 | 47.7 ± 11.8 | 47.1 ± 13.8 | 22.8 ± 15.3 | 12.9 ± 4.4 | 48.8 ± 13.3 | 35.6 ± 17.9 | 14.9 ± 9.8 | 15.6 ± 11.3 | 47.2 ± 10.7 |
| PAC w/o fixed filter | 48.0 ± 11.1 | 48.0 ± 11.4 | 48.8 ± 11.7 | 50.0 ± 13.5 | 38.7 ± 20.3 | 47.7 ± 10.7 | 47.8 ± 10.9 | 40.5 ± 17.4 | 14.2 ± 7.6 | 47.3 ± 11.0 | 44.0 ± 16.5 | 23.8 ± 14.6 | 18.1 ± 15.2 | 48.2 ± 11.3 |
| PAC-1DCNN (full) | 48.5 ± 11.2 | 48.4 ± 11.4 | 48.9 ± 11.7 | 50.8 ± 12.2 | 40.7 ± 14.5 | 47.7 ± 10.6 | 47.5 ± 11.3 | 29.6 ± 15.6 | 12.8 ± 3.9 | 47.5 ± 11.1 | 40.4 ± 16.7 | 18.9 ± 11.3 | 14.7 ± 7.0 | 48.6 ± 11.4 |
| WDCNN-style wide-kernel CNN | 48.6 ± 8.4 | 48.7 ± 8.2 | 49.6 ± 8.2 | 51.0 ± 8.9 | 29.0 ± 10.5 | 48.8 ± 7.9 | 50.8 ± 8.3 | 45.6 ± 6.3 | 18.7 ± 4.4 | 48.7 ± 8.0 | 50.5 ± 8.4 | 44.8 ± 7.3 | 18.1 ± 6.0 | 48.9 ± 8.2 |
| Envelope spectrum + Random Forest | 52.8 ± 14.7 | 57.3 ± 20.7 | 58.7 ± 22.7 | 55.5 ± 16.5 | 28.1 ± 7.4 | 61.1 ± 24.8 | 40.6 ± 20.4 | 30.0 ± 15.1 | 24.3 ± 11.2 | 62.2 ± 21.7 | 38.6 ± 20.0 | 30.3 ± 18.3 | 25.5 ± 14.4 | 55.2 ± 17.4 |

### Contrasts (macro-F1 difference a − b; significant after Holm correction in bold)

| Contrast | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q1 filter at baseline capacity: baseline_hp − baseline | +0.0 (4/15) | +0.8 (6/15) | -0.0 (5/15) | +1.6 (5/15) | +4.4 (8/15) | +1.6 (8/15) | +0.2 (8/15) | -4.9 (6/15) | -0.7 (4/15) | +2.4 (9/15) | -7.7 (3/15) | -4.7 (0/15) | +0.1 (2/15) | +0.4 (4/15) |
| Q1 filter at PAC capacity: physics − no_residual | +0.5 (4/15) | +0.4 (6/15) | +0.1 (4/15) | +0.8 (6/15) | +2.0 (8/15) | -0.0 (5/15) | -0.2 (4/15) | -10.8 (4/15) | -1.4 (3/15) | +0.1 (4/15) | -3.6 (5/15) | -4.9 (3/15) | -3.4 (3/15) | +0.5 (4/15) |
| PAC vs standard CNN: physics − baseline | +1.6 (7/15) | +2.0 (7/15) | +1.0 (7/15) | +2.9 (7/15) | +13.0 (12/15) | +1.5 (7/15) | +0.7 (6/15) | +1.9 (9/15) | -0.9 (3/15) | +1.2 (7/15) | -2.8 (8/15) | -0.7 (7/15) | -0.7 (4/15) | +1.8 (7/15) |
| PAC vs WDCNN-style: physics − wdcnn | -0.0 (6/15) | -0.2 (6/15) | -0.7 (7/15) | -0.2 (7/15) | +11.6 (12/15) | -1.1 (6/15) | -3.3 (3/15) | -16.0 (2/15) | -5.9 (2/15) | -1.3 (5/15) | -10.0 (2/15) | **-25.8 (0/15)** | -3.4 (2/15) | -0.3 (7/15) |
| PAC vs classical RF: physics − rf | -4.3 (2/15) | -8.8 (2/15) | -9.8 (2/15) | -4.7 (2/15) | +12.6 (11/15) | **-13.5 (1/15)** | +6.9 (9/15) | -0.4 (8/15) | -11.6 (4/15) | -14.7 (3/15) | +1.8 (8/15) | -11.4 (5/15) | -10.7 (5/15) | -6.5 (2/15) |

Cell = mean difference (runs where a > b / total).

### Per-class recall at clean / lowpass@0dB (PAC-1DCNN)

- clean: Normal (Healthy) 100.0%, Inner Race Fault 1.5%, Ball Fault 96.8%, Outer Race Fault 25.0%
- lowpass@0dB: Normal (Healthy) 100.0%, Inner Race Fault 6.3%, Ball Fault 95.9%, Outer Race Fault 34.1%

## Training noise: mixed

| Model | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 45.2 ± 6.9 | 45.6 ± 7.9 | 47.5 ± 10.4 | 47.1 ± 14.8 | 27.6 ± 14.2 | 44.7 ± 6.9 | 47.5 ± 8.7 | 39.0 ± 15.8 | 16.5 ± 11.0 | 45.7 ± 8.5 | 46.5 ± 12.7 | 26.5 ± 16.5 | 18.6 ± 15.2 | 45.5 ± 7.2 |
| Standard 1D-CNN + fixed filter | 45.8 ± 8.5 | 46.0 ± 9.1 | 46.6 ± 10.1 | 51.0 ± 11.0 | 36.7 ± 16.6 | 46.3 ± 9.8 | 49.6 ± 11.2 | 26.6 ± 15.0 | 13.9 ± 6.1 | 47.9 ± 11.4 | 45.4 ± 15.8 | 19.4 ± 13.6 | 14.3 ± 12.0 | 46.1 ± 8.9 |
| PAC w/o fixed filter | 47.2 ± 10.1 | 47.2 ± 10.2 | 48.5 ± 11.0 | 47.7 ± 15.9 | 32.6 ± 21.7 | 46.7 ± 9.6 | 48.6 ± 9.3 | 48.1 ± 14.9 | 13.8 ± 8.1 | 46.7 ± 10.1 | 47.0 ± 10.8 | 33.4 ± 13.5 | 22.7 ± 18.0 | 47.2 ± 10.1 |
| PAC-1DCNN (full) | 48.3 ± 11.3 | 48.6 ± 12.0 | 48.6 ± 12.8 | 47.5 ± 15.2 | 34.8 ± 15.2 | 47.4 ± 10.7 | 48.6 ± 10.8 | 43.6 ± 11.6 | 17.6 ± 8.0 | 47.6 ± 11.0 | 48.1 ± 12.1 | 27.9 ± 12.2 | 15.2 ± 11.9 | 48.4 ± 11.5 |
| WDCNN-style wide-kernel CNN | 51.6 ± 9.9 | 51.7 ± 9.8 | 52.9 ± 9.9 | 53.9 ± 9.9 | 27.3 ± 10.7 | 51.8 ± 9.5 | 53.4 ± 9.8 | 45.6 ± 12.0 | 21.2 ± 9.5 | 51.8 ± 9.6 | 52.8 ± 9.9 | 46.4 ± 11.2 | 22.3 ± 11.2 | 52.0 ± 10.1 |
| Envelope spectrum + Random Forest | 60.4 ± 23.4 | 61.0 ± 24.1 | 59.3 ± 21.7 | 56.4 ± 17.2 | 45.2 ± 7.3 | 62.4 ± 25.2 | 64.8 ± 21.9 | 32.3 ± 12.5 | 21.7 ± 5.9 | 61.4 ± 24.7 | 63.6 ± 21.6 | 29.8 ± 16.5 | 22.7 ± 10.6 | 60.4 ± 23.0 |

### Contrasts (macro-F1 difference a − b; significant after Holm correction in bold)

| Contrast | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q1 filter at baseline capacity: baseline_hp − baseline | +0.6 (7/15) | +0.5 (8/15) | -0.9 (6/15) | +3.9 (9/15) | +9.1 (9/15) | +1.5 (8/15) | +2.1 (10/15) | -12.5 (1/15) | -2.6 (5/15) | +2.1 (9/15) | -1.2 (8/15) | -7.1 (5/15) | -4.3 (2/15) | +0.6 (7/15) |
| Q1 filter at PAC capacity: physics − no_residual | +1.1 (6/15) | +1.4 (7/15) | +0.2 (7/15) | -0.1 (7/15) | +2.2 (10/15) | +0.8 (5/15) | -0.1 (6/15) | -4.5 (6/15) | +3.8 (6/15) | +1.0 (6/15) | +1.1 (6/15) | -5.4 (4/15) | -7.5 (4/15) | +1.2 (5/15) |
| PAC vs standard CNN: physics − baseline | +3.1 (10/15) | +3.1 (10/15) | +1.1 (9/15) | +0.5 (7/15) | +7.2 (11/15) | +2.7 (11/15) | +1.1 (8/15) | +4.6 (10/15) | +1.1 (6/15) | +1.9 (8/15) | +1.6 (6/15) | +1.4 (9/15) | -3.3 (4/15) | +2.9 (9/15) |
| PAC vs WDCNN-style: physics − wdcnn | -3.3 (4/15) | -3.1 (4/15) | -4.2 (4/15) | -6.4 (5/15) | +7.5 (10/15) | -4.4 (5/15) | -4.8 (5/15) | -2.0 (6/15) | -3.6 (4/15) | -4.1 (5/15) | -4.6 (5/15) | **-18.5 (0/15)** | -7.1 (3/15) | -3.7 (4/15) |
| PAC vs classical RF: physics − rf | **-12.1 (0/15)** | **-12.4 (0/15)** | **-10.6 (0/15)** | **-8.9 (1/15)** | -10.4 (4/15) | **-15.0 (0/15)** | **-16.2 (0/15)** | +11.3 (12/15) | -4.1 (5/15) | **-13.8 (0/15)** | **-15.5 (0/15)** | -1.9 (8/15) | -7.5 (3/15) | **-12.0 (0/15)** |

Cell = mean difference (runs where a > b / total).

### Per-class recall at clean / lowpass@0dB (PAC-1DCNN)

- clean: Normal (Healthy) 100.0%, Inner Race Fault 0.4%, Ball Fault 96.8%, Outer Race Fault 25.2%
- lowpass@0dB: Normal (Healthy) 100.0%, Inner Race Fault 1.2%, Ball Fault 92.1%, Outer Race Fault 34.3%

