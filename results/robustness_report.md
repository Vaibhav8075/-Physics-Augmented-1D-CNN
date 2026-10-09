# Robustness Study: Fault-Size Split, Noise Spectra, Stronger Baselines

Split: fault-size hold-out (one of 0.007/0.014/0.021 in. per fold, all loads); Normal split by time at 80% with a 1% gap. 15 epochs, final epoch kept, no calibration. Values are mean ± std of macro-F1 (%) over 15 runs (3 folds × 5 seeds). Noise conditions add only coloured noise (no drift or spikes); `drift+spikes` adds only drift and spikes at the test probabilities. Contrasts use a paired Wilcoxon signed-rank test over the 15 (fold, seed) pairs, Holm-corrected over all contrasts × conditions within each training regime and contrast family (primary: Q1-Q4; secondary: S1-S3, the added models).

**Limitation:** CWRU has one healthy bearing, so Normal windows in train and test come from the same recordings (split by time). The WDCNN-style network follows Table 2 of Zhang et al. (2017) except for the input length (1024 samples) and AdaBN.

## Training noise: lowpass

| Model | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 46.8 ± 17.0 | 46.6 ± 18.6 | 43.0 ± 23.5 | 26.7 ± 20.0 | 6.3 ± 7.8 | 48.9 ± 17.2 | 51.4 ± 17.2 | 39.3 ± 18.2 | 13.7 ± 10.9 | 50.1 ± 17.8 | 51.4 ± 19.4 | 34.5 ± 17.7 | 12.9 ± 8.7 | 47.0 ± 17.3 |
| Standard 1D-CNN + fixed filter | 44.2 ± 9.8 | 45.7 ± 13.1 | 42.1 ± 19.0 | 34.4 ± 22.8 | 19.8 ± 23.8 | 47.4 ± 15.5 | 41.7 ± 19.4 | 26.5 ± 18.7 | 13.8 ± 8.5 | 48.8 ± 18.6 | 44.2 ± 18.9 | 20.5 ± 12.8 | 13.4 ± 5.1 | 44.5 ± 10.3 |
| PAC w/o fixed filter | 45.8 ± 13.4 | 45.1 ± 14.5 | 39.6 ± 19.6 | 29.7 ± 21.6 | 14.6 ± 16.6 | 46.3 ± 12.7 | 49.8 ± 12.3 | 46.0 ± 13.6 | 23.5 ± 13.2 | 48.4 ± 13.4 | 53.0 ± 14.3 | 39.8 ± 10.8 | 18.2 ± 8.4 | 45.9 ± 13.5 |
| PAC-1DCNN (full) | 45.1 ± 12.6 | 44.5 ± 13.6 | 38.8 ± 18.2 | 27.9 ± 19.1 | 10.9 ± 13.8 | 45.2 ± 11.6 | 47.6 ± 12.2 | 42.5 ± 13.1 | 22.0 ± 10.2 | 46.8 ± 12.1 | 51.0 ± 14.3 | 37.2 ± 7.7 | 21.1 ± 8.9 | 44.7 ± 12.4 |
| WDCNN-style wide-kernel CNN | 47.5 ± 12.5 | 47.5 ± 13.1 | 47.2 ± 15.0 | 43.3 ± 17.8 | 29.6 ± 14.0 | 48.5 ± 12.6 | 51.0 ± 11.9 | 52.6 ± 8.9 | 35.6 ± 11.8 | 48.5 ± 12.4 | 51.0 ± 11.8 | 53.7 ± 7.8 | 35.3 ± 11.2 | 48.1 ± 12.3 |
| Envelope spectrum + Random Forest | 64.0 ± 25.4 | 62.9 ± 25.1 | 59.5 ± 25.2 | 50.6 ± 22.8 | 37.7 ± 18.9 | 62.8 ± 26.3 | 55.8 ± 28.3 | 32.6 ± 12.7 | 14.3 ± 4.9 | 61.3 ± 28.6 | 51.6 ± 27.2 | 30.3 ± 9.7 | 22.9 ± 9.0 | 63.9 ± 24.8 |
| Standard 1D-CNN + learnable filter | 44.4 ± 9.2 | 45.7 ± 12.1 | 41.9 ± 17.7 | 34.3 ± 22.4 | 18.6 ± 22.3 | 48.8 ± 13.4 | 42.6 ± 17.8 | 25.4 ± 16.1 | 14.6 ± 9.3 | 49.4 ± 17.1 | 43.9 ± 18.4 | 19.9 ± 15.1 | 12.9 ± 5.6 | 44.7 ± 9.7 |
| PAC single residual stream | 45.7 ± 10.9 | 47.3 ± 14.0 | 43.4 ± 18.7 | 35.5 ± 21.8 | 25.3 ± 27.0 | 48.1 ± 15.0 | 43.6 ± 18.7 | 32.7 ± 16.2 | 21.0 ± 9.4 | 49.5 ± 18.3 | 45.9 ± 20.1 | 19.1 ± 9.7 | 16.7 ± 4.6 | 46.0 ± 11.3 |
| Kinematic envelope features + Random Forest | 48.4 ± 22.5 | 47.2 ± 22.2 | 45.1 ± 21.3 | 41.5 ± 20.5 | 34.4 ± 17.5 | 46.0 ± 22.4 | 42.3 ± 20.9 | 35.1 ± 17.7 | 19.3 ± 5.2 | 47.4 ± 22.7 | 44.3 ± 22.1 | 39.6 ± 20.7 | 27.1 ± 11.6 | 47.5 ± 22.5 |

### Contrasts (macro-F1 difference a − b; significant after Holm correction in bold)

| Contrast | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q1 filter at baseline capacity (primary): baseline_hp − baseline | -2.6 (9/15) | -0.9 (10/15) | -0.9 (12/15) | **+7.8 (15/15)** | +13.5 (11/15) | -1.5 (9/15) | -9.7 (2/15) | **-12.8 (1/15)** | +0.1 (8/15) | -1.3 (8/15) | -7.2 (3/15) | **-14.0 (2/15)** | +0.5 (5/15) | -2.5 (8/15) |
| Q1 filter at PAC capacity (primary): physics − no_residual | -0.7 (4/15) | -0.6 (8/15) | -0.9 (10/15) | -1.8 (6/15) | -3.7 (4/15) | -1.1 (4/15) | -2.3 (3/15) | -3.4 (5/15) | -1.5 (7/15) | -1.6 (5/15) | -2.0 (5/15) | -2.6 (4/15) | +2.9 (9/15) | -1.2 (5/15) |
| PAC vs standard CNN (primary): physics − baseline | -1.7 (8/15) | -2.1 (9/15) | -4.2 (7/15) | +1.2 (10/15) | +4.6 (9/15) | -3.7 (7/15) | -3.8 (6/15) | +3.2 (9/15) | +8.3 (11/15) | -3.3 (7/15) | -0.4 (10/15) | +2.7 (11/15) | +8.2 (11/15) | -2.3 (9/15) |
| PAC vs WDCNN-style (primary): physics − wdcnn | -2.4 (3/15) | -2.9 (3/15) | -8.4 (2/15) | -15.4 (1/15) | -18.6 (1/15) | -3.3 (3/15) | -3.4 (6/15) | -10.1 (2/15) | -13.6 (3/15) | -1.7 (5/15) | -0.0 (8/15) | **-16.5 (0/15)** | -14.3 (1/15) | -3.4 (3/15) |
| PAC vs classical RF (primary): physics − rf | **-18.9 (0/15)** | **-18.3 (0/15)** | **-20.7 (0/15)** | **-22.7 (0/15)** | **-26.8 (0/15)** | **-17.6 (2/15)** | -8.2 (5/15) | +9.9 (12/15) | +7.7 (9/15) | -14.4 (4/15) | -0.6 (9/15) | +6.9 (12/15) | -1.9 (6/15) | **-19.2 (0/15)** |
| S1 learnable vs fixed filter (secondary): baseline_lhp − baseline_hp | +0.2 (10/15) | -0.0 (10/15) | -0.2 (10/15) | -0.1 (8/15) | -1.1 (3/15) | +1.4 (12/15) | +0.9 (9/15) | -1.1 (9/15) | +0.8 (6/15) | +0.5 (10/15) | -0.4 (8/15) | -0.6 (7/15) | -0.6 (6/15) | +0.2 (12/15) |
| S2 single residual stream vs PAC (secondary): no_dual − physics | +0.6 (11/15) | +2.7 (11/15) | +4.6 (12/15) | **+7.6 (15/15)** | **+14.4 (15/15)** | +2.9 (11/15) | -4.0 (6/15) | -9.8 (2/15) | -1.0 (8/15) | +2.7 (9/15) | -5.1 (4/15) | **-18.1 (0/15)** | -4.4 (3/15) | +1.3 (12/15) |
| S3 kinematic vs spectral features (secondary): kin − rf | -15.6 (5/15) | -15.7 (5/15) | **-14.4 (1/15)** | **-9.1 (0/15)** | -3.3 (6/15) | **-16.8 (0/15)** | **-13.5 (0/15)** | +2.5 (8/15) | +5.0 (12/15) | **-13.9 (0/15)** | **-7.3 (1/15)** | +9.3 (10/15) | +4.2 (10/15) | -16.4 (5/15) |

Cell = mean difference (runs where a > b / total).

### Per-class recall at clean / lowpass@0dB (PAC-1DCNN)

- clean: Normal (Healthy) 99.8%, Inner Race Fault 0.4%, Ball Fault 93.9%, Outer Race Fault 21.0%
- lowpass@0dB: Normal (Healthy) 97.6%, Inner Race Fault 0.0%, Ball Fault 68.9%, Outer Race Fault 24.6%

## Training noise: mixed

| Model | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 45.9 ± 13.4 | 45.2 ± 15.4 | 40.3 ± 19.8 | 25.8 ± 19.0 | 5.1 ± 4.2 | 46.3 ± 13.1 | 48.0 ± 13.5 | 42.1 ± 16.9 | 18.7 ± 12.6 | 46.7 ± 14.0 | 48.5 ± 17.2 | 41.0 ± 19.2 | 17.6 ± 9.6 | 46.2 ± 14.1 |
| Standard 1D-CNN + fixed filter | 45.9 ± 11.6 | 46.6 ± 13.7 | 43.1 ± 18.5 | 34.0 ± 22.7 | 19.5 ± 22.8 | 46.6 ± 13.7 | 43.3 ± 18.0 | 31.2 ± 19.5 | 17.4 ± 9.8 | 47.9 ± 15.1 | 46.2 ± 18.1 | 30.1 ± 14.2 | 18.2 ± 6.6 | 46.1 ± 11.8 |
| PAC w/o fixed filter | 46.0 ± 14.8 | 45.1 ± 16.1 | 40.2 ± 20.2 | 27.8 ± 21.0 | 10.5 ± 11.9 | 45.7 ± 14.8 | 47.8 ± 13.9 | 44.6 ± 16.7 | 17.4 ± 10.3 | 46.3 ± 15.2 | 50.2 ± 16.5 | 43.7 ± 17.4 | 20.0 ± 9.6 | 46.2 ± 14.9 |
| PAC-1DCNN (full) | 45.4 ± 13.6 | 45.0 ± 14.9 | 40.5 ± 19.5 | 26.7 ± 17.7 | 9.0 ± 10.5 | 45.3 ± 13.4 | 47.9 ± 12.8 | 43.6 ± 16.5 | 17.0 ± 11.0 | 45.9 ± 14.1 | 50.2 ± 15.4 | 42.3 ± 15.6 | 20.2 ± 9.4 | 45.7 ± 13.9 |
| WDCNN-style wide-kernel CNN | 46.1 ± 10.6 | 46.4 ± 11.4 | 47.2 ± 12.8 | 46.0 ± 15.4 | 33.3 ± 14.3 | 47.1 ± 10.4 | 49.4 ± 9.3 | 51.0 ± 10.6 | 32.0 ± 14.1 | 47.0 ± 10.3 | 49.2 ± 9.2 | 51.7 ± 9.0 | 31.6 ± 13.0 | 46.6 ± 10.8 |
| Envelope spectrum + Random Forest | 63.5 ± 24.3 | 63.9 ± 24.0 | 62.4 ± 23.0 | 53.4 ± 18.8 | 48.4 ± 14.7 | 63.9 ± 24.4 | 60.5 ± 25.6 | 42.4 ± 11.0 | 21.6 ± 8.2 | 62.4 ± 25.2 | 59.6 ± 26.9 | 44.3 ± 11.2 | 30.0 ± 4.9 | 63.6 ± 23.9 |
| Standard 1D-CNN + learnable filter | 46.2 ± 11.6 | 46.9 ± 13.3 | 42.5 ± 17.4 | 33.4 ± 22.3 | 14.7 ± 16.4 | 47.4 ± 12.5 | 45.5 ± 14.7 | 30.4 ± 14.9 | 17.8 ± 11.4 | 48.4 ± 13.7 | 48.0 ± 18.0 | 29.5 ± 14.6 | 18.8 ± 8.2 | 46.4 ± 11.9 |
| PAC single residual stream | 45.9 ± 11.1 | 47.4 ± 13.9 | 44.4 ± 18.6 | 35.3 ± 21.5 | 24.1 ± 25.0 | 47.5 ± 13.9 | 45.0 ± 17.4 | 41.0 ± 16.6 | 21.3 ± 8.8 | 48.4 ± 14.9 | 47.6 ± 19.1 | 35.5 ± 12.7 | 20.1 ± 9.5 | 46.2 ± 11.5 |
| Kinematic envelope features + Random Forest | 48.4 ± 22.5 | 47.2 ± 22.1 | 45.0 ± 21.2 | 41.6 ± 20.4 | 34.3 ± 17.2 | 46.2 ± 22.3 | 42.3 ± 20.9 | 35.2 ± 17.5 | 19.3 ± 5.3 | 47.4 ± 22.6 | 44.3 ± 22.0 | 39.6 ± 20.7 | 27.3 ± 11.8 | 47.5 ± 22.4 |

### Contrasts (macro-F1 difference a − b; significant after Holm correction in bold)

| Contrast | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q1 filter at baseline capacity (primary): baseline_hp − baseline | -0.0 (8/15) | +1.4 (9/15) | +2.8 (12/15) | +8.2 (13/15) | +14.3 (10/15) | +0.3 (7/15) | -4.8 (4/15) | **-10.9 (2/15)** | -1.3 (7/15) | +1.2 (8/15) | -2.3 (6/15) | -10.9 (4/15) | +0.6 (7/15) | -0.1 (8/15) |
| Q1 filter at PAC capacity (primary): physics − no_residual | -0.7 (7/15) | -0.1 (9/15) | +0.3 (6/15) | -1.1 (8/15) | -1.4 (4/15) | -0.4 (9/15) | +0.1 (8/15) | -1.0 (6/15) | -0.4 (5/15) | -0.4 (9/15) | -0.0 (7/15) | -1.4 (7/15) | +0.2 (5/15) | -0.5 (8/15) |
| PAC vs standard CNN (primary): physics − baseline | -0.5 (7/15) | -0.2 (10/15) | +0.2 (7/15) | +0.9 (10/15) | +3.9 (8/15) | -1.0 (9/15) | -0.1 (9/15) | +1.5 (9/15) | -1.7 (9/15) | -0.9 (9/15) | +1.6 (8/15) | +1.3 (11/15) | +2.5 (8/15) | -0.5 (7/15) |
| PAC vs WDCNN-style (primary): physics − wdcnn | -0.7 (4/15) | -1.4 (5/15) | -6.7 (5/15) | **-19.3 (0/15)** | **-24.3 (1/15)** | -1.7 (4/15) | -1.5 (6/15) | -7.4 (4/15) | **-15.0 (1/15)** | -1.2 (7/15) | +0.9 (10/15) | -9.4 (3/15) | -11.5 (4/15) | -0.9 (4/15) |
| PAC vs classical RF (primary): physics − rf | **-18.1 (0/15)** | **-18.8 (0/15)** | **-21.9 (0/15)** | **-26.7 (0/15)** | **-39.4 (0/15)** | **-18.5 (0/15)** | -12.6 (4/15) | +1.2 (10/15) | -4.6 (5/15) | **-16.5 (0/15)** | -9.4 (4/15) | -2.0 (7/15) | -9.9 (2/15) | **-17.9 (0/15)** |
| S1 learnable vs fixed filter (secondary): baseline_lhp − baseline_hp | +0.3 (10/15) | +0.3 (10/15) | -0.6 (9/15) | -0.6 (6/15) | -4.8 (3/15) | +0.8 (11/15) | +2.2 (10/15) | -0.8 (10/15) | +0.5 (6/15) | +0.5 (11/15) | +1.8 (10/15) | -0.6 (8/15) | +0.6 (5/15) | +0.3 (10/15) |
| S2 single residual stream vs PAC (secondary): no_dual − physics | +0.5 (10/15) | +2.4 (12/15) | +3.9 (12/15) | **+8.6 (15/15)** | **+15.1 (14/15)** | +2.2 (11/15) | -2.9 (6/15) | -2.6 (4/15) | +4.3 (10/15) | +2.5 (10/15) | -2.6 (7/15) | -6.8 (4/15) | -0.1 (6/15) | +0.5 (10/15) |
| S3 kinematic vs spectral features (secondary): kin − rf | -15.1 (5/15) | -16.7 (5/15) | -17.4 (3/15) | **-11.8 (0/15)** | **-14.1 (0/15)** | -17.6 (5/15) | **-18.2 (0/15)** | -7.3 (5/15) | -2.2 (6/15) | -15.0 (5/15) | **-15.2 (0/15)** | -4.6 (5/15) | -2.7 (8/15) | -16.1 (5/15) |

Cell = mean difference (runs where a > b / total).

### Per-class recall at clean / lowpass@0dB (PAC-1DCNN)

- clean: Normal (Healthy) 100.0%, Inner Race Fault 0.3%, Ball Fault 93.8%, Outer Race Fault 24.4%
- lowpass@0dB: Normal (Healthy) 100.0%, Inner Race Fault 0.0%, Ball Fault 68.5%, Outer Race Fault 20.5%

