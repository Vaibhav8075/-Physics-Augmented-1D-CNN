# Paderborn Bearing Study: Unseen Physical Bearings, Noise Spectra, Stronger Baselines

Data: Paderborn University KAt Bearing DataCenter (Lessmeier et al., 2016; CC BY-NC 4.0). Vibration channel decimated 64 kHz -> 16 kHz, 2048-sample windows, classes healthy / inner ring / outer ring. Every test bearing is a physical specimen never seen in training (healthy bearings included). Noise conditions add only coloured noise; `drift+spikes` adds only drift and spikes. Paired Wilcoxon signed-rank tests over (fold, seed) pairs, Holm-corrected over all contrasts × conditions within each protocol and training regime. `meas. acc.` = accuracy per 4-s measurement (mean of window probabilities).

## Protocol `real_cv`

10 fold(s) × 2 seeds = 20 runs per model; settings ['N15_M07_F10']; 15 epochs. Reference (Lessmeier et al., different features and classifiers): CART / RF / ensemble 98.3% (Table 11; the running text states 98.5%).

### Training noise: lowpass

| Model | meas. acc. (clean) | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 41.3 ± 13.6 | 36.8 ± 14.3 | 37.6 ± 15.4 | 35.7 ± 15.0 | 31.4 ± 10.8 | 23.9 ± 11.8 | 37.4 ± 14.7 | 31.0 ± 11.7 | 19.8 ± 6.3 | 19.3 ± 4.8 | 36.1 ± 14.5 | 32.0 ± 12.3 | 24.6 ± 13.1 | 22.2 ± 10.0 | 36.6 ± 14.1 |
| Standard 1D-CNN + fixed filter | 41.0 ± 13.4 | 36.7 ± 13.2 | 36.9 ± 12.8 | 34.6 ± 11.6 | 29.8 ± 12.7 | 20.5 ± 7.8 | 37.7 ± 13.8 | 29.1 ± 9.4 | 21.5 ± 9.0 | 18.7 ± 4.0 | 35.5 ± 12.8 | 29.2 ± 12.0 | 21.7 ± 9.0 | 18.2 ± 2.8 | 36.7 ± 13.0 |
| PAC w/o fixed filter | 42.7 ± 16.4 | 38.1 ± 14.8 | 38.0 ± 15.9 | 35.1 ± 13.0 | 32.0 ± 11.0 | 22.3 ± 9.1 | 38.0 ± 14.9 | 30.9 ± 11.4 | 18.3 ± 4.3 | 17.3 ± 1.5 | 38.8 ± 15.1 | 33.5 ± 16.2 | 24.6 ± 13.7 | 18.8 ± 5.5 | 38.3 ± 14.9 |
| PAC-1DCNN (full) | 39.7 ± 12.2 | 37.0 ± 12.0 | 36.1 ± 13.0 | 34.2 ± 13.4 | 31.8 ± 10.6 | 22.3 ± 8.5 | 36.3 ± 12.8 | 31.5 ± 13.1 | 19.9 ± 7.2 | 18.7 ± 5.7 | 37.1 ± 14.7 | 30.7 ± 14.8 | 22.3 ± 11.5 | 17.4 ± 2.0 | 36.8 ± 11.9 |
| WDCNN-style wide-kernel CNN | 42.9 ± 17.3 | 39.9 ± 14.5 | 40.2 ± 14.9 | 39.4 ± 13.7 | 29.8 ± 10.8 | 21.6 ± 7.3 | 39.8 ± 14.6 | 39.0 ± 14.1 | 30.3 ± 10.5 | 21.8 ± 6.9 | 39.8 ± 14.4 | 39.4 ± 14.5 | 34.6 ± 13.0 | 22.6 ± 7.2 | 39.8 ± 14.5 |
| Envelope spectrum + Random Forest | 48.5 ± 17.2 | 46.4 ± 14.0 | 46.0 ± 12.8 | 43.8 ± 13.1 | 29.8 ± 7.0 | 19.4 ± 2.9 | 46.1 ± 14.4 | 32.8 ± 12.5 | 18.3 ± 4.2 | 19.4 ± 4.9 | 47.7 ± 14.7 | 47.5 ± 17.4 | 36.9 ± 17.1 | 27.1 ± 11.5 | 45.4 ± 13.5 |

Macro-F1 (%) per window unless stated.

**Contrasts** (a − b macro-F1; bold = significant after Holm):

| Contrast | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q1 filter at baseline capacity: baseline_hp − baseline | -0.0 (8/20) | -0.7 (11/20) | -1.1 (9/20) | -1.5 (8/20) | -3.4 (4/20) | +0.3 (13/20) | -1.9 (8/20) | +1.6 (7/20) | -0.5 (3/20) | -0.6 (10/20) | -2.8 (3/20) | -2.8 (5/20) | -4.0 (3/20) | +0.1 (8/20) |
| Q1 filter at PAC capacity: physics − no_residual | -1.0 (7/20) | -1.9 (6/20) | -0.9 (9/20) | -0.2 (12/20) | -0.1 (7/20) | -1.8 (9/20) | +0.6 (9/20) | +1.6 (6/20) | +1.4 (4/20) | -1.7 (7/20) | -2.8 (4/20) | -2.3 (2/20) | -1.4 (3/20) | -1.5 (6/20) |
| PAC vs standard CNN: physics − baseline | +0.3 (7/20) | -1.5 (6/20) | -1.4 (5/20) | +0.5 (8/20) | -1.6 (7/20) | -1.1 (7/20) | +0.5 (10/20) | +0.1 (5/20) | -0.6 (4/20) | +1.0 (11/20) | -1.3 (7/20) | -2.3 (2/20) | -4.8 (1/20) | +0.2 (7/20) |
| PAC vs WDCNN-style: physics − wdcnn | -2.9 (7/20) | -4.1 (6/20) | -5.2 (7/20) | +2.0 (11/20) | +0.7 (10/20) | -3.5 (10/20) | -7.5 (7/20) | -10.4 (4/20) | -3.1 (5/20) | -2.7 (10/20) | -8.7 (5/20) | -12.3 (6/20) | -5.2 (3/20) | -3.0 (7/20) |
| PAC vs classical RF: physics − rf | **-9.3 (1/20)** | **-9.9 (1/20)** | **-9.6 (1/20)** | +2.0 (9/20) | +2.9 (10/20) | **-9.8 (1/20)** | -1.3 (9/20) | +1.7 (4/20) | -0.7 (5/20) | **-10.6 (0/20)** | **-16.8 (2/20)** | **-14.6 (2/20)** | **-9.7 (1/20)** | **-8.6 (2/20)** |

Per-class recall, clean (healthy / inner / outer): Standard 1D-CNN 59/36/28; Standard 1D-CNN + fixed filter 62/36/24; PAC w/o fixed filter 57/32/36; PAC-1DCNN (full) 58/29/35; WDCNN-style wide-kernel CNN 67/24/40; Envelope spectrum + Random Forest 71/33/46

### Training noise: mixed

| Model | meas. acc. (clean) | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 41.5 ± 13.9 | 37.9 ± 13.0 | 38.9 ± 14.4 | 36.2 ± 13.9 | 30.3 ± 13.6 | 21.7 ± 10.0 | 37.5 ± 13.7 | 31.6 ± 11.0 | 17.5 ± 2.0 | 18.5 ± 4.0 | 38.7 ± 13.7 | 36.8 ± 14.9 | 25.9 ± 12.9 | 18.7 ± 5.0 | 37.9 ± 12.9 |
| Standard 1D-CNN + fixed filter | 42.7 ± 14.2 | 39.1 ± 13.7 | 38.5 ± 13.3 | 33.6 ± 11.2 | 29.4 ± 12.2 | 20.0 ± 6.3 | 38.7 ± 12.6 | 30.3 ± 11.1 | 16.8 ± 1.0 | 18.2 ± 5.5 | 39.3 ± 13.4 | 33.6 ± 13.9 | 21.0 ± 8.6 | 19.6 ± 7.2 | 39.1 ± 13.6 |
| PAC w/o fixed filter | 43.2 ± 12.7 | 39.8 ± 14.4 | 38.9 ± 15.8 | 36.4 ± 15.1 | 31.9 ± 12.4 | 23.4 ± 10.5 | 38.4 ± 15.2 | 33.3 ± 14.2 | 19.0 ± 7.2 | 17.6 ± 2.2 | 39.3 ± 15.2 | 33.2 ± 15.1 | 25.6 ± 13.7 | 22.9 ± 10.9 | 39.9 ± 14.8 |
| PAC-1DCNN (full) | 44.6 ± 15.1 | 39.9 ± 14.2 | 38.9 ± 14.6 | 35.4 ± 13.6 | 30.0 ± 11.9 | 21.5 ± 6.6 | 37.9 ± 14.1 | 30.3 ± 12.2 | 18.4 ± 4.9 | 18.7 ± 6.1 | 39.3 ± 15.0 | 32.9 ± 14.9 | 23.9 ± 12.0 | 19.1 ± 6.7 | 39.5 ± 13.9 |
| WDCNN-style wide-kernel CNN | 47.2 ± 15.6 | 43.3 ± 14.8 | 43.3 ± 15.2 | 41.5 ± 14.8 | 31.9 ± 12.2 | 21.7 ± 6.3 | 43.1 ± 14.6 | 41.4 ± 13.9 | 32.2 ± 12.0 | 21.8 ± 7.3 | 43.2 ± 14.3 | 42.2 ± 13.9 | 36.7 ± 14.1 | 23.2 ± 9.0 | 43.4 ± 14.7 |
| Envelope spectrum + Random Forest | 50.2 ± 15.7 | 47.1 ± 13.7 | 47.7 ± 12.9 | 44.4 ± 14.0 | 30.2 ± 7.3 | 20.5 ± 3.4 | 45.2 ± 14.0 | 33.4 ± 12.3 | 18.0 ± 3.0 | 19.1 ± 5.0 | 48.0 ± 14.3 | 48.4 ± 16.4 | 40.5 ± 16.4 | 28.7 ± 10.8 | 45.7 ± 13.1 |

Macro-F1 (%) per window unless stated.

**Contrasts** (a − b macro-F1; bold = significant after Holm):

| Contrast | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q1 filter at baseline capacity: baseline_hp − baseline | +1.2 (13/20) | -0.4 (11/20) | -2.6 (8/20) | -0.9 (8/20) | -1.7 (8/20) | +1.2 (11/20) | -1.4 (9/20) | -0.7 (3/20) | -0.3 (2/20) | +0.6 (12/20) | -3.2 (8/20) | -4.9 (5/20) | +0.9 (3/20) | +1.3 (13/20) |
| Q1 filter at PAC capacity: physics − no_residual | +0.1 (8/20) | +0.0 (8/20) | -1.0 (7/20) | -2.0 (9/20) | -1.9 (6/20) | -0.5 (10/20) | -3.0 (6/20) | -0.5 (3/20) | +1.2 (3/20) | +0.0 (9/20) | -0.4 (7/20) | -1.7 (4/20) | -3.8 (2/20) | -0.4 (9/20) |
| PAC vs standard CNN: physics − baseline | +2.0 (12/20) | +0.0 (9/20) | -0.9 (5/20) | -0.3 (12/20) | -0.2 (10/20) | +0.4 (11/20) | -1.4 (6/20) | +0.9 (3/20) | +0.3 (4/20) | +0.6 (11/20) | -3.9 (5/20) | -2.0 (3/20) | +0.4 (4/20) | +1.7 (12/20) |
| PAC vs WDCNN-style: physics − wdcnn | -3.4 (7/20) | -4.3 (7/20) | -6.1 (7/20) | -1.9 (8/20) | -0.2 (6/20) | -5.2 (7/20) | -11.1 (6/20) | **-13.8 (3/20)** | -3.0 (5/20) | -3.9 (7/20) | -9.4 (4/20) | -12.9 (4/20) | -4.1 (6/20) | -3.9 (6/20) |
| PAC vs classical RF: physics − rf | **-7.2 (1/20)** | **-8.8 (0/20)** | **-9.0 (2/20)** | -0.2 (9/20) | +1.0 (9/20) | **-7.3 (0/20)** | -3.1 (7/20) | +0.5 (2/20) | -0.4 (3/20) | **-8.7 (2/20)** | **-15.6 (0/20)** | **-16.6 (2/20)** | **-9.6 (2/20)** | **-6.2 (2/20)** |

Per-class recall, clean (healthy / inner / outer): Standard 1D-CNN 60/37/26; Standard 1D-CNN + fixed filter 63/36/28; PAC w/o fixed filter 57/37/35; PAC-1DCNN (full) 59/40/32; WDCNN-style wide-kernel CNN 67/30/42; Envelope spectrum + Random Forest 70/35/47

## Protocol `a2r`

1 fold(s) × 20 seeds = 20 runs per model; settings ['N15_M07_F10']; 15 epochs. Reference (Lessmeier et al., different features and classifiers): ensemble 75.0%, RF 64.1% (Table 9).

### Training noise: lowpass

| Model | meas. acc. (clean) | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 39.3 ± 4.3 | 30.1 ± 3.8 | 30.9 ± 5.0 | 31.7 ± 8.2 | 26.8 ± 7.8 | 20.8 ± 6.7 | 29.0 ± 3.3 | 27.4 ± 4.0 | 26.7 ± 3.7 | 25.0 ± 3.3 | 28.1 ± 3.5 | 27.4 ± 2.1 | 26.7 ± 3.7 | 22.0 ± 2.7 | 30.2 ± 3.8 |
| Standard 1D-CNN + fixed filter | 39.4 ± 3.2 | 30.9 ± 4.5 | 31.8 ± 4.5 | 33.1 ± 4.4 | 33.3 ± 4.7 | 26.0 ± 5.7 | 30.3 ± 4.4 | 30.8 ± 4.2 | 28.5 ± 3.5 | 24.6 ± 3.6 | 27.4 ± 3.5 | 28.2 ± 3.5 | 27.2 ± 3.9 | 22.0 ± 2.8 | 31.0 ± 4.5 |
| PAC w/o fixed filter | 40.1 ± 5.0 | 33.1 ± 4.4 | 34.9 ± 4.8 | 36.5 ± 6.5 | 30.2 ± 8.5 | 20.9 ± 9.8 | 32.5 ± 4.4 | 30.4 ± 5.1 | 25.4 ± 3.5 | 21.9 ± 4.4 | 29.3 ± 3.7 | 26.4 ± 1.2 | 25.2 ± 2.8 | 22.3 ± 3.0 | 33.6 ± 4.4 |
| PAC-1DCNN (full) | 42.0 ± 3.6 | 33.0 ± 5.6 | 34.6 ± 6.0 | 36.4 ± 7.3 | 32.0 ± 5.9 | 22.0 ± 6.4 | 32.0 ± 4.8 | 29.9 ± 4.5 | 26.1 ± 2.6 | 23.5 ± 3.1 | 28.7 ± 3.3 | 26.5 ± 1.5 | 25.5 ± 2.4 | 21.3 ± 1.3 | 33.3 ± 5.7 |
| WDCNN-style wide-kernel CNN | 43.6 ± 7.2 | 30.1 ± 3.3 | 30.0 ± 3.3 | 29.9 ± 3.7 | 27.5 ± 4.4 | 14.1 ± 6.7 | 29.9 ± 3.4 | 28.7 ± 4.2 | 25.5 ± 5.0 | 18.2 ± 8.2 | 30.0 ± 3.3 | 29.2 ± 3.7 | 26.9 ± 5.0 | 23.2 ± 5.9 | 30.1 ± 3.1 |
| Envelope spectrum + Random Forest | 58.3 ± 1.4 | 39.6 ± 0.9 | 37.0 ± 1.0 | 35.1 ± 1.0 | 34.6 ± 0.9 | 23.5 ± 1.3 | 37.5 ± 0.8 | 36.4 ± 0.9 | 37.5 ± 1.4 | 33.0 ± 2.2 | 38.7 ± 0.9 | 38.4 ± 0.9 | 37.2 ± 0.9 | 33.4 ± 1.2 | 39.0 ± 0.8 |

Macro-F1 (%) per window unless stated.

**Contrasts** (a − b macro-F1; bold = significant after Holm):

| Contrast | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q1 filter at baseline capacity: baseline_hp − baseline | +0.8 (11/20) | +1.0 (10/20) | +1.4 (10/20) | +6.5 (18/20) | +5.2 (16/20) | +1.3 (12/20) | +3.4 (15/20) | +1.8 (13/20) | -0.4 (10/20) | -0.7 (11/20) | +0.8 (12/20) | +0.4 (10/20) | -0.0 (5/20) | +0.8 (11/20) |
| Q1 filter at PAC capacity: physics − no_residual | -0.1 (11/20) | -0.3 (11/20) | -0.1 (9/20) | +1.7 (12/20) | +1.2 (10/20) | -0.5 (8/20) | -0.5 (11/20) | +0.8 (9/20) | +1.6 (12/20) | -0.7 (8/20) | +0.1 (12/20) | +0.3 (13/20) | -0.9 (4/20) | -0.2 (9/20) |
| PAC vs standard CNN: physics − baseline | +2.9 (13/20) | +3.7 (14/20) | +4.7 (13/20) | +5.2 (15/20) | +1.2 (10/20) | +2.9 (16/20) | +2.5 (13/20) | -0.5 (6/20) | -1.6 (7/20) | +0.6 (13/20) | -0.9 (8/20) | -1.2 (5/20) | -0.6 (5/20) | +3.1 (14/20) |
| PAC vs WDCNN-style: physics − wdcnn | +3.0 (11/20) | +4.6 (16/20) | +6.5 (14/20) | +4.4 (16/20) | +8.0 (17/20) | +2.1 (11/20) | +1.2 (9/20) | +0.6 (12/20) | +5.3 (13/20) | -1.3 (7/20) | -2.7 (2/20) | -1.4 (8/20) | -1.9 (7/20) | +3.3 (13/20) |
| PAC vs classical RF: physics − rf | **-6.6 (3/20)** | -2.4 (8/20) | +1.2 (12/20) | -2.6 (8/20) | -1.5 (9/20) | **-5.5 (4/20)** | **-6.5 (3/20)** | **-11.4 (0/20)** | **-9.5 (0/20)** | **-10.0 (1/20)** | **-12.0 (0/20)** | **-11.7 (0/20)** | **-12.0 (0/20)** | **-5.7 (3/20)** |

Per-class recall, clean (healthy / inner / outer): Standard 1D-CNN 17/51/30; Standard 1D-CNN + fixed filter 15/49/35; PAC w/o fixed filter 28/51/31; PAC-1DCNN (full) 24/53/30; WDCNN-style wide-kernel CNN 0/37/59; Envelope spectrum + Random Forest 0/82/45

### Training noise: mixed

| Model | meas. acc. (clean) | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 40.8 ± 3.6 | 31.9 ± 5.3 | 32.5 ± 5.8 | 31.3 ± 6.3 | 25.7 ± 7.0 | 19.0 ± 6.8 | 32.4 ± 4.8 | 32.0 ± 4.8 | 29.6 ± 6.5 | 26.3 ± 5.6 | 30.4 ± 4.6 | 29.3 ± 3.7 | 30.2 ± 3.2 | 22.9 ± 3.7 | 32.2 ± 5.2 |
| Standard 1D-CNN + fixed filter | 39.8 ± 2.7 | 31.2 ± 5.5 | 32.1 ± 4.7 | 32.8 ± 5.9 | 30.7 ± 9.2 | 20.7 ± 11.3 | 32.1 ± 4.6 | 32.7 ± 7.0 | 28.9 ± 7.6 | 25.5 ± 4.5 | 31.1 ± 5.5 | 32.7 ± 6.3 | 28.0 ± 5.4 | 22.1 ± 3.0 | 31.3 ± 5.4 |
| PAC w/o fixed filter | 40.6 ± 4.3 | 34.8 ± 4.4 | 37.0 ± 5.4 | 38.4 ± 7.6 | 31.1 ± 8.4 | 18.5 ± 9.7 | 34.7 ± 4.6 | 32.8 ± 6.8 | 27.4 ± 6.5 | 22.0 ± 5.7 | 31.2 ± 4.3 | 28.2 ± 3.1 | 26.2 ± 3.5 | 21.5 ± 1.2 | 35.5 ± 4.8 |
| PAC-1DCNN (full) | 42.1 ± 4.9 | 35.1 ± 5.8 | 37.1 ± 6.3 | 37.3 ± 5.8 | 28.4 ± 8.9 | 16.2 ± 10.4 | 36.0 ± 6.0 | 35.7 ± 5.5 | 28.0 ± 6.0 | 20.6 ± 6.3 | 32.6 ± 5.7 | 29.7 ± 5.3 | 25.9 ± 3.0 | 22.1 ± 2.2 | 35.2 ± 5.5 |
| WDCNN-style wide-kernel CNN | 44.9 ± 5.3 | 30.9 ± 2.5 | 30.7 ± 2.6 | 30.0 ± 3.2 | 26.4 ± 4.2 | 14.1 ± 8.4 | 30.4 ± 2.6 | 29.1 ± 3.8 | 25.7 ± 4.6 | 16.2 ± 7.6 | 30.6 ± 2.6 | 29.5 ± 3.4 | 27.7 ± 4.4 | 22.7 ± 6.6 | 30.7 ± 2.4 |
| Envelope spectrum + Random Forest | 58.8 ± 2.1 | 39.3 ± 0.8 | 36.8 ± 0.9 | 35.0 ± 0.8 | 34.6 ± 1.2 | 23.6 ± 1.6 | 37.5 ± 0.8 | 36.3 ± 0.8 | 37.8 ± 1.4 | 32.6 ± 3.1 | 39.2 ± 0.9 | 39.3 ± 0.9 | 37.9 ± 0.8 | 33.7 ± 1.0 | 38.7 ± 0.7 |

Macro-F1 (%) per window unless stated.

**Contrasts** (a − b macro-F1; bold = significant after Holm):

| Contrast | clean | lowpass@10dB | lowpass@5dB | lowpass@0dB | lowpass@-5dB | white@10dB | white@5dB | white@0dB | white@-5dB | highpass@10dB | highpass@5dB | highpass@0dB | highpass@-5dB | drift+spikes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q1 filter at baseline capacity: baseline_hp − baseline | -0.7 (9/20) | -0.4 (11/20) | +1.5 (12/20) | +4.9 (14/20) | +1.7 (12/20) | -0.3 (13/20) | +0.7 (13/20) | -0.7 (11/20) | -0.9 (6/20) | +0.8 (16/20) | +3.5 (17/20) | -2.2 (9/20) | -0.8 (4/20) | -0.9 (9/20) |
| Q1 filter at PAC capacity: physics − no_residual | +0.3 (9/20) | +0.1 (9/20) | -1.1 (8/20) | -2.6 (6/20) | -2.3 (9/20) | +1.3 (13/20) | +2.9 (13/20) | +0.6 (15/20) | -1.4 (9/20) | +1.4 (11/20) | +1.5 (12/20) | -0.3 (9/20) | +0.6 (7/20) | -0.3 (8/20) |
| PAC vs standard CNN: physics − baseline | +3.2 (13/20) | +4.6 (12/20) | +6.0 (15/20) | +2.7 (11/20) | -2.8 (7/20) | +3.6 (16/20) | +3.6 (13/20) | -1.6 (9/20) | -5.8 (4/20) | +2.2 (13/20) | +0.4 (11/20) | **-4.3 (3/20)** | -0.9 (7/20) | +3.0 (14/20) |
| PAC vs WDCNN-style: physics − wdcnn | +4.2 (16/20) | **+6.5 (17/20)** | **+7.3 (17/20)** | +2.0 (12/20) | +2.1 (12/20) | **+5.6 (16/20)** | **+6.6 (19/20)** | +2.3 (13/20) | +4.4 (12/20) | +2.0 (14/20) | +0.1 (10/20) | -1.8 (5/20) | -0.6 (8/20) | +4.5 (16/20) |
| PAC vs classical RF: physics − rf | -4.3 (6/20) | +0.4 (11/20) | +2.3 (11/20) | -6.1 (5/20) | -7.4 (5/20) | -1.6 (8/20) | -0.6 (11/20) | **-9.8 (0/20)** | **-12.1 (0/20)** | **-6.7 (5/20)** | **-9.6 (3/20)** | **-12.0 (0/20)** | **-11.6 (0/20)** | -3.4 (7/20) |

Per-class recall, clean (healthy / inner / outer): Standard 1D-CNN 25/51/31; Standard 1D-CNN + fixed filter 23/49/33; PAC w/o fixed filter 31/50/32; PAC-1DCNN (full) 34/52/31; WDCNN-style wide-kernel CNN 0/42/54; Envelope spectrum + Random Forest 0/83/44

