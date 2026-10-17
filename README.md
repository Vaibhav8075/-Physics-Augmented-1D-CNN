# Physics-Augmented 1D-CNN for Bearing Fault Diagnosis: an Honest Evaluation

[![Python](https://img.shields.io/badge/tested%20with-Python%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/tested%20with-PyTorch%202.9-red.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-green.svg)](https://onnxruntime.ai/)

A 1D-CNN with a fixed "physics" residual filter, dual-stream convolutions, channel/temporal attention and post-hoc temperature scaling (**PAC-1DCNN**), evaluated on the [CWRU bearing data](https://engineering.case.edu/bearingdatacenter) and the [Paderborn University (KAt) bearing data](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter) against a standard CNN (with a fixed or a learnable copy of the same filter), a WDCNN-style wide-kernel CNN and two Random Forests (generic envelope-spectrum features; bearing-kinematic fault-frequency features), under evaluation protocols of increasing strictness. The manuscript in [`paper/`](paper/) is generated from these results.

---

## Status and key findings

Earlier versions of this README reported 87.58% cross-load accuracy, 0% false alarms and gains from every component. Those numbers came from a split that leaked validation data into training and could not be reproduced. All results below are regenerated from the scripts in this repository (reports in `results/`).

**Data corrections (October 2026).** The four CWRU normal-baseline recordings (files 97-100) exist only at 48 kHz, while the fault recordings are 12 kHz. Earlier versions of this pipeline read them as 12 kHz; `preprocess_data.py` now decimates them. Decimating only the healthy class creates a second artefact: its anti-aliasing filter removes the 4.8-6 kHz content that the fault recordings keep, and the raw fault recordings keep their ADC quantisation steps. Both identify healthy windows (separability up to 0.99 and 1.00, `cwru_shortcut_check.py`). Every CWRU recording is therefore low-passed to a common band limit, after which both cues are at chance (at most 0.54 and 0.51). All CWRU results below use the corrected data; a version without the band limit had shown a significant leave-one-load-out advantage for PAC-1DCNN at -5 and -10 dB that disappeared after the correction.

**Interference line (fixed, October 2026).** Smith and Randall (MSSP 2015, Sec. 6.1.1) report electromagnetic interference at harmonics of about 4.2 kHz in the CWRU data. Because the healthy vibration is weak, the 4.0-4.4 kHz band holds 27-64% of a standardised healthy window's power against at most 0.8% of a faulty one, so it marks the healthy class. The first band limit (4 kHz) attenuated it by only 0-21 dB and it still separated the classes under the 15 dB training noise. The band limit now ends at 3.6 kHz (47 dB down at 4.0 kHz, 73 dB at 4.2 kHz): the line lies 61-66 dB below a healthy window's power and its separability under the training noise is at most 0.53. The cost is the 3.6-4 kHz band, which holds a median of 11.9% (inner race), 0.6% (ball) and 9.8% (outer race) of a fault window's power; the strongest fault content above 2 kHz (2.8-3.5 kHz) stays in the passband. All CWRU results were re-run with the 3.6 kHz band limit.

**Sensitivity check.** To test whether losing the 3.6-4 kHz band matters, P1 and P2 were repeated with a sharp elliptic low-pass instead (order 11, flat to 3.9 kHz, at least 120 dB down from 4.0 kHz; `CWRU_VARIANT=sharp`, results in `results/sensitivity_sharp/`). The fault windows then keep their 3.6-4 kHz content, and the interference band separates the classes neither in clean windows (at most 0.58) nor under the training noise (at most 0.53). Every mean difference keeps its sign and every conclusion below holds: PAC-1DCNN leads the CNN at 0 dB under leave-one-load-out (+13.8, significant) and is never significantly better with unseen defect sizes, and WDCNN-s and the envelope Random Forest beat it in 6 and 16 of 28 P2 conditions, as before. Two significance calls near the threshold change (PAC-1DCNN's +1.6 at 5 dB in P1 becomes significant; the filter's white-noise loss in the mixed regime does not remain so). A narrower band-stop over 4.08-4.40 kHz was tried first and did not remove the cue: the interference has components down to about 4.03 kHz.

1. **The evaluation protocol decides the result.** On CWRU each faulty bearing is recorded at all four motor loads, so splitting by load still tests on bearings seen in training (Hendriks et al., MSSP 2022; Abburi et al., arXiv 2023; Vieira et al., MSSP 2026). Under leave-one-load-out every model scores at least 99.8% clean macro-F1. When whole defect sizes are held out, the networks drop to **45-48%** and the envelope-spectrum Random Forest to 61%. Healthy windows are still recognised (the single healthy bearing leaks through the time split), but inner- and outer-race faults of an unseen size are mostly given the wrong fault type (recall 0-30% for the inner race and 16-38% for the outer race across models, mixed-noise training). Averaged over the three fault classes only (fault-class F1, which leaves out the leaking healthy class), the networks reach 29-37% and the two Random Forests 48.5% and 49.7%.
2. **PAC-1DCNN beats a standard CNN significantly only when the test bearings were seen in training.** Leave-one-load-out at 0 dB: 74.8 vs 62.8 macro-F1 (20/20 runs, Holm-adjusted p < 0.001); without its fixed filter PAC-1DCNN reaches 70.7, not significantly different from the full model, so the gain does not come from the filter. Fault-size split: 0 of 28 comparisons significant; Paderborn: none in P4, and in P3 it is significantly worse in one condition. The ablation that keeps only a single stream on the filtered signal is significantly better than the full model under low-pass noise (leave-one-load-out: 89.2 vs 74.8 at 0 dB, 41.9 vs 17.3 at -5 dB; fault-size split: 5 of 28 conditions, all low-pass) and significantly worse under white noise at 0 dB on the fault-size split (-15.8, low-pass training).
3. **The fixed residual filter tends to help only when the noise lies below its cut-off.** On the fault-size split it lowers the standard CNN's macro-F1 with white noise at 0 dB (-14.0 and -11.8 in the two training regimes): 2 of 56 front-end comparisons significant, both in the direction its frequency response predicts. The other differences point the same way without reaching significance: +6.5 and +9.0 with low-pass noise at 0 dB (13/15 runs each), -16.1 and -13.5 with high-pass noise. A learnable 11-tap version behaves like the fixed one (0 of 28 different). Inside PAC-1DCNN, whose raw-signal stream bypasses the filter, it has no significant effect, and on Paderborn none in 112 comparisons.
4. **Simpler baselines do as well or better.** On the fault-size split the WDCNN-style CNN is significantly better than PAC-1DCNN in 6 of 28 conditions (low-pass noise at 5, 0 and -5 dB; by 9-28 points) and has the highest mean of the original networks at 0 and -5 dB for every noise spectrum; the envelope-spectrum Random Forest is significantly better in 16 of 28 (by 15-42 points).
5. **A bigger pre-trained network does not help; adapting to the test data does, under strong noise.** Two models were added last and compared with the WDCNN-style CNN. An ImageNet-pre-trained ResNet-18 on spectrograms (the strongest network under the fault-size split of Hendriks et al., 2022) matches it on clean data but is significantly worse in 9 of 28 fault-size conditions, all at 0 or -5 dB, and never significantly better. The same WDCNN-style CNN with AdaBN (batch-norm statistics re-estimated on the unlabelled test windows of each condition, as in the original WDCNN) is significantly better at -5 dB under every noise spectrum: by 15-19 points on the fault-size split (6 of 28 conditions), 14-19 on Paderborn P4 (14 of 28) and 7-17 on P3 (15 of 28), and significantly worse nowhere. Its clean scores do not change significantly (45.6 vs 47.5 macro-F1 on the fault-size split), so it fixes the noise shift, not the bearing shift. It needs a batch of unlabelled data from each new condition before diagnosing.
6. **On Paderborn, where every test bearing is a different physical specimen, all networks are close to chance.** Clean macro-F1 is 37-45% (real-damage cross-validation, P4; random guessing 33.3%) and 30-35% (artificial-to-real damage, P3; random guessing 30.4%). The same envelope-spectrum features separate the bearings almost perfectly (99.9% macro-F1) when windows of the same bearings appear in training and test, but reach 50% when bearings are held out: the models learn bearing identity, not transferable fault features.
7. **Physics helped as a feature prior, not as a filter, and only on Paderborn.** The kinematic Random Forest (envelope-spectrum peaks at the bearing fault frequencies x shaft speed) has the highest macro-F1 of the original models in every 0 and -5 dB condition on both Paderborn protocols (of all models in every such P3 condition; in P4 the AdaBN network is higher in 10 of 12), is significantly better than the generic Random Forest in all 28 P3 and 8 of 28 P4 comparisons, and has the best measurement-level accuracy (52.3% P4, 62.5% P3). On CWRU's fault-size split it was significantly worse than the generic Random Forest in 5 of 28 conditions, although over the fault classes alone the two are close (49.7% and 48.5% clean fault-class F1, mixed-noise training). Lessmeier et al. (2016) report 98.3% (P4, Table 11) and 62.3-75.0% (P3, Table 9) with different features and classifiers.
8. **Statistics.** Runs that share a fold are not independent. Averaged over seeds within folds, every significant leave-one-load-out and fault-size contrast has the same sign in every fold; on Paderborn P4, no contrast survives the corrected resampled t-test of Nadeau & Bengio (2003) over the ten folds (`fold_level_stats.py`).
9. **Temperature scaling does not fix calibration under noise.** The calibration windows are classified almost without error, so the fitted temperature (0.30-0.35) makes the models more confident, and the calibration error at 0 and -5 dB increases for all five leave-one-load-out models (Ovadia et al., NeurIPS 2019, describe this limitation under dataset shift).

A 30-epoch convergence check (`training_curves.py`) shows the training loss near zero at epoch 15 for every network; test scores at epoch 30 were lower than at epoch 15 in 12 of 14 cases, so the shared 15-epoch recipe is not what holds the models back.

**Limitations:** CWRU has a single healthy bearing, so the healthy class is split by time and still leaks, and its recordings come from a separate 48 kHz acquisition; with one bearing per fault type and size, the fault-size split has only three test bearings per fault type. The noise is synthetic. The WDCNN-style network follows Table 2 of Zhang et al. (2017) but uses 1024-sample windows on CWRU; its AdaBN variant is a separate model. The pre-trained ResNet-18 shares the other networks' training recipe (learning rate 1e-3, 15 epochs) and sees 128 x 128 spectrograms rather than the 224 x 224 images of its pre-training. The Paderborn study uses one operating setting, decimates the signals to 16 kHz and classifies 0.128-s windows; we did not reproduce the features of Lessmeier et al.

---

## Model

```
                 INPUT: 2 channels (drive end + fan end), 1024 samples @ 12 kHz
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
          Fixed residual filter                       Raw signal x(t)
          r(t) = x(t) − MovingAvg11(x)                      │
          (high-pass, cut-off ≈ 660 Hz)                     │
                        │                                   │
              Stream 1: Conv1D k=5                Stream 2: Conv1D k=15
                        └─────────────────┬─────────────────┘
                                          ▼
                              Fusion Conv1D (128 ch)
                                          ▼
                    Channel (learned features) + temporal attention
                                          ▼
                    Pooling → 4-class logits → logits / T
```

Classes: Normal, Inner Race, Ball, Outer Race. T is fitted post hoc by minimizing NLL on held-out data. The `use_residual_filter`, `use_dual_stream` and `use_attention` flags in `models.py` switch components off for ablations; `Baseline1DCNN(highpass_input=True, learnable_highpass=True)` is the learnable-filter variant. The channel attention weights learned feature maps, not the physical sensors.

---

## Evaluation protocols and results

### 1. Leave-one-load-out (`run_benchmark.py`)
Train on three loads, test on the fourth; all fault sizes (0.007″, 0.014″, 0.021″); 4 folds × 5 seeds; low-pass noise plus drift and impulses at test time. **Leaks bearing identity across loads.**

| Macro-F1 (%) | clean | 5 dB | 0 dB | −5 dB | −10 dB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 99.9 ± 0.2 | 98.4 ± 3.3 | 62.8 ± 12.6 | 12.9 ± 9.1 | 10.7 ± 7.7 |
| PAC-1DCNN | 99.9 ± 0.3 | 99.2 ± 1.7 | 74.8 ± 12.2 | 17.3 ± 12.6 | 11.0 ± 8.6 |
| PAC-1DCNN w/o fixed filter | 99.9 ± 0.4 | 99.2 ± 1.8 | 70.7 ± 14.3 | 23.2 ± 14.2 | 13.9 ± 9.1 |
| PAC-1DCNN, single stream on the filtered signal | 99.9 ± 0.1 | 99.4 ± 1.2 | 89.2 ± 9.0 | 41.9 ± 16.2 | 12.5 ± 1.7 |

At 0 dB the models miss 8-37% of faulty windows (labelled healthy) and raise almost no false alarms (at most 0.03%). Full table, ablations and calibration: [`results/benchmark_report.md`](results/benchmark_report.md).

### 2. Fault-size hold-out (`run_robustness_study.py`)
Each fold holds out one defect size (unseen faulty bearings) across all loads; healthy data split by time. Test noise is low-pass, white or high-pass at 10/5/0/−5 dB; models are trained on low-pass or mixed-spectrum noise; 3 folds × 5 seeds; paired Wilcoxon with Holm correction (primary contrasts and the added models in separate families).

| Clean, mixed-noise training (%) | Macro-F1 | Fault-class F1 |
| :--- | :--- | :--- |
| Standard 1D-CNN | 47.7 ± 15.5 | 32.8 ± 18.4 |
| Standard 1D-CNN + fixed filter | 45.6 ± 10.4 | 29.3 ± 12.2 |
| Standard 1D-CNN + learnable filter | 45.1 ± 9.5 | 28.7 ± 10.7 |
| PAC-1DCNN w/o fixed filter | 45.5 ± 12.7 | 30.1 ± 14.3 |
| PAC-1DCNN | 44.8 ± 12.7 | 29.4 ± 14.2 |
| PAC-1DCNN, single stream on the filtered signal | 46.4 ± 11.6 | 30.2 ± 13.7 |
| WDCNN-style wide-kernel CNN | 47.5 ± 14.1 | 33.7 ± 15.2 |
| Envelope spectrum + Random Forest | 60.7 ± 26.1 | 48.5 ± 33.9 |
| Kinematic envelope features + Random Forest | 48.3 ± 22.2 | 49.7 ± 25.8 |
| Pre-trained ResNet-18 on spectrograms | 45.8 ± 7.8 | 29.5 ± 12.4 |
| WDCNN-style CNN + AdaBN | 45.6 ± 8.0 | 37.0 ± 12.7 |

Fault-class F1 is the mean F1 of the inner-race, ball and outer-race classes; it leaves out the healthy class, whose windows leak through the time split.

Full tables and all contrasts: [`results/robustness_report.md`](results/robustness_report.md); plot: `results/robustness_noise_spectra.png`.

### 3. Paderborn bearing-disjoint protocols (`run_paderborn_study.py`)
The two protocols of Lessmeier et al. (2016): **P4** real-damage cross-validation (3 of 5 bearings per class for training, the other 2 for testing; 10 folds × 2 seeds) and **P3** training on artificial damage and testing on real damage (one split, 20 seeds). Vibration signal decimated 64 → 16 kHz, setting N15_M07_F10, classes healthy / inner ring / outer ring; same noise conditions, models and statistics as above.

| Clean, mixed-noise training | P4 macro-F1 | P4 meas. acc. | P3 macro-F1 | P3 meas. acc. |
| :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 37.9 ± 13.0 | 41.5 | 31.9 ± 5.3 | 40.8 |
| Standard 1D-CNN + fixed filter | 39.1 ± 13.7 | 42.7 | 31.2 ± 5.5 | 39.8 |
| Standard 1D-CNN + learnable filter | 39.0 ± 13.3 | 43.6 | 30.5 ± 5.3 | 38.9 |
| PAC-1DCNN w/o fixed filter | 39.8 ± 14.4 | 43.2 | 34.8 ± 4.4 | 40.6 |
| PAC-1DCNN | 39.9 ± 14.2 | 44.6 | 35.1 ± 5.8 | 42.1 |
| PAC-1DCNN, single stream on the filtered signal | 38.6 ± 12.7 | 41.4 | 31.5 ± 5.3 | 37.7 |
| WDCNN-style wide-kernel CNN | 43.2 ± 16.7 | 45.1 | 30.2 ± 2.9 | 43.4 |
| Envelope spectrum + Random Forest | 47.1 ± 13.7 | 50.2 | 39.3 ± 0.8 | 58.8 |
| Kinematic envelope features + Random Forest | 45.7 ± 11.3 | 52.3 | 42.8 ± 0.3 | 62.5 |
| Pre-trained ResNet-18 on spectrograms | 43.5 ± 14.9 | 47.7 | 30.3 ± 5.2 | 44.3 |
| WDCNN-style CNN + AdaBN | 45.0 ± 15.4 | 46.7 | 30.6 ± 2.7 | 43.1 |
| *Random guessing / majority class* | *33.3* | *33.3* | *30.4* | *45.5* |

"meas. acc." = accuracy per 4-s measurement (mean window probabilities), the unit used by Lessmeier et al., who report 98.3% (P4, Table 11) and 62.3–75.0% (P3, Table 9) with their own features. `paderborn_sanity_check.py` is the control: the Random Forest scores 99.9% on a random (leaky) window split of the same bearings and 50.2% on the bearing-disjoint folds. Full tables: [`results/paderborn_report.md`](results/paderborn_report.md).

### 4. Demonstration model (`train_and_evaluate.py`)
The dashboard and the edge export use models trained on a fixed load split (train 0–1 HP, validate 2 HP, test 3 HP, 0.007″ faults only). This split shares bearings between training and test and is **not** an evaluation of generalisation. On it, the standard CNN reaches 77.6% and PAC-1DCNN 79.0% test accuracy (5 seeds). Both recognise ball faults poorly on this split; the dashboard shows the recall of every class. The earlier component ablation on this split is in [`archive/`](archive/).

---

## Edge inference

Measured on an Intel desktop CPU (ONNX Runtime, 1 thread, batch 1, 1,000 timed runs) with the demonstration model; see [`results/edge_deployment_report.md`](results/edge_deployment_report.md).

| Runtime | Size (KB) | Mean latency (ms) |
| :--- | :--- | :--- |
| PyTorch FP32 | 401.1 | 1.89 |
| ONNX Runtime FP32 | 386.8 | 0.43 |
| ONNX Runtime INT8 (static QDQ) | 132.5 | 0.30 |
| ONNX Runtime INT8 (dynamic) | 120.4 | 5.35 |

INT8 changed test accuracy by less than half a point (FP32 79.30%, static INT8 79.09%, dynamic INT8 78.88%; seed-0 model). No Raspberry Pi measurements have been made yet; run `rpi_edge_diagnostic.py` on the board to obtain them.

---

## Dashboard and prognostics demo

```bash
streamlit run app.py
```
Shows the drive-end/fan-end waveforms, the residual filter output and the attention maps for impaired test windows of the demonstration split. `prognostics_engine.py` is a **simulation** of a health-index/RUL trajectory and has not been validated on run-to-failure data.

---

## Repository structure
```
industrial_fault_ai/
├── config.py                 # paths, splits, impairment settings, seeds, CWRU band limit
├── common.py                 # impairments (incl. noise spectra), training, calibration, AdaBN, metrics
├── models.py                 # standard CNN (fixed/learnable filter), WDCNN-style CNN, spectrogram ResNet-18, PAC-1DCNN (ablation flags)
├── kinematic_features.py     # bearing fault-frequency envelope features (Kin-RF)
├── download_data.py          # 40 CWRU recordings (normal + 0.007/0.014/0.021" faults, loads 0-3)
├── preprocess_data.py        # 48 -> 12 kHz for the normal recordings, common 3.6 kHz band limit, windowing
├── cwru_shortcut_check.py    # checks whether preprocessing artefacts or the 4.2 kHz interference line identify the healthy class, and what the band limit removes
├── train_and_evaluate.py     # demonstration model on a fixed load split (dashboard, edge export)
├── run_benchmark.py          # leave-one-load-out benchmark + noise sweep (P1)
├── run_robustness_study.py   # fault-size hold-out, noise spectra, all baselines (P2)
├── download_paderborn.py     # 29 Paderborn bearings (needs 7-Zip or bsdtar to unpack .rar)
├── preprocess_paderborn.py   # vibration channel, decimation 64 -> 16 kHz
├── run_paderborn_study.py    # Paderborn protocols P3/P4; --add-models merges new models into saved results
├── paderborn_sanity_check.py # leaky vs bearing-disjoint control, chance levels
├── fold_level_stats.py       # fold-level agreement and corrected resampled t-test
├── training_curves.py        # 30-epoch convergence check
├── paper/                    # IEEE manuscript generated from results/ (see paper/README.md)
├── export_edge.py            # ONNX FP32 / INT8 export and latency benchmark
├── rpi_edge_diagnostic.py    # standalone ONNX inference script for a Raspberry Pi
├── app.py                    # Streamlit dashboard (demonstration model)
├── prognostics_engine.py     # RUL simulation (demo only)
├── tests/                    # data, preprocessing, feature, impairment and model tests
├── archive/                  # superseded drafts and results (see archive/README.md)
└── results/                  # metrics JSON, reports, figures, models; sensitivity_sharp/ holds the band-limit check
```

---

## Quick start

### 1. Clone and install
```bash
git clone https://github.com/Vaibhav8075/-Physics-Augmented-1D-CNN.git
cd -- -Physics-Augmented-1D-CNN
pip install -r requirements.txt        # or requirements-lock.txt for the exact tested versions
```

### 2. Download and preprocess the data
```bash
python download_data.py        # 40 CWRU recordings: normal + 0.007"/0.014"/0.021" faults, loads 0-3 HP
python preprocess_data.py      # demonstration split + per-load arrays for the benchmarks
python cwru_shortcut_check.py  # artefact check (with and without the common band limit)
pytest tests                   # data, preprocessing, impairment and model checks
```

### 3. Run the evaluations
```bash
python run_benchmark.py         # leave-one-load-out benchmark, all fault sizes, noise sweep (~25 min on GPU)
python run_robustness_study.py  # fault-size (bearing-wise) split, 3 noise spectra, all models (~2.5 h on GPU)
python train_and_evaluate.py    # demonstration model (fixed load split, 5 seeds)
```
For Paderborn (data licence CC BY-NC 4.0, non-commercial use only):
```bash
python download_paderborn.py   # downloads and unpacks 29 bearings into paderborn_data/raw/
python preprocess_paderborn.py # decimated arrays in paderborn_data/processed/
python run_paderborn_study.py  # P4 then P3, all models (~6 h on GPU); --resume continues an interrupted run
python paderborn_sanity_check.py
python fold_level_stats.py
python training_curves.py
python paper/make_paper_assets.py
```
Band-limit sensitivity check (separate data and results folders; the main results are untouched):
```bash
CWRU_VARIANT=sharp python preprocess_data.py && CWRU_VARIANT=sharp python cwru_shortcut_check.py
CWRU_VARIANT=sharp python run_benchmark.py && CWRU_VARIANT=sharp python run_robustness_study.py
```
`run_robustness_study.py --folds 14 21 --resume` continues an interrupted run. `--add-models --models resnet2d wdcnn_adabn` (both study scripts) trains only the given models and merges them into the saved results; with `--folds` the CWRU study reruns only those folds. Results go to `results/*_metrics.json` with Markdown reports alongside.

### 4. Edge export and dashboard
```bash
python export_edge.py
streamlit run app.py
```

---

## Data and references
- Vibration data: Case Western Reserve University Bearing Data Center (bearing geometry and fault frequencies from its "Bearing Information" page).
- Lessmeier, Kimotho, Zimmer & Sextro, "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification", PHM Society European Conference 3(1) (2016), doi:10.36001/phme.2016.v3i1.1577. Data: KAt-DataCenter, Paderborn University (CC BY-NC 4.0).
- Smith & Randall, "Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study", *Mechanical Systems and Signal Processing* 64-65 (2015) 100-131.
- Hendriks, Dumond & Knox, "Towards better benchmarking using the CWRU bearing fault dataset", *Mechanical Systems and Signal Processing* 169 (2022) 108732.
- Abburi et al., "A Closer Look at Bearing Fault Classification Approaches", [arXiv:2309.17001](https://arxiv.org/abs/2309.17001) (2023).
- Vieira, Bauler, Rosa & Silva, "Towards a more realistic evaluation of machine learning models for bearing fault diagnosis", *Mechanical Systems and Signal Processing* 258 (2026) 114640 ([arXiv:2509.22267](https://arxiv.org/abs/2509.22267)).
- Zhang et al., "A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals", *Sensors* 17(2) (2017) 425.
- Nadeau & Bengio, "Inference for the Generalization Error", *Machine Learning* 52(3) (2003) 239-281.
- Ovadia et al., "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift", NeurIPS 2019.

## Licence and reuse
No licence is granted at this time. All rights are reserved by the author: beyond viewing and forking on GitHub, which GitHub's Terms of Service allow for public repositories, the code, results and manuscript may not be used, copied, modified or redistributed without the author's written permission. A licence will be chosen when the accompanying paper is published. The datasets are not included in this repository and remain under their own terms (CWRU Bearing Data Center; Paderborn KAt-DataCenter, CC BY-NC 4.0).

## Author
**Vaibhav Goel** — School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology — [vaibhav.goel0531@gmail.com](mailto:vaibhav.goel0531@gmail.com) | GitHub: [@Vaibhav8075](https://github.com/Vaibhav8075)
