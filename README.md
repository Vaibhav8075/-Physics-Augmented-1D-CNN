# Physics-Augmented 1D-CNN for Bearing Fault Diagnosis: an Honest Evaluation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-green.svg)](https://onnxruntime.ai/)

A 1D-CNN with a fixed "physics" residual filter, dual-stream convolutions, sensor-temporal attention and post-hoc temperature scaling (**PAC-1DCNN**), evaluated on the [CWRU bearing data](https://engineering.case.edu/bearingdatacenter) and the [Paderborn University (KAt) bearing data](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter) against a standard CNN, a WDCNN-style wide-kernel CNN and an envelope-spectrum Random Forest, under evaluation protocols of increasing strictness. The manuscript in [`paper/`](paper/) is generated from these results.

---

## Status and key findings

Earlier versions of this README reported 87.58% cross-load accuracy, 0% false alarms and gains from every component. Those numbers came from a split that leaked validation data into training and could not be reproduced. All results below are regenerated from the scripts in this repository (5 seeds each; reports in `results/`).

1. **The evaluation protocol decides the result.** On CWRU each faulty bearing is recorded at all four motor loads, so splitting by load still tests on bearings seen in training (see Hendriks et al., MSSP 2022; Abburi et al., arXiv 2023; Vieira et al., MSSP 2026). Under leave-one-load-out every model scores ~100% macro-F1 on clean data. When whole defect sizes are held out (unseen bearings), every model drops to **~45–60% macro-F1** and labels most unseen inner- and outer-race faults as ball faults.
2. **No statistically confirmed robustness gain for PAC-1DCNN.** Under leave-one-load-out, PAC-1DCNN was ahead of the standard CNN at 0 dB (+4.0 macro-F1, 17/20 runs) and −5 dB (+16.4, 16/20 runs) of low-pass noise (Wilcoxon p = 0.003 each), but neither survives Holm correction over the 24 comparisons of that benchmark (adjusted p = 0.065 and 0.073). Under the fault-size split, none of 28 PAC-vs-CNN comparisons is significant after Holm correction.
3. **The fixed residual filter shows no significant effect.** Its direction depends on the noise spectrum: it tends to help with low-pass noise (the kind the training augmentation uses, and which the filter removes) and to hurt with white or high-pass noise. Not significant after correction.
4. **Simpler baselines do as well or better.** A WDCNN-style CNN beats PAC-1DCNN by ~18–26 macro-F1 under high-pass noise at 0 dB (15/15 runs). An envelope-spectrum Random Forest beats it on clean and moderate-noise data when trained on mixed noise, driven mainly by one fold.
5. **On Paderborn, where every test bearing is a different physical specimen, all networks are close to chance.** Clean macro-F1 is 37–43% (real-damage cross-validation; random guessing 33.3%) and 30–35% (artificial→real damage; random guessing 30.4%). The same envelope-spectrum features separate the bearings almost perfectly (99.9% macro-F1) when windows of the same bearings appear in training and test, but reach 50% when bearings are held out. So the models learn bearing identity, not transferable fault features. The fixed filter is not significant in any of 112 Paderborn comparisons, and the Random Forest has the highest clean scores. Our measurement-level accuracy (best 50.2%) is far below the 98.3% that Lessmeier et al. (2016, Table 11) report with different features and classifiers on the same bearing split.
6. **Temperature scaling does not fix calibration under noise.** The calibration data is classified 100% correctly, so the fitted temperature cannot anticipate errors on noisy or unseen data. This matches the known limitation of temperature scaling under dataset shift (Ovadia et al., NeurIPS 2019).

**Limitations:** CWRU has a single healthy bearing, so the healthy class is split by time and still leaks; with one bearing per fault type and size, the fault-size split has only three test bearings per fault type. The noise is synthetic. The WDCNN-style layer sizes were not checked against the original paper's architecture table. The Paderborn study uses one operating setting, decimates the signals to 16 kHz and classifies 0.128-s windows; we did not reproduce the features of Lessmeier et al.

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
                     Sensor (DE/FE) + temporal attention
                                          ▼
                    Pooling → 4-class logits → logits / T
```

Classes: Normal, Inner Race, Ball, Outer Race. T is fitted post hoc by minimizing NLL on held-out data. The `use_residual_filter`, `use_dual_stream` and `use_attention` flags in `models.py` switch components off for ablations.

---

## Evaluation protocols and results

### 1. Leave-one-load-out (`run_benchmark.py`)
Train on three loads, test on the fourth; all fault sizes (0.007″, 0.014″, 0.021″); 4 folds × 5 seeds. **Leaks bearing identity across loads.**

| Macro-F1 (%) | clean | 5 dB | 0 dB | −5 dB |
| :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 100.0 | 99.9 | 93.4 ± 5.8 | 31.0 ± 15.5 |
| PAC-1DCNN | 99.9 | 99.7 | 97.4 ± 3.4 | 47.4 ± 17.7 |

Full table, ablations and calibration: [`results/benchmark_report.md`](results/benchmark_report.md).

### 2. Fault-size hold-out (`run_robustness_study.py`)
Each fold holds out one defect size (unseen bearings) across all loads; healthy data split by time. Test noise is low-pass, white or high-pass at 10/5/0/−5 dB; models are trained on low-pass or mixed-spectrum noise; 3 folds × 5 seeds; paired Wilcoxon with Holm correction.

| Clean macro-F1 (%), mixed-noise training | |
| :--- | :--- |
| Standard 1D-CNN | 45.2 ± 6.9 |
| Standard 1D-CNN + fixed filter | 45.8 ± 8.5 |
| PAC-1DCNN w/o fixed filter | 47.2 ± 10.1 |
| PAC-1DCNN | 48.3 ± 11.3 |
| WDCNN-style wide-kernel CNN | 51.6 ± 9.9 |
| Envelope spectrum + Random Forest | 60.4 ± 23.4 |

Full tables and all contrasts: [`results/robustness_report.md`](results/robustness_report.md); plot: `results/robustness_noise_spectra.png`.

### 3. Paderborn bearing-disjoint protocols (`run_paderborn_study.py`)
The two protocols of Lessmeier et al. (2016): **P4** real-damage cross-validation (3 of 5 bearings per class for training, the other 2 for testing; 10 folds × 2 seeds) and **P3** training on artificial damage and testing on real damage (one split, 20 seeds). Vibration signal decimated 64 → 16 kHz, setting N15_M07_F10, classes healthy / inner ring / outer ring; same noise conditions, models and statistics as above.

| Clean, mixed-noise training | P4 macro-F1 | P4 meas. acc. | P3 macro-F1 | P3 meas. acc. |
| :--- | :--- | :--- | :--- | :--- |
| Standard 1D-CNN | 37.9 ± 13.0 | 41.5 | 31.9 ± 5.3 | 40.8 |
| Standard 1D-CNN + fixed filter | 39.1 ± 13.7 | 42.7 | 31.2 ± 5.5 | 39.8 |
| PAC-1DCNN w/o fixed filter | 39.8 ± 14.4 | 43.2 | 34.8 ± 4.4 | 40.6 |
| PAC-1DCNN | 39.9 ± 14.2 | 44.6 | 35.1 ± 5.8 | 42.1 |
| WDCNN-style wide-kernel CNN | 43.3 ± 14.8 | 47.2 | 30.9 ± 2.5 | 44.9 |
| Envelope spectrum + Random Forest | 47.1 ± 13.7 | 50.2 | 39.3 ± 0.8 | 58.8 |
| *Random guessing / majority class* | *33.3* | *33.3* | *30.4* | *45.5* |

"meas. acc." = accuracy per 4-s measurement (mean window probabilities), the unit used by Lessmeier et al., who report 98.3% (P4, Table 11) and 62.3–75.0% (P3, Table 9) with their own features. `paderborn_sanity_check.py` is the control: the Random Forest scores 99.9% on a random (leaky) window split of the same bearings and 50.2% on the bearing-disjoint folds. Full tables: [`results/paderborn_report.md`](results/paderborn_report.md).

### 4. Original fixed split (`train_and_evaluate.py`, `run_ablation_study.py`)
Train 0–1 HP, validate 2 HP, test 3 HP, 0.007″ faults only. Both models reach 85.8% accuracy; every 3 HP ball-fault window is classified as inner race. These results were produced before the per-batch GPU noise augmentation was introduced and have not been regenerated.

---

## Edge inference

Measured on an Intel desktop CPU (ONNX Runtime, 1 thread, batch 1, 1,000 timed runs) on the fixed-split model; see [`results/edge_deployment_report.md`](results/edge_deployment_report.md).

| Runtime | Size (KB) | Mean latency (ms) |
| :--- | :--- | :--- |
| PyTorch FP32 | 401.1 | 1.69 |
| ONNX Runtime FP32 | 386.8 | 0.42 |
| ONNX Runtime INT8 (static QDQ) | 132.5 | 0.29 |
| ONNX Runtime INT8 (dynamic) | 120.4 | 5.21 |

INT8 did not change test accuracy. No Raspberry Pi measurements have been made yet; run `rpi_edge_diagnostic.py` on the board to obtain them.

---

## Dashboard and prognostics demo

```bash
streamlit run app.py
```
Shows the drive-end/fan-end waveforms, the residual filter output and the attention maps for impaired test windows. `prognostics_engine.py` is a **simulation** of a health-index/RUL trajectory and has not been validated on run-to-failure data.

---

## Repository structure
```
industrial_fault_ai/
├── config.py                 # paths, splits, impairment settings, seeds
├── common.py                 # impairments (incl. noise spectra), training, calibration, metrics
├── models.py                 # standard CNN, WDCNN-style CNN, PAC-1DCNN (with ablation flags)
├── download_data.py          # 40 CWRU recordings (normal + 0.007/0.014/0.021" faults, loads 0-3)
├── preprocess_data.py        # windowing; fixed split + per-load arrays
├── train_and_evaluate.py     # fixed split, multi-seed
├── run_ablation_study.py     # component ablation on the fixed split
├── run_benchmark.py          # leave-one-load-out benchmark + noise sweep
├── run_robustness_study.py   # fault-size hold-out, noise spectra, stronger baselines
├── download_paderborn.py     # 29 Paderborn bearings (needs 7-Zip or bsdtar to unpack .rar)
├── preprocess_paderborn.py   # vibration channel, decimation 64 -> 16 kHz
├── run_paderborn_study.py    # Paderborn protocols P3/P4, noise spectra, all baselines
├── paderborn_sanity_check.py # leaky vs bearing-disjoint control, chance levels
├── paper/                    # IEEE manuscript generated from results/ (see paper/README.md)
├── export_edge.py            # ONNX FP32 / INT8 export and latency benchmark
├── rpi_edge_diagnostic.py    # standalone ONNX inference script for a Raspberry Pi
├── app.py                    # Streamlit dashboard
├── prognostics_engine.py     # RUL simulation (demo only)
├── tests/                    # data-split, impairment and model tests
└── results/                  # metrics JSON, reports, figures, models
```
The IEEE paper drafts (`IEEE_Research_Paper_Physics_1DCNN.*`) predate these results and still contain the old, unsupported claims; they are superseded by `paper/main.tex`.

---

## Quick start

### 1. Clone and install
```bash
git clone https://github.com/Vaibhav8075/-Physics-Augmented-1D-CNN.git
cd -- -Physics-Augmented-1D-CNN
pip install -r requirements.txt
```

### 2. Download and preprocess the data
```bash
python download_data.py      # 40 CWRU recordings: normal + 0.007"/0.014"/0.021" faults, loads 0-3 HP
python preprocess_data.py    # fixed split (0.007" only) + per-load arrays for the benchmarks
pytest tests                 # data-split, impairment and model checks
```

### 3. Run the evaluations
```bash
python run_robustness_study.py # fault-size (bearing-wise) split, 3 noise spectra, all baselines (~1.5 h on GPU)
python run_benchmark.py        # leave-one-load-out benchmark, all fault sizes, noise sweep (~1.5 h on GPU)
python train_and_evaluate.py   # fixed split: train 0-1 HP, validate 2 HP, test 3 HP (5 seeds)
python run_ablation_study.py   # component ablation on the fixed split (5 seeds)
```
For Paderborn (data licence CC BY-NC 4.0, non-commercial use only):
```bash
python download_paderborn.py   # downloads and unpacks 29 bearings into paderborn_data/raw/
python preprocess_paderborn.py # decimated arrays in paderborn_data/processed/
python run_paderborn_study.py  # P4 then P3 (~2 h on GPU); --resume continues an interrupted run
python paderborn_sanity_check.py
python paper/make_paper_assets.py
```
`run_robustness_study.py --folds 14 21 --resume` continues an interrupted run. Results go to `results/*_metrics.json` with Markdown reports alongside.

### 4. Edge export and dashboard
```bash
python export_edge.py
streamlit run app.py
```

---

## Data and references
- Vibration data: Case Western Reserve University Bearing Data Center.
- Lessmeier, Kimotho, Zimmer & Sextro, "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification", PHM Society European Conference 3(1) (2016), doi:10.36001/phme.2016.v3i1.1577. Data: KAt-DataCenter, Paderborn University (CC BY-NC 4.0).
- Hendriks, Dumond & Knox, "Towards better benchmarking using the CWRU bearing fault dataset", *Mechanical Systems and Signal Processing* 169 (2022) 108732.
- Abburi et al., "A Closer Look at Bearing Fault Classification Approaches", [arXiv:2309.17001](https://arxiv.org/abs/2309.17001) (2023).
- Vieira, Bauler, Rosa & Silva, "Towards a more realistic evaluation of machine learning models for bearing fault diagnosis", *Mechanical Systems and Signal Processing* 258 (2026) 114640 ([arXiv:2509.22267](https://arxiv.org/abs/2509.22267)).
- Zhang et al., "A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals", *Sensors* 17(2) (2017) 425.
- Ovadia et al., "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift", NeurIPS 2019.

## Author
**Vaibhav Goel** — [vaibhav.goel0531@gmail.com](mailto:vaibhav.goel0531@gmail.com) | GitHub: [@Vaibhav8075](https://github.com/Vaibhav8075)
