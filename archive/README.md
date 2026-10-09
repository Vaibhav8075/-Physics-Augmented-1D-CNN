# Archive

Superseded material, kept for the record. Nothing here is used by the current pipeline or the
manuscript in `paper/`, and none of it should be cited.

| Item | Why it is archived |
| :--- | :--- |
| `IEEE_Research_Paper_Physics_1DCNN.md`, `.tex` | Early drafts. They report results from a split that leaked validation data into training, describe measurements that were never made (Raspberry Pi), and present simulated prognostics as results. Superseded by `paper/main.tex`. |
| `run_ablation_study.py`, `results/ablation_*` | Component ablation on the legacy fixed split (0.007 in. faults, loads 0-1 / 2 / 3), which shares bearings between training and test; computed before the CWRU healthy recordings were resampled correctly. The ablations are now part of `run_benchmark.py` (leave-one-load-out) and `run_robustness_study.py`. |
| `results/confusion_matrices_under_noise.png`, `results/patent_explainability_saliency.png` | Figures from an early version of the code; no current script produces them. |
