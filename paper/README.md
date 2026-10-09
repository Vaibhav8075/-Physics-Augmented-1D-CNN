# Paper source

IEEE conference manuscript (`\documentclass[conference]{IEEEtran}`) built from the results in `../results/`.

| File | Purpose |
| :--- | :--- |
| `main.tex` | Manuscript. Result values appear only as `\res{key}` look-ups; result tables are `\tabinput` from `generated/`. |
| `references.bib` | Bibliography. Each entry records how it was verified (Crossref, arXiv API or publisher page). |
| `make_paper_assets.py` | Reads `results/*.json` (and `results/edge_deployment_report.md`) and writes `generated/numbers.tex`, `generated/tab_*.tex` and `figures/*.pdf`. Missing results become *TBD* entries and a labelled placeholder figure. |
| `check_paper.py` | Checks citations against the bibliography, `\res` keys against generated values, labels and references, remaining *TBD*/PENDING items, and the LaTeX/BibTeX logs. Exits non-zero on problems. |
| `generated/`, `figures/` | Generated assets (committed so the paper also builds without Python, e.g. on Overleaf). |

## Build

From the repository root:

```bash
python paper/make_paper_assets.py      # regenerate numbers, tables and figures from results/
cd paper && latexmk -pdf main.tex      # pdflatex + bibtex; needs IEEEtran, algorithms, algorithmicx, pgf/tikz
cd .. && python paper/check_paper.py   # consistency checks
```

On Overleaf, upload `main.tex`, `references.bib`, `generated/` and `figures/`.

## Status

All sections are complete; `check_paper.py` reports no missing or TBD values. The
Paderborn numbers need `results/paderborn_{real_cv,a2r}_metrics.json` (from
`run_paderborn_study.py`) and `results/paderborn_sanity.json` (from
`paderborn_sanity_check.py`). All CWRU numbers, including the ranges quoted in the prose,
are generated keys; they were regenerated after the 48 kHz normal-baseline recordings were
decimated to 12 kHz and all CWRU recordings were given a common 4 kHz band limit (see the main
README). Fold-level statistics and the convergence check come from `fold_level_stats.py` and
`training_curves.py`. Values quoted from Lessmeier et al. (Tables 9
and 11) are typed into `main.tex` by hand and were checked against the published PDF
(Table 11 gives 98.3%, its running text 98.5%). Statements from Smith and Randall (2015) and
Hendriks et al. (2022), including the record numbers typed into `main.tex`, were checked against
the published PDFs; `references.bib` records which section or table supports each one.

**Review Draft — Venue and Authorship Pending.** `main.tex` prints this banner on every page
while `\reviewdrafttrue` is set; `check_paper.py --submission` reports it as a problem. The
`\documentclass[conference]{IEEEtran}` layout is provisional. Open before submission:

- target venue, its template and page limit (the draft is 11 pages plus references on a 12th;
  the abstract is at 250 words);
- review mode: for double-blind review, replace the author block and anonymise the repository URL;
- authorship and acknowledgements (the supervisor's role is not yet decided);
- the 4.2 kHz interference line that survives the CWRU band limit (main README, "Known remaining
  cue"): either repeat the CWRU experiments with a band limit below 4 kHz or keep it as the
  stated limitation;
- the repository URL points at `main`, which does not yet contain this version (branch
  `feat/harder-benchmark`).

The earlier drafts in `archive/` (`IEEE_Research_Paper_Physics_1DCNN.*`) contain claims these
results do not support and are superseded by this manuscript.
