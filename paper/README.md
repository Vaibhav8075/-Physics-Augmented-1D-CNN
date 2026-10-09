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
decimated to 12 kHz (see the main README). Values quoted from Lessmeier et al. (Tables 9
and 11) are typed into `main.tex` by hand and were checked against the published PDF
(Table 11 gives 98.3%, its running text 98.5%).

Open items for submission: the author's school/department (placeholder in the author
block; `check_paper.py --submission` fails until it is filled in), and the target venue
with its template, page limit and review mode (the manuscript is currently 10 pages plus
two references on an 11th; an anonymous author block is needed for double-blind review).

The earlier drafts in the repository root (`IEEE_Research_Paper_Physics_1DCNN.*`) contain
claims these results do not support and are superseded by this manuscript.
