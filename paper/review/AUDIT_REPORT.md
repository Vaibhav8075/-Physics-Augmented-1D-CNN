# Review Draft — Venue and Authorship Pending: audit report

Date: 9 October 2026. Manuscript: `paper/main.tex` (branch `feat/harder-benchmark`).
This is a review draft for discussion with the supervisor, not a submission-ready paper.

## Deliverables

| Item | Location |
| :--- | :--- |
| LaTeX source | `paper/main.tex`, `paper/references.bib`, `paper/generated/`, `paper/figures/` |
| Review PDF (12 pages) | `paper/review/PAC1DCNN_Review_Draft_2026-10-09.pdf` (copy of `paper/main.pdf`) |
| Supporting files changed | `cwru_shortcut_check.py`, `results/cwru_shortcut_check.json`, `paper/make_paper_assets.py`, `paper/check_paper.py`, `README.md`, `paper/README.md` |

Build (from the repository root): `python paper/make_paper_assets.py`, then `cd paper && latexmk -pdf main.tex`,
then `python paper/check_paper.py`. The provisional layout is `\documentclass[conference]{IEEEtran}`.

## Checks performed

| Check | Result |
| :--- | :--- |
| Regenerate all numbers, tables and figures from `results/` | Every existing value reproduced exactly; only the 14 new keys below were added. Figure files differ only in their embedded creation date. |
| Compile (latexmk: pdflatex + BibTeX) | No errors, undefined citations or references, missing result keys, or overfull boxes. Underfull-box warnings only (cosmetic). 12 pages. |
| `check_paper.py` | No problems. All fonts, including those in figures, are embedded. `--submission` fails only on the review-draft banner, as intended. |
| Unit tests (`pytest tests`) | 39 passed. |
| Visual inspection | All 12 pages rendered and read: no clipping, table overflow, broken equations or unreadable figures. |
| Method text against code | Noise kernels, drift and impulse parameters, Env-RF and Kin-RF settings, ECE bins, software versions and test counts (24/70/42/112) agree. |
| Equations (1)–(2) | Re-derived; the moving-average residual and its frequency response are correct. |

## Corrections made

No experimental result was altered, and every new number in the text is a generated `\res{}` key.

1. **Hendriks et al. 2022, checked against the published PDF.** "Common CWRU partitions lead to over-optimistic results" was vaguer than the source. It now says that splits by operating condition reuse the same faulted bearings, that the authors proposed a fault-size division, and that CNN accuracy dropped by about 45% on average under it (Sec. 5). The protocol sentence now cites Hendriks directly ("train on one defect size and test on another", Table 2, Fig. 4) instead of second-hand through Rosa et al.
2. **Smith and Randall 2015, checked against the published PDF.** Added with section or table references:
   - 48 kHz normal baseline and the nominal speeds 1772/1750 rpm used for recordings 098/099 (Table A1).
   - Slip of 1–2%, which justifies the ±3% Kin-RF tolerance (Sec. 2).
   - Rolling-element order 4.7135 = 2 × ball spin frequency 2.357 (Table 2). Our Paderborn 6203 orders agree with their SKF 6203 values (4.947, 3.053, 2 × 1.994).
   - Motor load mainly lowers shaft speed, by about 4%, and adds no radial load (Sec. 4). This is now in the P1 description.
   - Diagnosability depends more on the rig assembly than on fault size (Sec. 6.2). This is now in Related Work and Threats.
   - The vague "some records are not diagnosable" is replaced by the specifics: 12 of our 36 fault recordings are only partly diagnosable or not diagnosable (Table B2), and records 236/237 are clipped (Table 3).
3. **New finding: an interference line near 4.2 kHz.** Smith and Randall (Sec. 6.1.1) report electromagnetic interference at harmonics of about 4.2 kHz. `cwru_shortcut_check.py` now measures it in our windows.
   - With the band limit, the 4.0–4.4 kHz band has a similar absolute level in both classes (2–5 dB higher in healthy windows).
   - It holds 3.5–11.7% of a standardised healthy window's power, against at most 0.09% of a faulty one.
   - It separates the classes (separability 1.00 at every load, also under the 15 dB training noise).
   - It was not created by our preprocessing, but it marks the healthy class.
   - It is disclosed in the abstract, contributions, Data section, P2 results, Threats, Conclusion and README. **It is not removed** (see the open issues).
   - The earlier entries of `cwru_shortcut_check.json` are bit-identical after the re-run.
4. **AdamW reference.** It was cited as an arXiv preprint with the venue unverified. The arXiv record (v3) states "Published as a conference paper at ICLR 2019", so it is now cited as ICLR 2019.
5. **Wording driven by generated values.** "Almost always recognised (lowest recall 100.0%)" and "better in at most 0/15 runs" were rephrased so they read correctly for any value.
6. **Review-draft label.** The banner "Review Draft — Venue and Authorship Pending (date)" appears on every page, with a `\reviewdrafttrue` toggle. The comment on the author block notes possible anonymisation and the pending authorship decision. The author block is as you specified.
7. **Documentation.** `paper/README.md` has a status and open-items list, and now points the old drafts to `archive/`. `README.md` has a "Known remaining cue" paragraph.

## Unresolved issues

1. **Interference line (most important).** Decide whether to repeat CWRU P1/P2 with a band limit whose stop band starts below 4.0 kHz. The ~3.5 kHz fault resonance must be kept. The alternative is to keep the line as a stated limitation. A re-run needs several hours of GPU time and may change the CWRU numbers.
2. **Length.** The draft is 12 pages: 11 plus references. The venue's limit is unknown, and the paper was not shortened.
3. **Abstract.** It is exactly 250 words, the common IEEE maximum.
4. **Review mode.** If the venue is double-blind, the author block and the GitHub URL must be anonymised.
5. **Authorship and acknowledgements.** Pending; nothing was added.
6. **Repository URL.** The paper's GitHub URL opens `main`, which does not yet contain this version (branch `feat/harder-benchmark`). Merge, or cite a tagged release, before submission.
7. **Scope of reference checks.** Full text was checked for Lessmeier et al. 2016, Zhang et al. 2017, Hendriks et al. 2022 and Smith and Randall 2015. The others were checked against Crossref, arXiv or publisher records (per-entry notes in `references.bib`).
8. **Kin-RF explanation.** Why Kin-RF helps on Paderborn but not CWRU remains an untested hypothesis, and the paper says so.
9. **Rounding.** Tables print "−0.0" for tiny negative differences. This is cosmetic.
10. **Old draft and decks.** The untracked old draft (`Final_IEEE_PAC_1DCNN_Research_Paper.tex`) and the `.pptx` decks still contain unsupported claims (87.58%, Raspberry Pi results). They were not modified and should not be circulated.

## Questions to confirm with the supervisor

1. Which venue (conference or journal), and what is the deadline?
2. Which template and page limit? Do references count toward the limit, and can material move to a supplement?
3. Single- or double-blind review? If double-blind, the author block and repository link will be anonymised.
4. Authorship: will the supervisor be a co-author (and in what order, with what affiliation line), or acknowledged?
5. Should the CWRU experiments be repeated with a lower band limit before submission, or is the documented limitation acceptable?
6. Is a largely negative, evaluation-focused paper a fit for the chosen venue, and is the title and framing appropriate?
7. Does VIT or the venue require statements on funding, data licences (the Paderborn data are CC BY-NC 4.0), code availability or use of AI tools in preparing the paper?
8. May the code repository be public at submission time?
