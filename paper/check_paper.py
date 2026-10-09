"""
Consistency checks for paper/main.tex (run after make_paper_assets.py and a LaTeX build):
citations vs. references.bib, \\res{} keys vs. generated/numbers.tex, labels vs. references,
TBD placeholders still in use, and errors/warnings in the LaTeX and BibTeX logs.
Exit code 1 if anything that would be wrong in a submitted paper is found.
Usage (from the repository root):  python paper/check_paper.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
read = lambda name: open(os.path.join(HERE, name), encoding="utf-8", errors="ignore").read()

tex = re.sub(r"(?<!\\)%.*", "", read("main.tex"))  # ignore LaTeX comments
bib = read("references.bib")
nums = read(os.path.join("generated", "numbers.tex"))
problems, notes = [], []

cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", tex) for k in grp.split(",")}
entries = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
if cited - entries:
    problems.append(f"cited but missing from references.bib: {sorted(cited - entries)}")
if entries - cited:
    notes.append(f"in references.bib but not cited (not printed by BibTeX): {sorted(entries - cited)}")

used = set(re.findall(r"\\res\{([^}]*)\}", tex))
defined = dict(re.findall(r"res@(.+?)\\endcsname\{(.*)\}$", nums, re.M))
if used - set(defined):
    problems.append(f"\\res keys with no generated value: {sorted(used - set(defined))}")
tbd = sorted(k for k in used if "TBD" in defined.get(k, ""))
if tbd:
    notes.append(f"{len(tbd)} \\res values are still TBD: {tbd}")

labels = re.findall(r"\\label\{([^}]*)\}", tex)
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", tex))
dups = sorted({l for l in labels if labels.count(l) > 1})
if dups:
    problems.append(f"duplicate labels: {dups}")
if refs - set(labels):
    problems.append(f"references to undefined labels: {sorted(refs - set(labels))}")
if set(labels) - refs:
    notes.append(f"labels never referenced: {sorted(set(labels) - refs)}")

pending = len(re.findall(r"\\pending\{", tex)) + tex.count("[PENDING")
if pending:
    notes.append(f"{pending} PENDING blocks remain in the text")

if os.path.exists(os.path.join(HERE, "main.log")):
    log = read("main.log")
    for pattern, what in [(r"^! .*", "LaTeX error"), (r"Citation `[^']+' .*undefined", "undefined citation"),
                          (r"Reference `[^']+' .*undefined", "undefined reference"),
                          (r"Missing result key \S+", "missing result key"),
                          (r"multiply defined", "multiply defined label")]:
        hits = sorted(set(re.findall(pattern, log, re.M)))
        if hits:
            problems.append(f"{what} in main.log: {hits[:10]}")
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*lines? (\d+)", log)
    big = [(float(w), l) for w, l in over if float(w) > 1.0]
    if big:
        notes.append(f"overfull hboxes > 1pt (width, source line): {big}")
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    notes.append(f"pages: {pages.group(1) if pages else 'unknown (no PDF written?)'}")
else:
    notes.append("main.log not found: LaTeX build not checked")

if os.path.exists(os.path.join(HERE, "main.blg")):
    blg = read("main.blg")
    warn = re.findall(r"^Warning--.*", blg, re.M)
    if warn:
        notes.append(f"BibTeX warnings: {warn}")

print("PROBLEMS:" if problems else "No problems found.")
for p in problems:
    print("  -", p)
for n in notes:
    print("  note:", n)
sys.exit(1 if problems else 0)
