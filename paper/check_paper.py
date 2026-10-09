"""
Consistency checks for paper/main.tex (run after make_paper_assets.py and a LaTeX build):
citations vs. references.bib, \\res{} keys vs. generated/numbers.tex, labels vs. references,
missing figure/table files, generated assets older than the results, TBD and author placeholders,
errors/warnings in the LaTeX and BibTeX logs, and fonts not embedded in main.pdf.
Exit code 1 if anything that would be wrong in a submitted paper is found; with --submission,
remaining placeholders (e.g. the author's department) also count as problems.
Usage (from the repository root):  python paper/check_paper.py [--submission]
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
read = lambda name: open(os.path.join(HERE, name), encoding="utf-8", errors="ignore").read()

tex = re.sub(r"(?<!\\)%.*", "", read("main.tex"))  # ignore LaTeX comments
bib = read("references.bib")
nums = read(os.path.join("generated", "numbers.tex"))
problems, notes = [], []

cited = {k.strip() for grp in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]*)\}", tex) for k in grp.split(",")}
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

placeholders = re.findall(r"\[[^\]]*to be (?:completed|provided)[^\]]*\]", tex, re.I)
if placeholders:
    msg = f"placeholders to complete before submission: {placeholders}"
    (problems if "--submission" in sys.argv else notes).append(msg)

# every \tabinput / \includegraphics target must exist
files = re.findall(r"\\tabinput\{([^}]*)\}", tex)
files += re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
missing = [f for f in files if not os.path.exists(os.path.join(HERE, f))]
if missing:
    problems.append(f"included files that do not exist: {missing}")

# generated assets must be at least as new as the results they are built from
results = glob.glob(os.path.join(HERE, "..", "results", "*.json"))
generated = os.path.join(HERE, "generated", "numbers.tex")
if results and os.path.exists(generated):
    newer = [os.path.basename(r) for r in results if os.path.getmtime(r) > os.path.getmtime(generated)]
    if newer:
        problems.append(f"results newer than generated/numbers.tex (rerun make_paper_assets.py): {newer}")

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


def fonts_without_file(resources, seen):
    """Base names of non-Type3 fonts without an embedded font file, including fonts inside
    form XObjects (e.g. the matplotlib figures)."""
    out = set()
    if resources is None:
        return out
    resources = resources.get_object()
    for ref in (resources.get("/Font") or {}).values():
        font = ref.get_object()
        if id(font) in seen:
            continue
        seen.add(id(font))
        desc = font.get("/FontDescriptor")
        if font.get("/Subtype") == "/Type0":
            desc = font["/DescendantFonts"][0].get_object().get("/FontDescriptor")
        desc = desc.get_object() if desc is not None else None
        if font.get("/Subtype") != "/Type3" and (desc is None or not any(
                k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3"))):
            out.add(str(font.get("/BaseFont")))
    for ref in (resources.get("/XObject") or {}).values():
        xobj = ref.get_object()
        if xobj.get("/Subtype") == "/Form":
            out |= fonts_without_file(xobj.get("/Resources"), seen)
    return out


pdf = os.path.join(HERE, "main.pdf")
if os.path.exists(pdf):
    try:
        from pypdf import PdfReader
        seen, not_embedded = set(), set()
        for page in PdfReader(pdf).pages:
            not_embedded |= fonts_without_file(page.get("/Resources"), seen)
        if not_embedded:
            problems.append(f"fonts not embedded in main.pdf: {sorted(not_embedded)}")
        else:
            notes.append("all fonts in main.pdf (including figures) are embedded")
    except ImportError:
        notes.append("pypdf not installed: font embedding not checked")

print("PROBLEMS:" if problems else "No problems found.")
for p in problems:
    print("  -", p)
for n in notes:
    print("  note:", n)
sys.exit(1 if problems else 0)
