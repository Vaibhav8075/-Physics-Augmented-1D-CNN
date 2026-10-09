"""
Generates every result-dependent number, table and figure used in paper/main.tex
from the JSON/Markdown files in results/. Nothing in the paper's results is typed
by hand: in-text numbers are \\res{key} look-ups into generated/numbers.tex.
Missing inputs (e.g. Paderborn runs not finished yet) produce "TBD" entries and a
clearly labelled placeholder figure instead of values.

Usage (from the repository root):  python paper/make_paper_assets.py
"""
import os
import re
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import config  # noqa: E402
from common import noise_kernel  # noqa: E402
from run_robustness_study import holm  # noqa: E402

GEN = os.path.join("paper", "generated")
FIG = os.path.join("paper", "figures")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

NUM = {}
TBD = r"\textit{TBD}"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
COLORS = {"baseline": "#475569", "baseline_hp": "#0ea5e9", "no_residual": "#dc2626", "physics": "#059669",
          "wdcnn": "#7c3aed", "rf": "#d97706", "no_dual": "#f97316", "no_attention": "#ca8a04"}
SHORT = {"baseline": "CNN", "baseline_hp": "CNN+HP", "no_residual": "PAC$-$HP", "physics": "PAC",
         "wdcnn": "WDCNN-s", "rf": "Env-RF", "no_dual": "PAC$-$dual", "no_attention": "PAC$-$att"}


def put(key, value, fmt="{:.1f}"):
    key = key.replace("_", "")  # keys are used inside \csname; model ids lose their underscores
    NUM[key] =fmt.format(value) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pm(mean, std):
    return f"{mean:.1f}\\,$\\pm$\\,{std:.1f}"


def write(name, text):
    with open(os.path.join(GEN, name), "w", encoding="utf-8") as fh:
        fh.write(text)


def sgn(v):
    """Signed number in math mode, so negative values get a real minus sign."""
    return f"${v:+.1f}$"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# Models: parameter counts (computed, not transcribed)
# ---------------------------------------------------------------------------
def model_table():
    from models import Baseline1DCNN, WideKernelCNN, PhysicsAugmentedCalibratedCNN
    count = lambda m: sum(p.numel() for p in m.parameters())
    specs = [
        ("baseline", "Standard 1D-CNN (3 conv layers)", lambda c, k: Baseline1DCNN(c, k)),
        ("baseline_hp", "Standard 1D-CNN with fixed HP front end", lambda c, k: Baseline1DCNN(c, k, highpass_input=True)),
        ("no_residual", "PAC-1DCNN without fixed HP front end", lambda c, k: PhysicsAugmentedCalibratedCNN(c, k, use_residual_filter=False)),
        ("physics", "PAC-1DCNN (full)", lambda c, k: PhysicsAugmentedCalibratedCNN(c, k)),
        ("wdcnn", "WDCNN-style wide-kernel CNN", lambda c, k: WideKernelCNN(c, k)),
    ]
    rows = []
    for key, name, make in specs:
        cw, pu = count(make(2, 4)), count(make(1, 3))
        put(f"params-{key}-cwru", f"{cw:,}".replace(",", "{,}"))
        rows.append(f"{name} & {SHORT[key]} & {cw:,} & {pu:,} \\\\".replace(",", "{,}"))
    rows.append("Envelope spectrum + random forest (300 trees) & Env-RF & -- & -- \\\\")
    write("tab_models.tex", "\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# Edge / latency numbers (parsed from the measured report)
# ---------------------------------------------------------------------------
def edge_numbers():
    path = os.path.join("results", "edge_deployment_report.md")
    if not os.path.exists(path):
        return
    text = open(path, encoding="utf-8").read()
    for label, key in [("ONNX Runtime FP32", "ortfp"), ("ONNX Runtime INT8 Dynamic", "ortdyn"),
                       ("ONNX Runtime INT8 Static", "ortstatic"), ("PyTorch FP32", "torchfp")]:
        m = re.search(r"\| " + re.escape(label) + r" \| \w+ \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", text)
        if m:
            put(f"edge-{key}-kb", float(m.group(1)))
            put(f"edge-{key}-ms", float(m.group(2)), "{:.2f}")
    cpu = re.search(r"Measured on `([^`]+)`", text)
    put("edge-cpu", cpu.group(1) if cpu else TBD)


# ---------------------------------------------------------------------------
# CWRU leave-one-load-out benchmark
# ---------------------------------------------------------------------------
LOLO_SNRS = ["clean", "10dB", "5dB", "0dB", "-5dB", "-10dB"]
LOLO_MODELS = ["baseline", "physics", "no_residual", "no_dual", "no_attention"]


def lolo():
    b = load(os.path.join("results", "benchmark_metrics.json"))
    if b is None:
        return
    runs = b["runs"]

    def vals(model, snr, metric="macro_f1", cal="calibrated"):
        rs = sorted((r for r in runs if r["model"] == model), key=lambda r: (r["test_load"], r["seed"]))
        return np.array([r["test"][snr][cal][metric] for r in rs])

    # Holm over all contrasts (each model vs PAC) x SNR conditions
    tests = []
    for m in LOLO_MODELS:
        if m == "physics":
            continue
        for s in LOLO_SNRS:
            a, p_ = vals(m, s), vals("physics", s)
            d = p_ - a
            p = 1.0 if np.allclose(d, 0) else float(wilcoxon(p_, a).pvalue)
            tests.append((m, s, float(d.mean()), int((d > 0).sum()), len(d), p))
    adj = holm(np.array([t[5] for t in tests]))
    sig = {}
    for (m, s, d, wins, n, p), pa in zip(tests, adj):
        sig[(m, s)] = pa < 0.05
        k = f"lolo-{m}-{s}"
        put(k + "-delta", sgn(d))
        put(k + "-absdelta", abs(d))
        put(k + "-wins", f"{wins}/{n}")
        put(k + "-p", fmt_p(p))
        put(k + "-pholm", fmt_p(pa))
    put("lolo-ntests", len(tests), "{:d}")
    put("lolo-nruns", len(vals("physics", "clean")), "{:d}")

    rows = []
    for m in LOLO_MODELS:
        cells = []
        for s in LOLO_SNRS:
            v = vals(m, s)
            put(f"lolo-{m}-{s}-f1", v.mean())
            cell = pm(v.mean(), v.std(ddof=1))
            if m != "physics" and sig[(m, s)]:
                cell += "$^\\dagger$"
            cells.append(cell)
        name = {"baseline": "Standard 1D-CNN", "physics": "PAC-1DCNN (full)", "no_residual": "PAC $-$ fixed HP filter",
                "no_dual": "PAC $-$ dual stream", "no_attention": "PAC $-$ attention"}[m]
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    write("tab_lolo.tex", "\n".join(rows) + "\n")

    # calibration under LOLO
    cal_rows = []
    for m in LOLO_MODELS:
        rs = [r for r in runs if r["model"] == m]
        T = np.mean([r["temperature"] for r in rs])
        acc = np.mean([r["calib_accuracy"] for r in rs])
        put(f"lolo-{m}-T", T, "{:.2f}")
        put(f"lolo-{m}-calibacc", acc)
        cells = []
        for s in ["0dB", "-5dB"]:
            ec, eu = vals(m, s, "ece").mean(), vals(m, s, "ece", "uncalibrated").mean()
            put(f"lolo-{m}-{s}-ece-cal", ec)
            put(f"lolo-{m}-{s}-ece-uncal", eu)
            cells += [f"{eu:.1f}", f"{ec:.1f}"]
        name = {"baseline": "Standard 1D-CNN", "physics": "PAC-1DCNN", "no_residual": "PAC $-$ HP",
                "no_dual": "PAC $-$ dual", "no_attention": "PAC $-$ att."}[m]
        cal_rows.append(f"{name} & {T:.2f} & {acc:.1f} & " + " & ".join(cells) + " \\\\")
    write("tab_calibration.tex", "\n".join(cal_rows) + "\n")

    # false alarms at -5 dB (LOLO)
    for m in ("baseline", "physics"):
        put(f"lolo-{m}--5dB-far", vals(m, "-5dB", "false_alarm_rate").mean())
        put(f"lolo-{m}-0dB-far", vals(m, "0dB", "false_alarm_rate").mean())
    return vals


# ---------------------------------------------------------------------------
# CWRU fault-size hold-out study
# ---------------------------------------------------------------------------
FS_CONDS = ["clean", "lowpass@0dB", "lowpass@-5dB", "white@0dB", "white@-5dB", "highpass@0dB", "highpass@-5dB",
            "drift+spikes"]
FS_MODELS = ["baseline", "baseline_hp", "no_residual", "physics", "wdcnn", "rf"]
CONTRAST_LABELS = {("baseline_hp", "baseline"): "CNN+HP $-$ CNN",
                   ("physics", "no_residual"): "PAC $-$ (PAC$-$HP)",
                   ("physics", "baseline"): "PAC $-$ CNN",
                   ("physics", "wdcnn"): "PAC $-$ WDCNN-s",
                   ("physics", "rf"): "PAC $-$ Env-RF"}


def cond_key(c):
    return c.replace("@", "").replace("+", "")


def results_table(summary_means, regimes, models, conds, prefix):
    rows = []
    for regime in regimes:
        rows.append(f"\\multicolumn{{{len(conds) + 1}}}{{l}}{{\\textit{{Training noise: {regime}}}}} \\\\")
        for m in models:
            cells = []
            for c in conds:
                e = summary_means[regime][m][c]["macro_f1"]
                put(f"{prefix}-{regime}-{m}-{cond_key(c)}-f1", e["mean"])
                cells.append(pm(e["mean"], e["std"]))
            rows.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def contrast_table(contrasts, regimes, conds, prefix):
    rows = []
    for regime in regimes:
        rows.append(f"\\multicolumn{{{len(conds) + 1}}}{{l}}{{\\textit{{Training noise: {regime}}}}} \\\\")
        for (a, b), label in CONTRAST_LABELS.items():
            rs = {r["condition"]: r for r in contrasts[regime] if r["a"] == a and r["b"] == b}
            if not rs:
                continue
            cells = []
            for c in conds:
                r = rs[c]
                put(f"{prefix}-{regime}-{a}-{b}-{cond_key(c)}-delta", sgn(r["delta"]))
                put(f"{prefix}-{regime}-{a}-{b}-{cond_key(c)}-absdelta", abs(r["delta"]))
                put(f"{prefix}-{regime}-{a}-{b}-{cond_key(c)}-wins", f"{r['a_wins']}/{r['n']}")
                put(f"{prefix}-{regime}-{a}-{b}-{cond_key(c)}-pholm", fmt_p(r["p_holm"]))
                cell = f"{sgn(r['delta'])} ({r['a_wins']}/{r['n']})"
                cells.append(f"\\textbf{{{cell}}}" if r["p_holm"] < 0.05 else cell)
            rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def fault_size():
    rb = load(os.path.join("results", "robustness_metrics.json"))
    if rb is None:
        return None
    s = rb["summary"]
    regimes = list(rb["protocol"]["regimes"])
    write("tab_faultsize.tex", results_table(s["means"], regimes, FS_MODELS, FS_CONDS, "fs"))
    write("tab_faultsize_contrasts.tex", contrast_table(s["contrasts"], regimes, FS_CONDS, "fs"))

    n_tests = sum(len(s["contrasts"][r]) for r in regimes)
    pac_cnn = [r for reg in regimes for r in s["contrasts"][reg] if r["a"] == "physics" and r["b"] == "baseline"]
    filt = [r for reg in regimes for r in s["contrasts"][reg]
            if (r["a"], r["b"]) in (("baseline_hp", "baseline"), ("physics", "no_residual"))]
    put("fs-ntests-per-regime", len(s["contrasts"][regimes[0]]), "{:d}")
    put("fs-pac-cnn-ntests", len(pac_cnn), "{:d}")
    put("fs-pac-cnn-nsig", sum(r["p_holm"] < 0.05 for r in pac_cnn), "{:d}")
    put("fs-filter-ntests", len(filt), "{:d}")
    put("fs-filter-nsig", sum(r["p_holm"] < 0.05 for r in filt), "{:d}")
    put("fs-nruns", rb["summary"]["contrasts"][regimes[0]][0]["n"], "{:d}")
    put("fs-ntests-total", n_tests, "{:d}")
    rf_sig = [r for r in s["contrasts"]["mixed"] if r["a"] == "physics" and r["b"] == "rf" and r["p_holm"] < 0.05]
    put("fs-rf-nsig-mixed", len(rf_sig), "{:d}")
    if rf_sig:
        put("fs-rf-sig-min", min(-r["delta"] for r in rf_sig))
        put("fs-rf-sig-max", max(-r["delta"] for r in rf_sig))

    # per-fold clean macro-F1 and per-class recall (mixed regime)
    runs = rb["runs"]
    for regime in regimes:
        for m in FS_MODELS:
            for size in (7, 14, 21):
                v = [r["test"]["clean"]["macro_f1"] for r in runs
                     if r["regime"] == regime and r["model"] == m and r["held_out_size"] == size]
                put(f"fs-{regime}-{m}-fold{size}-clean-f1", float(np.mean(v)))
            rec = np.mean([r["test"]["clean"]["recall_per_class"] for r in runs
                           if r["regime"] == regime and r["model"] == m], axis=0)
            for name, v in zip(["normal", "inner", "ball", "outer"], rec):
                put(f"fs-{regime}-{m}-recall-{name}", v)
            for c in ("white@0dB", "highpass@0dB", "lowpass@-5dB"):
                put(f"fs-{regime}-{m}-{cond_key(c)}-far",
                    float(np.mean([r["test"][c]["false_alarm_rate"] for r in runs
                                   if r["regime"] == regime and r["model"] == m])))
    return rb


def recall_figure(rb):
    if rb is None:
        return
    runs = rb["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.9), sharey=True)
    classes = ["Normal", "Inner race", "Ball", "Outer race"]
    for ax, size in zip(axes, (7, 14, 21)):
        width = 0.13
        for i, m in enumerate(FS_MODELS):
            rec = np.mean([r["test"]["clean"]["recall_per_class"] for r in runs
                           if r["regime"] == "mixed" and r["model"] == m and r["held_out_size"] == size], axis=0)
            ax.bar(np.arange(4) + (i - 2.5) * width, rec, width, color=COLORS[m], label=SHORT[m].replace("$-$", "-"))
        ax.set_xticks(np.arange(4), classes)
        ax.set_title(f'Held-out defect size {size / 1000:.3f}"')
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].set_ylabel("Recall (%), clean test data")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(FS_MODELS), frameon=False, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    plt.savefig(os.path.join(FIG, "fig_cwru_recall.pdf"))
    plt.close()


def protocol_figure(vals_lolo, rb):
    if vals_lolo is None or rb is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    x_lolo = ["clean", "10dB", "5dB", "0dB", "-5dB"]
    for m in ("baseline", "physics"):
        mean = [vals_lolo(m, s).mean() for s in x_lolo]
        axes[0].plot(range(5), mean, marker="o", ms=3, color=COLORS[m], label=SHORT[m])
    axes[0].set_title("Leave-one-load-out\n(low-pass noise + drift + spikes)")
    means = rb["summary"]["means"]["lowpass"]
    x_fs = ["clean", "lowpass@10dB", "lowpass@5dB", "lowpass@0dB", "lowpass@-5dB"]
    for m in ("baseline", "physics", "wdcnn", "rf"):
        axes[1].plot(range(5), [means[m][c]["macro_f1"]["mean"] for c in x_fs], marker="o", ms=3,
                     color=COLORS[m], label=SHORT[m])
    axes[1].set_title("Fault-size hold-out\n(low-pass noise only)")
    for ax in axes:
        ax.set_xticks(range(5), ["clean", "10", "5", "0", "$-$5"])
        ax.set_xlabel("Test SNR (dB)")
        ax.set_ylim(0, 102)
        ax.grid(linestyle=":", alpha=0.6)
    axes[0].set_ylabel("Macro-F1 (%)")
    axes[1].legend(loc="upper right", frameon=False, ncol=2)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(FIG, "fig_cwru_protocols.pdf"))
    plt.close()


def spectra_figure():
    """Normalised power spectra of the three noise models and the residual filter's power response."""
    n, fs = 4096, config.SAMPLING_RATE_HZ
    f = np.fft.rfftfreq(n, 1 / fs)
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    for spec, color in (("lowpass", "#2563eb"), ("white", "#64748b"), ("highpass", "#dc2626")):
        H = np.abs(np.fft.rfft(noise_kernel(spec), n)) ** 2
        ax.plot(f / 1000, 10 * np.log10(H / H.max() + 1e-12), color=color, label=f"{spec} noise PSD")
        share = H[f < 1000].sum() / H.sum() * 100
        put(f"noise-{spec}-below1k", share, "{:.0f}")
    ma = np.ones(11) / 11
    Hr = np.abs(1 - np.fft.rfft(ma, n) * np.exp(-1j * 2 * np.pi * f / fs * -5)) ** 2
    ax.plot(f / 1000, 10 * np.log10(Hr / Hr.max() + 1e-12), "k--", label="residual filter $|H_r|^2$")
    put("filter-halfamp-hz", float(f[np.argmax(np.sqrt(Hr) > 0.5)]), "{:.0f}")
    put("filter-halfamp-hz-pu", float(f[np.argmax(np.sqrt(Hr) > 0.5)]) * 16000 / fs, "{:.0f}")
    ax.set_xlim(0, fs / 2000)
    ax.set_ylim(-40, 3)
    ax.set_xlabel("Frequency (kHz) at 12 kHz sampling")
    ax.set_ylabel("Normalised power (dB)")
    ax.grid(linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(FIG, "fig_noise_spectra.pdf"))
    plt.close()


# ---------------------------------------------------------------------------
# Paderborn study
# ---------------------------------------------------------------------------
PU_CONDS = ["clean", "lowpass@0dB", "lowpass@-5dB", "white@0dB", "white@-5dB", "highpass@0dB", "highpass@-5dB"]


def paderborn():
    out = {}
    for protocol in ("real_cv", "a2r"):
        res = load(os.path.join("results", f"paderborn_{protocol}_metrics.json"))
        prefix = f"pu-{protocol.replace('_', '')}"
        if res is None or "summary" not in res:
            ncol = len(PU_CONDS) + 1
            write(f"tab_pu_{protocol}.tex", f"\\multicolumn{{{ncol}}}{{c}}{{{TBD}: experiment running}} \\\\\n")
            write(f"tab_pu_{protocol}_contrasts.tex", f"\\multicolumn{{{ncol}}}{{c}}{{{TBD}: experiment running}} \\\\\n")
            put(f"{prefix}-status", "pending")
            # same keys as the complete case, explicitly marked TBD
            for regime in ("lowpass", "mixed"):
                for m in FS_MODELS:
                    put(f"{prefix}-{regime}-{m}-measacc", TBD)
                    for c in PU_CONDS:
                        put(f"{prefix}-{regime}-{m}-{cond_key(c)}-f1", TBD)
            for (a, b) in CONTRAST_LABELS:
                put(f"{prefix}-{a}-{b}-nsig", TBD)
                put(f"{prefix}-{a}-{b}-ntests", TBD)
            put(f"{prefix}-nruns", TBD)
            continue
        regimes = list(res["header"]["regimes"])
        means, contrasts = res["summary"]["means"], res["summary"]["contrasts"]
        models = [m for m in FS_MODELS if m in means[regimes[0]]]
        write(f"tab_pu_{protocol}.tex", results_table(means, regimes, models, PU_CONDS, prefix))
        write(f"tab_pu_{protocol}_contrasts.tex", contrast_table(contrasts, regimes, PU_CONDS, prefix))
        for regime in regimes:
            for m in models:
                put(f"{prefix}-{regime}-{m}-measacc", means[regime][m]["clean"]["measurement_accuracy"]["mean"])
                rec = np.mean([r["test"]["clean"]["recall_per_class"] for r in res["runs"]
                               if r["regime"] == regime and r["model"] == m], axis=0)
                for name, v in zip(["healthy", "inner", "outer"], rec):
                    put(f"{prefix}-{regime}-{m}-recall-{name}", v)
                for c in ("white@0dB", "highpass@0dB", "lowpass@-5dB"):
                    put(f"{prefix}-{regime}-{m}-{cond_key(c)}-far",
                        float(np.mean([r["test"][c]["false_alarm_rate"] for r in res["runs"]
                                       if r["regime"] == regime and r["model"] == m])))
        all_rows = [r for reg in regimes for r in contrasts[reg]]
        for (a, b) in CONTRAST_LABELS:
            rows = [r for r in all_rows if r["a"] == a and r["b"] == b]
            put(f"{prefix}-{a}-{b}-nsig", sum(r["p_holm"] < 0.05 for r in rows), "{:d}")
            put(f"{prefix}-{a}-{b}-ntests", len(rows), "{:d}")
        put(f"{prefix}-nruns", contrasts[regimes[0]][0]["n"], "{:d}")
        put(f"{prefix}-status", "complete")
        out[protocol] = (res, regimes, models)
    return out


def paderborn_figure(pu):
    path = os.path.join(FIG, "fig_paderborn.pdf")
    if "real_cv" not in pu:
        fig, ax = plt.subplots(figsize=(7.1, 1.6))
        ax.text(0.5, 0.5, "PLACEHOLDER - Paderborn experiment not finished.\nThis panel is regenerated from "
                "results/paderborn_real_cv_metrics.json by make_paper_assets.py.", ha="center", va="center")
        ax.set_axis_off()
        plt.savefig(path)
        plt.close()
        return
    res, regimes, models = pu["real_cv"]
    regime = "mixed" if "mixed" in regimes else regimes[0]
    means = res["summary"]["means"][regime]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.9), sharey=True)
    for ax, spec in zip(axes, ("lowpass", "white", "highpass")):
        cs = ["clean"] + [f"{spec}@{s}dB" for s in ("10", "5", "0", "-5")]
        for m in models:
            ax.plot(range(5), [means[m][c]["macro_f1"]["mean"] for c in cs], marker="o", ms=3, color=COLORS[m],
                    label=SHORT[m], linewidth=1.8 if m == "physics" else 1.0)
        ax.set_xticks(range(5), ["clean", "10", "5", "0", "$-$5"])
        ax.set_title(f"Test noise: {spec}")
        ax.set_xlabel("Test SNR (dB)")
        ax.set_ylim(0, 102)
        ax.grid(linestyle=":", alpha=0.6)
    axes[0].set_ylabel("Macro-F1 (%)")
    axes[2].legend(loc="lower left", frameon=False, ncol=2)
    plt.tight_layout(pad=0.3)
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------------
# Dataset facts (computed from the processed arrays where available)
# ---------------------------------------------------------------------------
def dataset_numbers():
    total = 0
    for load in config.ALL_LOADS:
        p = os.path.join(config.PROCESSED_DIR, f"load{load}.npz")
        if os.path.exists(p):
            total += len(np.load(p)["y"])
    put("cwru-windows", f"{total:,}".replace(",", "{,}") if total else TBD)
    from download_data import FILES
    put("cwru-recordings", len(FILES), "{:d}")
    from download_paderborn import BEARINGS
    put("pu-bearings", len(BEARINGS), "{:d}")


def write_numbers():
    lines = ["% Generated by paper/make_paper_assets.py -- do not edit by hand.", "\\makeatletter"]
    for k in sorted(NUM):
        lines.append(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{NUM[k]}}}")
    lines.append("\\makeatother")
    write("numbers.tex", "\n".join(lines) + "\n")
    print(f"[paper] wrote {len(NUM)} numbers, tables and figures to {GEN} and {FIG}")


if __name__ == "__main__":
    model_table()
    edge_numbers()
    vals_lolo = lolo()
    rb = fault_size()
    recall_figure(rb)
    protocol_figure(vals_lolo, rb)
    spectra_figure()
    pu = paderborn()
    paderborn_figure(pu)
    dataset_numbers()
    write_numbers()
