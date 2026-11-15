#!/usr/bin/env python3
"""Draw the report's figures from the raw results, as vector PDFs.

  .venv-bench/bin/python evaluation/make_figures.py <out_dir>

Writes throughput.pdf, speedup.pdf, ttft.pdf, progression.pdf and gpu_clocks.pdf.
"""
import csv
import json
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
TESTS = {"in6k_out512": "512 output tokens", "in6k_out1024": "1024 output tokens"}
# Fixed categorical order (validated palette), with a marker per series for print and CVD.
SERIES = [
    ("seesaw_28L", "Seesaw 28/20 (ours)", "#2a78d6", "o", 2.2),
    ("tp2", "vLLM TP2", "#eb6834", "^", 1.4),
    ("seesaw_26f", "Seesaw 26/22 (ours)", "#1baf7a", "s", 1.4),
    ("pp2", "vLLM PP2", "#eda100", "v", 1.4),
    ("engine_tp_f", "Own engine, TP only", "#e87ba4", "D", 1.0),
    ("engine_pp_f", "Own engine, PP only", "#008300", "x", 1.0),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def value(config, test, n, key):
    path = os.path.join(ROOT, "results", config, f"{test}_n{n}.json")
    d = json.load(open(path))
    assert d.get("failed", 0) == 0, path
    return d[key]


def concurrency_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([str(n) for n in LEVELS])
    ax.set_xlabel("concurrency (requests in flight)")


def throughput(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
    for ax, (test, title) in zip(axes, TESTS.items()):
        for cfg, label, color, marker, lw in SERIES:
            ys = [value(cfg, test, n, "total_token_throughput") for n in LEVELS]
            ax.plot(LEVELS, ys, color=color, marker=marker, markersize=4, linewidth=lw, label=label)
        concurrency_axis(ax)
        ax.set_title(title, color=INK)
        ax.set_ylabel("total tokens/s (input + output)")
        ax.set_ylim(0, None)
    # Legend in its own band above both panels, so it never covers a line.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(os.path.join(out, "throughput.pdf"))


def speedup(out):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    width = 0.38
    xs = range(len(LEVELS))
    for i, (test, title) in enumerate(TESTS.items()):
        r = [value("seesaw_28L", test, n, "total_token_throughput")
             / max(value(b, test, n, "total_token_throughput") for b in ("tp2", "pp2")) for n in LEVELS]
        color = "#2a78d6" if i == 0 else "#eb6834"
        bars = ax.bar([x + (i - 0.5) * width for x in xs], r, width * 0.92, color=color, label=title)
        for b, v in zip(bars, r):
            if v >= 1.25:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=5.5, color=INK)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=":")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(n) for n in LEVELS])
    ax.set_xlabel("concurrency")
    ax.set_ylabel("Seesaw ÷ best vLLM baseline")
    ax.set_ylim(0.8, 1.42)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "speedup.pdf"))


def ttft(out):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for cfg, label, color, marker, lw in SERIES:
        if cfg in ("seesaw_26f", "engine_pp_f"):
            continue
        ys = [value(cfg, "in6k_out512", n, "median_ttft_ms") / 1000 for n in LEVELS]
        ax.plot(LEVELS, ys, color=color, marker=marker, markersize=4, linewidth=lw, label=label)
    concurrency_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("median time to first token (s)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ttft.pdf"))


def progression(out):
    builds = [("seesaw", "24/24"), ("seesaw_26", "26/22"), ("seesaw_26f", "26/22\n+ fusion"),
              ("seesaw_28L", "28/20 lean\n+ fusion")]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    width = 0.38
    for i, (test, title) in enumerate(TESTS.items()):
        base = value("tp2", test, 64, "total_token_throughput")
        r = [value(c, test, 64, "total_token_throughput") / base for c, _ in builds]
        color = "#2a78d6" if i == 0 else "#eb6834"
        bars = ax.bar([x + (i - 0.5) * width for x in range(len(builds))], r, width * 0.92, color=color,
                      label=title)
        for b, v in zip(bars, r):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=5.5, color=INK)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(builds)))
    ax.set_xticklabels([b for _, b in builds])
    ax.set_ylabel("÷ vLLM TP2 at c = 64")
    ax.set_ylim(0.9, 1.42)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "progression.pdf"))


def gpu_clocks(out):
    """SM clock of both GPUs over 20 minutes of the final Seesaw run (c = 64, 1024 out)."""
    path = os.path.join(ROOT, "results", "seesaw_28L", "gpu_20261105_132938.csv")
    rows = list(csv.reader(open(path)))[1:]
    t0 = None
    series = {"0": ([], []), "1": ([], [])}
    for r in rows:
        t = datetime.strptime(r[0].strip()[:19], "%Y/%m/%d %H:%M:%S")
        t0 = t0 or t
        minutes = (t - t0).total_seconds() / 60
        if 10 <= minutes <= 30:
            series[r[1].strip()][0].append(minutes - 10)
            series[r[1].strip()][1].append(float(r[2].split()[0]))
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.plot(*series["0"], color="#2a78d6", linewidth=1.0, label="GPU0 RTX A5000 (230 W)")
    ax.plot(*series["1"], color="#eb6834", linewidth=1.0, label="GPU1 RTX PRO 4000 (70 W cap)")
    ax.set_xlabel("minutes")
    ax.set_ylabel("SM clock (MHz)")
    ax.set_ylim(0, 2600)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gpu_clocks.pdf"))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "figures"
    os.makedirs(out, exist_ok=True)
    for f in (throughput, speedup, ttft, progression, gpu_clocks):
        f(out)
        print("wrote", os.path.join(out, f.__name__ + ".pdf"))
