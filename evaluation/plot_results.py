#!/usr/bin/env python3
"""Plot a serving concurrency sweep from vllm bench serve or genai-perf.

Single sweep (raw result files -> one PNG per metric + <tag>_summary.csv):

  python evaluation/plot_results.py --source vllm       "results/pp2/in6k_out512_n*.json" --tag pp2_in6k_out512
  python evaluation/plot_results.py --source genai-perf "artifacts/synthetic_n*"          --tag synthetic --gpu-rate-inr 210

Compare configs (summary CSVs written by the mode above -> one PNG per metric,
one line per config):

  python evaluation/plot_results.py --compare "results/*/plots/*_in6k_out512_summary.csv" \
      --baseline tp2 pp2 --tag in6k_out512 --out-dir results/compare

  Entries are paths, globs, or label=path. Default labels are the CSV names with
  the part shared by all of them dropped (pp2_in6k_out512 / tp2_in6k_out512 ->
  pp2 / tp2). --baseline adds a speedup plot against the best baseline at each
  concurrency.

Single mode also prints M --- the highest concurrency whose p95 TPOT still meets
--tpot-target. Pass --overlay with a second glob to draw another tool's sweep of
the same config on the same axes. All money is in rupees.
"""
import argparse
import csv
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAN = float("nan")


def ok(x):
    return isinstance(x, (int, float)) and x == x


# --------------------------------------------------------------- readers
def _pick(d, *names):
    for n in names:
        if isinstance(d, dict) and d.get(n) is not None:
            return d[n]
    return NAN


def read_vllm(path):
    """vllm bench serve --save-result JSON: flat keys, milliseconds."""
    d = json.load(open(path))
    return {
        "n": _pick(d, "max_concurrency", "num_prompts"),
        "ttft_med": _pick(d, "median_ttft_ms"),
        "ttft_p95": _pick(d, "p95_ttft_ms", "p99_ttft_ms"),
        "tpot_med": _pick(d, "median_tpot_ms"),
        "tpot_p95": _pick(d, "p95_tpot_ms", "p99_tpot_ms"),
        "e2el_med": _pick(d, "median_e2el_ms"),
        "e2el_p95": _pick(d, "p95_e2el_ms", "p99_e2el_ms"),
        "out_tps": _pick(d, "output_throughput"),
        "total_tps": _pick(d, "total_token_throughput"),
        "req_ps": _pick(d, "request_throughput"),
    }


def _stat(metrics, keys, field):
    """Pull one percentile out of a genai-perf metric block."""
    for k in keys:
        blk = metrics.get(k)
        if isinstance(blk, dict) and blk.get(field) is not None:
            return blk[field]
        if isinstance(blk, (int, float)) and field == "avg":
            return blk
    return NAN


def read_genai_perf(path):
    """genai-perf artifact dir (or its profile_export_genai_perf.json).

    Metrics are nested {avg,p50,p95,p99} blocks. Some releases report
    latencies in microseconds and some in milliseconds, so we normalise
    by the unit string when it is present and fall back to a magnitude
    check when it is not.
    """
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "**", "*genai_perf*.json"),
                                 recursive=True))
        if not cands:
            raise FileNotFoundError(f"no genai-perf json under {path}")
        fpath = cands[0]
    else:
        fpath = path
    d = json.load(open(fpath))

    metrics = d.get("metrics", d)
    ttft_keys = ["time_to_first_token"]
    itl_keys = ["inter_token_latency"]
    e2e_keys = ["request_latency"]
    thr_keys = ["output_token_throughput"]
    rps_keys = ["request_throughput"]

    scale = 1.0
    unit = ""
    blk = metrics.get("time_to_first_token")
    if isinstance(blk, dict):
        unit = str(blk.get("unit", "")).lower()
    if "us" in unit or "micro" in unit:
        scale = 1e-3
    elif "ns" in unit:
        scale = 1e-6
    elif not unit:
        probe = _stat(metrics, ttft_keys, "avg")
        if probe == probe and probe > 20000:      # implausible in ms
            scale = 1e-3

    # concurrency: from the run config if present, else from the dir name
    n = NAN
    cfg = d.get("input_config", d.get("config", {})) or {}
    for k in ("concurrency", "concurrency_range"):
        if isinstance(cfg, dict) and cfg.get(k) is not None:
            try:
                n = int(str(cfg[k]).split(",")[0])
            except ValueError:
                pass
    if n != n:
        m = re.search(r"_n(\d+)", path)
        if m:
            n = int(m.group(1))

    return {
        "n": n,
        "ttft_med": _stat(metrics, ttft_keys, "p50") * scale,
        "ttft_p95": _stat(metrics, ttft_keys, "p95") * scale,
        "tpot_med": _stat(metrics, itl_keys, "p50") * scale,
        "tpot_p95": _stat(metrics, itl_keys, "p95") * scale,
        "e2el_med": _stat(metrics, e2e_keys, "p50") * scale,
        "e2el_p95": _stat(metrics, e2e_keys, "p95") * scale,
        "out_tps": _stat(metrics, thr_keys, "avg"),
        "total_tps": NAN,                         # genai-perf has no input-token rate
        "req_ps": _stat(metrics, rps_keys, "avg"),
    }


READERS = {"vllm": read_vllm, "genai-perf": read_genai_perf}
FIELDS = ["n", "ttft_med", "ttft_p95", "tpot_med", "tpot_p95", "e2el_med",
          "e2el_p95", "out_tps", "total_tps", "req_ps"]


def load(pattern, source):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"nothing matched: {pattern}")
    rows = [READERS[source](p) for p in paths]
    rows.sort(key=lambda r: r["n"])
    return rows


def read_summary(path):
    """A <tag>_summary.csv written by this script. Missing columns become NaN."""
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            r = {}
            for k in FIELDS + ["cost_per_1m_inr", "l_over_m_inr"]:
                try:
                    r[k] = float(raw.get(k, "nan"))
                except ValueError:
                    r[k] = NAN
            rows.append(r)
    rows.sort(key=lambda r: r["n"])
    return rows


def add_cost(rows, rate):
    for r in rows:
        r["cost_per_1m_inr"] = (
            rate / (r["out_tps"] * 3600) * 1e6
            if rate and ok(r["out_tps"]) and r["out_tps"] > 0
            else NAN
        )
        r["l_over_m_inr"] = rate / r["n"] if rate and r["n"] else NAN
    return rows


def compare_series(entries):
    """--compare entries -> [(label, rows)]. Entries: path, glob, or label=path."""
    named, auto = [], []
    for e in entries:
        label, _, path = e.rpartition("=") if "=" in e else ("", "", e)
        paths = sorted(glob.glob(path)) if any(ch in path for ch in "*?[") else [path]
        if not paths:
            raise SystemExit(f"nothing matched: {path}")
        for p in paths:
            (named if label else auto).append((label, p))
    # default labels: CSV stem minus "_summary", minus the tokens every stem shares
    stems = [os.path.basename(p).removesuffix(".csv").removesuffix("_summary").split("_")
             for _, p in auto]
    if len(stems) > 1:
        while all(len(s) > 1 for s in stems) and len({s[-1] for s in stems}) == 1:
            stems = [s[:-1] for s in stems]
        while all(len(s) > 1 for s in stems) and len({s[0] for s in stems}) == 1:
            stems = [s[1:] for s in stems]
    auto = [("_".join(s), p) for s, (_, p) in zip(stems, auto)]
    out = [(label, read_summary(p)) for label, p in named + auto]
    labels = [l for l, _ in out]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate labels {labels}; name them with label=path")
    return out


# --------------------------------------------------------------- style
# Categorical slots in fixed order (validated light-surface palette). A series
# keeps its slot across every plot, so pp2 is the same color everywhere.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
    "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "lines.markersize": 6,
})


# --------------------------------------------------------------- main
ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pattern", nargs="?",
                help="glob of result files (vllm) or artifact dirs (genai-perf)")
ap.add_argument("--source", choices=list(READERS), default="vllm")
ap.add_argument("--label", help="legend name for the single sweep (default: --tag)")
ap.add_argument("--overlay", metavar="PATTERN",
                help="second sweep to draw on the same axes")
ap.add_argument("--overlay-source", choices=list(READERS), default="genai-perf")
ap.add_argument("--compare", nargs="+", metavar="CSV",
                help="summary CSVs of several configs (path, glob, or label=path)")
ap.add_argument("--baseline", nargs="+", metavar="LABEL",
                help="with --compare: plot speedup vs the best of these labels")
ap.add_argument("--tag", default="run")
ap.add_argument("--out-dir", default="plots", help="where PNGs and the CSV go")
ap.add_argument("--logy", action="store_true",
                help="log y axis on latency plots (TTFT spans orders of magnitude "
                     "once requests queue)")
ap.add_argument("--tpot-target", type=float, default=50.0,
                help="p95 TPOT target in ms; sets M and the line on the plot")
ap.add_argument("--gpu-rate-inr", type=float, default=0.0,
                help="GPU cost in rupees per hour")
args = ap.parse_args()

if bool(args.pattern) == bool(args.compare):
    ap.error("give either a result-file pattern or --compare, not both")
if args.baseline and not args.compare:
    ap.error("--baseline needs --compare")

if args.compare:
    series = compare_series(args.compare)
    if args.gpu_rate_inr:
        series = [(l, add_cost(rows, args.gpu_rate_inr)) for l, rows in series]
else:
    series = [(args.label or args.tag,
               add_cost(load(args.pattern, args.source), args.gpu_rate_inr))]
    if args.overlay:
        series[0] = (f"{series[0][0]} ({args.source})", series[0][1])
        series.append((f"{args.label or args.tag} ({args.overlay_source})",
                       add_cost(load(args.overlay, args.overlay_source), args.gpu_rate_inr)))
if len(series) > len(PALETTE):
    raise SystemExit(f"{len(series)} series but only {len(PALETTE)} distinguishable "
                     "colors; split the comparison into groups")
if args.baseline:
    missing = set(args.baseline) - {l for l, _ in series}
    if missing:
        raise SystemExit(f"--baseline labels not found: {sorted(missing)}; "
                         f"have {[l for l, _ in series]}")

os.makedirs(args.out_dir, exist_ok=True)
written = []
LEVELS = sorted({r["n"] for _, rows in series for r in rows if ok(r["n"])})
MULTI = len(series) > 1


def has(key):
    return any(ok(r.get(key)) for _, rows in series for r in rows)


def new_axes(title, ylabel, xlabel="concurrency (max in-flight requests)"):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_title(f"{title}  ·  {args.tag}", loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y")
    return fig, ax


def conc_axis(ax):
    ax.set_xscale("log", base=2)       # 1,2,4..128 evenly spaced
    ax.set_xticks(LEVELS, [f"{n:g}" for n in LEVELS])
    ax.minorticks_off()


def save(fig, ax, name, legend_loc="best"):
    if ax.get_legend_handles_labels()[0] and (MULTI or len(ax.lines) > 1):
        ax.legend(loc=legend_loc)
    fig.tight_layout()
    path = os.path.join(args.out_dir, f"{args.tag}_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)


def latency_plot(name, title, med, p95, scale, unit, target=None):
    """Median solid + p95 dashed, same color per config."""
    if not (has(med) or has(p95)):
        return
    fig, ax = new_axes(title, unit)
    for i, (label, rows) in enumerate(series):
        c = PALETTE[i]
        ns = [r["n"] for r in rows]
        ax.plot(ns, [r[med] * scale for r in rows], "o-", color=c, label=f"{label} median")
        ax.plot(ns, [r[p95] * scale for r in rows], "s--", color=c, alpha=0.6, lw=1.5,
                label=f"{label} p95")
    if target is not None:
        ax.axhline(target, color=INK2, lw=1, ls=":", label=f"target {target:g} {unit}")
    conc_axis(ax)
    if args.logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    save(fig, ax, name)


def value_plot(name, title, key, unit):
    if not has(key):
        return
    fig, ax = new_axes(title, unit)
    for i, (label, rows) in enumerate(series):
        ax.plot([r["n"] for r in rows], [r[key] for r in rows], "o-",
                color=PALETTE[i], label=label)
    conc_axis(ax)
    ax.set_ylim(bottom=0)
    save(fig, ax, name)


THR = "total_tps" if has("total_tps") else "out_tps"
THR_NAME = "total (input + output) tok/s" if THR == "total_tps" else "output tok/s"

latency_plot("ttft", "Time to first token", "ttft_med", "ttft_p95", 1e-3, "s")
latency_plot("tpot", "Time per output token", "tpot_med", "tpot_p95", 1.0, "ms",
             target=args.tpot_target)
latency_plot("e2el", "End-to-end request latency", "e2el_med", "e2el_p95", 1e-3, "s")
value_plot("throughput_total", "Total token throughput (input + output)", "total_tps", "tok/s")
value_plot("throughput_output", "Output token throughput", "out_tps", "tok/s")
value_plot("throughput_requests", "Request throughput", "req_ps", "requests/s")
if args.gpu_rate_inr:
    value_plot("cost", f"Cost per 1M output tokens (Rs {args.gpu_rate_inr:g}/hr)",
               "cost_per_1m_inr", "rupees")

# Throughput vs latency: each point is one concurrency level. Up-and-left is
# better, so a config whose curve sits above another's wins at equal latency.
if has(THR) and has("e2el_med"):
    fig, ax = new_axes("Throughput vs latency", THR_NAME,
                       xlabel="median end-to-end latency (s)")
    for i, (label, rows) in enumerate(series):
        pts = [(r["e2el_med"] * 1e-3, r[THR], r["n"]) for r in rows
               if ok(r["e2el_med"]) and ok(r[THR])]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=PALETTE[i], label=label)
        for x, y, n in (pts[0], pts[-1]):          # label the ends, not every point
            ax.annotate(f"c={n:g}", (x, y), textcoords="offset points", xytext=(6, -10),
                        fontsize=8, color=INK2)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    save(fig, ax, "throughput_vs_latency")

# Speedup over the best baseline at each concurrency (paper Fig. 10 style).
speedup = {}
if args.baseline:
    by_label = {l: {r["n"]: r for r in rows} for l, rows in series}

    def best_base(n):
        vals = [by_label[b][n][THR] for b in args.baseline
                if n in by_label[b] and ok(by_label[b][n][THR])]
        return max(vals) if vals else NAN

    others = [(i, l) for i, (l, _) in enumerate(series) if l not in args.baseline]
    for _, l in others:
        speedup[l] = {n: by_label[l][n][THR] / best_base(n)
                      for n in LEVELS if n in by_label[l] and ok(best_base(n))}
    if others:
        base_name = " / ".join(args.baseline)
        fig, ax = new_axes(f"Speedup over best of {base_name}",
                           f"{THR_NAME} ÷ best baseline")
        width = 0.8 / len(others)
        for k, (i, l) in enumerate(others):
            xs = [LEVELS.index(n) + (k - (len(others) - 1) / 2) * width for n in speedup[l]]
            ax.bar(xs, list(speedup[l].values()), width=width - 0.04, color=PALETTE[i],
                   label=l, zorder=2)
        ax.axhline(1.0, color=INK2, lw=1, ls=":", zorder=3, label="= best baseline")
        ax.set_xticks(range(len(LEVELS)), [f"{n:g}" for n in LEVELS])
        ax.set_ylim(bottom=0)
        save(fig, ax, "speedup", legend_loc="lower right")

# --------------------------------------------------------------- tables
print()
if args.compare:
    labels = [l for l, _ in series]
    w = max(9, *(len(l) for l in labels))
    hdr = f"{'N':>4}  " + "  ".join(f"{l:>{w}}" for l in labels)
    hdr += "".join(f"  {('x ' + l):>{w}}" for l in speedup)
    print(f"{THR_NAME} per config" + (f"; 'x' = speedup vs best of {'/'.join(args.baseline)}"
                                      if speedup else ""))
    print(hdr)
    print("-" * len(hdr))
    by_label = {l: {r["n"]: r for r in rows} for l, rows in series}
    for n in LEVELS:
        line = f"{n:>4g}  " + "  ".join(
            f"{by_label[l][n][THR]:>{w}.1f}" if n in by_label[l] else f"{'-':>{w}}"
            for l in labels)
        line += "".join(f"  {speedup[l][n]:>{w}.2f}" if n in speedup[l] else f"  {'-':>{w}}"
                        for l in speedup)
        print(line)
    for l, s in speedup.items():
        if s:
            geo = math.exp(sum(math.log(v) for v in s.values()) / len(s))
            print(f"\n{l}: geomean speedup {geo:.2f}x, best {max(s.values()):.2f}x "
                  f"at N = {max(s, key=s.get):g}")
else:
    # summary csv + M, from the primary series
    rows = series[0][1]
    csv_path = os.path.join(args.out_dir, f"{args.tag}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    written.append(csv_path)

    hdr = (f"{'N':>4}  {'p95 TTFT':>9}  {'p95 TPOT':>9}  {'p95 E2E':>9}  "
           f"{'out tok/s':>9}  {'tot tok/s':>9}  {'req/s':>7}")
    if args.gpu_rate_inr:
        hdr += f"  {'L/M (Rs)':>9}  {'Rs/1M out':>10}"
    hdr += "  meets"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        met = "yes" if (ok(r["tpot_p95"]) and r["tpot_p95"] <= args.tpot_target) else "NO"
        line = (f"{r['n']:>4}  {r['ttft_p95']:>9.0f}  {r['tpot_p95']:>9.1f}  {r['e2el_p95']:>9.0f}  "
                f"{r['out_tps']:>9.1f}  {r['total_tps']:>9.1f}  {r['req_ps']:>7.3f}")
        if args.gpu_rate_inr:
            line += f"  {r['l_over_m_inr']:>9.2f}  {r['cost_per_1m_inr']:>10.2f}"
        line += f"  {met:>5}"
        print(line)
    print(f"\n(latencies in ms; 'meets' = p95 TPOT <= {args.tpot_target:g} ms)")

    good = [r for r in rows if ok(r["tpot_p95"]) and r["tpot_p95"] <= args.tpot_target]
    print()
    if good:
        b = max(good, key=lambda r: r["n"])
        print(f"M = {b['n']} concurrent users at p95 TPOT <= {args.tpot_target:g} ms")
        print(f"  output throughput : {b['out_tps']:.1f} tok/s")
        print(f"  p95 TTFT          : {b['ttft_p95']:.0f} ms")
        if args.gpu_rate_inr:
            print(f"  L/M               : Rs {b['l_over_m_inr']:.3f} per user-hour")
            print(f"  cost / 1M out tok : Rs {b['cost_per_1m_inr']:.2f}")
    else:
        print(f"no concurrency level met p95 TPOT <= {args.tpot_target:g} ms")

    # cheapest point regardless of the target, for the cost/latency trade-off
    if args.gpu_rate_inr:
        valid = [r for r in rows if ok(r["cost_per_1m_inr"])]
        if valid:
            c = min(valid, key=lambda r: r["cost_per_1m_inr"])
            print(f"\ncheapest point ignoring the target: N = {c['n']}, "
                  f"Rs {c['cost_per_1m_inr']:.2f} / 1M out tok, p95 TPOT {c['tpot_p95']:.1f} ms")

print("\nwrote:")
for p in written:
    print(f"  {p}")
