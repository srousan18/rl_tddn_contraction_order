"""Figures for the 5-23 qubit benchmarks: per-size lines and per-instance panels.

Produces, for any set of RL results dirs and one baseline results dir:

    peak_mean.{png,pdf}   peak_geomean.{png,pdf}    max TDD nodes vs qubits
    time_mean.{png,pdf}   time_geomean.{png,pdf}    total time vs qubits
    per_instance_peak.png per_instance_time.png     one panel per family

Aggregate lines: per qubit size, the mean over that size's instances (one per
family) with a 95% bootstrap CI band (1000 resamples, seed 0; geometric
variant bootstrapped in log space). Arithmetic means are dominated by
W-State; the geometric variant shows the other families too. Each method is
drawn over the instances IT solved — a timed-out instance drops out of that
method's point (see the solve counts from ``tddn.eval.compare``).

CLI:
    python -m tddn.eval.plots --rl "two-stage=results/two_stage/benchmarks" \\
        --baselines reference_results/baselines/benchmarks --out results/plots/benchmarks \\
        [--heldout lists/benchmark_train.txt] [--note TEXT]
"""

from __future__ import annotations

import argparse
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullLocator,
)

from tddn.eval.compare import family_of, heldout_filter, load_baselines, load_rl

TEXT, MUTED, GRID = "#3d3d3a", "#73726c", "#e8e6dd"
RL_COLORS = ["#D62728", "#17BECF", "#E377C2", "#BCBD22", "#7F7F7F"]
RL_MARKERS = ["o", "P", "*", "h", "p"]
BASELINE_STYLE = {"Greedy": ("#2CA02C", "X"), "Random Greedy": ("#8C564B", "s"),
                  "Betweenness": ("#9467BD", "D"), "KaHyPar": ("#FF7F0E", "^"),
                  "Counting": ("#1F77B4", "v")}
FAMILIES = ["bv", "dj", "ghz", "qnn", "wstate"]

plt.rcParams.update({
    "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.facecolor": "white", "svg.fonttype": "none",
})


def _plain_log_axis(ax):
    """Log y axis with plain tick labels (20, 50, 1k) instead of 2x10^1."""
    def fmt(v, _):
        return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_minor_locator(NullLocator())


def _styles(methods):
    out, i = {}, 0
    for label in methods:
        if label in BASELINE_STYLE:
            out[label] = BASELINE_STYLE[label]
        else:
            out[label] = (RL_COLORS[i % len(RL_COLORS)], RL_MARKERS[i % len(RL_MARKERS)])
            i += 1
    return out


def size_lines(methods: dict, metric: str, geo: bool, out: Path, note: str,
               keep=lambda c: True):
    """One line per method: mean (or geomean) of ``metric`` per qubit size."""
    agg = ((lambda a, axis=None: np.exp(np.log(a).mean(axis=axis))) if geo
           else (lambda a, axis=None: a.mean(axis=axis)))
    rng = np.random.default_rng(0)
    styles = _styles(methods)
    fig, ax = plt.subplots(figsize=(8, 6.4))
    qmin, qmax = 99, 0
    for label, m in methods.items():
        byq = defaultdict(list)
        for c, v in m["solved"].items():
            if keep(c):
                byq[v["qubits"]].append(max(v[metric], 1e-6))
        qs = sorted(byq)
        if not qs:
            continue
        qmin, qmax = min(qmin, qs[0]), max(qmax, qs[-1])
        mean = [float(agg(np.array(byq[q]))) for q in qs]
        lo, hi = [], []
        for q in qs:
            v = np.array(byq[q])
            boots = agg(v[rng.integers(0, len(v), size=(1000, len(v)))], axis=1)
            lo.append(float(np.percentile(boots, 2.5)))
            hi.append(float(np.percentile(boots, 97.5)))
        color, marker = styles[label]
        ax.fill_between(qs, lo, hi, color=color, alpha=0.15, lw=0)
        # Greedy often equals Random Greedy: dashed and on top so it stays visible.
        dashed = label == "Greedy"
        ax.plot(qs, mean, color=color, lw=3, marker=marker, ms=9, label=label,
                ls="--" if dashed else "-", zorder=4 if dashed else 3)
    ax.set_yscale("log")
    what = "Max TDD Nodes" if metric == "peak" else "Total Time (s)"
    ax.set_ylabel(f"{what} ({'geo. mean' if geo else 'mean'})", fontsize=14)
    ax.set_xlabel("Qubit Count", fontsize=14)
    if qmax >= qmin:
        ax.set_xticks(range(qmin, qmax + 1, 2 if qmax - qmin > 10 else 1))
    ax.tick_params(labelsize=12)
    ax.grid(True, which="major", color="#dddddd")
    ax.legend(title="Method", loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncols=2, fontsize=11, title_fontsize=12, frameon=True)
    lines = textwrap.wrap(note + ("Line = geometric mean per size, " if geo
                                  else "Line = mean per size, ")
                          + "band = 95% bootstrap CI.", 115)
    fig.text(0.01, 0.005, "\n".join(lines), fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02 * len(lines), 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print(f"wrote {out}.png/.pdf")


def per_instance(methods: dict, metric: str, out: Path, keep=lambda c: True):
    """One panel per family: every instance's value per method, vs qubits."""
    styles = _styles(methods)
    fams = [f for f in FAMILIES if any(family_of(c) == f and keep(c)
                                       for m in methods.values() for c in m["all"])]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 4.4),
                             squeeze=False)
    for ax, f in zip(axes[0], fams):
        for label, m in methods.items():
            pts = sorted((v["qubits"], max(v[metric], 1e-6)) for c, v in m["solved"].items()
                         if keep(c) and family_of(c) == f)
            if pts:
                color, marker = styles[label]
                ax.plot(*zip(*pts), color=color, marker=marker, ms=5, lw=1.2,
                        alpha=0.9, label=label, ls="--" if label == "Greedy" else "-")
        ax.set_yscale("log")
        _plain_log_axis(ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f, fontsize=12)
        ax.set_xlabel("qubits")
    axes[0][0].set_ylabel("max TDD nodes" if metric == "peak" else "total time (s)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=min(len(labels), 7),
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl", action="append", default=[], help="label=results_dir")
    ap.add_argument("--baselines", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default=None,
                    help="training list: plot only instances not in it")
    ap.add_argument("--note", default="", help="footnote prefix, e.g. the instance set")
    args = ap.parse_args(argv)

    methods = {}
    for spec in args.rl:
        label, _, path = spec.partition("=")
        methods[label] = load_rl(path)
    if args.baselines:
        methods.update(load_baselines(args.baselines))
    keep = heldout_filter(args.heldout) if args.heldout else (lambda c: True)
    note = args.note + (" " if args.note else "")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for metric in ("peak", "time"):
        size_lines(methods, metric, False, out / f"{metric}_mean", note, keep)
        size_lines(methods, metric, True, out / f"{metric}_geomean", note, keep)
        per_instance(methods, metric, out / f"per_instance_{metric}", keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
