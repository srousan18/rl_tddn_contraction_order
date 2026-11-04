"""Figures for the 24-50 qubit scaling results (``tddn.eval.scaling``).

One figure per family and metric (peak TDD nodes, total time):

  - point     median over successful repeats
  - whiskers  min-max over successful repeats (cotengra repeats)
  - gap       a size where every repeat failed, followed later by a success
  - X marker  first size after which the method never succeeds again
  - hollow    RL result under the extended time limit (``results_ext.csv``)

Only ``ok`` rows whose final TDD is ``identity`` count as successes;
``incomplete`` rows are failures. A ``not_identity`` row on these equivalent
circuits is reported and makes the command exit with status 1.

CLI:
    python -m tddn.eval.scaling_plots --results reference_results/baselines/scaling \\
        --results results/two_stage/scaling --out results/plots/scaling
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TIMEOUT_S = 120.0

BASELINES = [
    ("counting", "Counting", "#1f77b4"),
    ("cotengra_kahypar", "KaHyPar", "#ff7f0e"),
    ("cotengra_greedy", "Greedy", "#2ca02c"),
    ("cotengra_rgreedy", "Random Greedy", "#8c564b"),
    ("cotengra_between", "Betweenness", "#9467bd"),
]
RL_COLORS = ["#d62728", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22", "#000000"]
FAMILY_NAMES = {"bv": "BV", "dj": "DJ", "ghz": "GHZ", "qnn": "QNN", "wstate": "W-State"}


def read_rows(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def method_list(rows, family):
    """(key, label, color): one RL line per model, then the baselines."""

    models = list(dict.fromkeys(r["model"] for r in rows if r["method"] == "rl"))
    hybrid = " (Hybrid)" if family == "wstate" else ""
    out = [(("rl", m), f"RL {m}{hybrid}", RL_COLORS[i % len(RL_COLORS)])
           for i, m in enumerate(models)]
    present = {r["variant"] for r in rows if r["method"] != "rl"}
    return out + [((v, ""), lbl, c) for v, lbl, c in BASELINES if v in present]


def _key(r):
    return ("rl", r["model"]) if r["method"] == "rl" else (r["variant"], "")


def _ok(r):
    return r["status"] == "ok" and r["identity_status"] == "identity"


def load_series(rows, ext_rows, family):
    rows = [r for r in rows if r["family"] == family]
    # A wrong answer is not_identity; "incomplete" (e.g. the env's 100k-node
    # guard banned the last edge) is a failed run and is reported, not drawn.
    contradictions = [r for r in rows if r["status"] == "ok"
                      and r["identity_status"] == "not_identity"]
    for r in rows:
        if r["status"] == "ok" and r["identity_status"] not in ("identity", "not_identity"):
            print(f"failed ({r['identity_status']}): {r['file']} {r['variant']} "
                  f"{r['model']} r{r['repeat']}")
    qubits = sorted({int(r["qubits"]) for r in rows})
    methods = method_list(rows, family)
    series = {}
    for key, label, color in methods:
        pts = {}          # q -> dict(peaks=[], times=[]) over successful repeats
        fails = []        # sizes where every measured repeat failed
        for q in qubits:
            reps = [r for r in rows if _key(r) == key and int(r["qubits"]) == q]
            ok = [r for r in reps if _ok(r)]
            if ok:
                pts[q] = {
                    "peaks": [int(r["peak_tdd_nodes"]) for r in ok],
                    "times": [float(r["total_time_s"]) for r in ok],
                }
            elif reps:
                fails.append(q)
        # X marker: first all-fail size never followed by a success
        last_ok = max(pts) if pts else None
        fail_at = None
        for q in fails:
            if last_ok is None or q > last_ok:
                fail_at = q
                break
        ext = {int(r["qubits"]): {"peaks": [int(r["peak_tdd_nodes"])],
                                  "times": [float(r["total_time_s"])]}
               for r in ext_rows
               if r["family"] == family and _key(r) == key and _ok(r)}
        series[label] = {"key": key, "color": color, "pts": pts, "fails": fails,
                         "fail_at": fail_at, "ext": ext}
    return series, qubits, contradictions


def style_axes(ax, ylabel, qubits, family):
    ax.set_yscale("log")
    ax.set_xlabel(f"Qubits ({FAMILY_NAMES.get(family, family)})", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if len(qubits) <= 8:
        ax.set_xticks(qubits)
    else:
        ax.set_xticks([q for q in qubits if q % 2 == 0])
    ax.tick_params(labelsize=12)
    ax.grid(True, which="major", linestyle="--", alpha=0.6)


def draw(series, qubits, family, value_key, ylabel, out, marker, x_y_value,
         timeout_line=False):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    handles = {}
    # dodge X markers that land on the same qubit count so none is hidden
    fail_counts = {}
    for s in series.values():
        q = s["fail_at"]
        if q is not None:
            fail_counts[q] = fail_counts.get(q, 0) + 1
    fail_seen = {}
    for label, s in series.items():
        # NaN at all-fail sizes so the line breaks instead of bridging the gap
        xs = sorted(set(list(s["pts"]) + s["fails"]))
        med = np.array([statistics.median(s["pts"][q][value_key])
                        if q in s["pts"] else np.nan for q in xs])
        lo = np.array([med[i] - min(s["pts"][q][value_key])
                       if q in s["pts"] else np.nan for i, q in enumerate(xs)])
        hi = np.array([max(s["pts"][q][value_key]) - med[i]
                       if q in s["pts"] else np.nan for i, q in enumerate(xs)])
        if s["pts"]:
            # RL lines on top: several models often tie the baselines exactly.
            h = ax.errorbar(xs, med, yerr=[lo, hi], color=s["color"], marker=marker,
                            markersize=7, linewidth=2.0, capsize=3,
                            zorder=3.5 if s["key"][0] == "rl" else 3)
        else:
            h, = ax.plot([], [], color=s["color"], marker=marker, markersize=7,
                         linewidth=2.0)
        handles[label] = h
        for q, d in sorted(s["ext"].items()):
            v = statistics.median(d[value_key])
            ax.plot([q], [v], marker=marker, markersize=9, linestyle="none",
                    markerfacecolor="white", markeredgecolor=s["color"],
                    markeredgewidth=2.0, zorder=4)
            # dotted connectors to the neighbouring protocol points
            for nb in (q - 1, q + 1):
                if nb in s["pts"]:
                    nv = statistics.median(s["pts"][nb][value_key])
                    ax.plot([nb, q], [nv, v], color=s["color"],
                            linestyle=":", linewidth=1.6, zorder=2)
        if s["fail_at"] is not None:
            q = s["fail_at"]
            i = fail_seen.get(q, 0)
            fail_seen[q] = i + 1
            offset = (i - (fail_counts[q] - 1) / 2) * 0.55
            ax.plot([q + offset], [x_y_value], marker="X", markersize=16,
                    color=s["color"], markeredgecolor="#333333",
                    markeredgewidth=1.0, linestyle="none", zorder=4)
    if timeout_line:
        ax.axhline(TIMEOUT_S, color="gray", linestyle=":", linewidth=1.8)
        ax.text(qubits[0] - 0.3, TIMEOUT_S * 1.12, f"Timeout ({int(TIMEOUT_S)}s)",
                color="gray", fontsize=12)
    style_axes(ax, ylabel, qubits, family)
    labels = list(series)
    ax.legend([handles[l] for l in labels], labels, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), fontsize=12, frameon=True)
    fig.tight_layout()
    for ext_ in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext_}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.with_suffix(".png"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, action="append",
                    help="tddn.eval.scaling output directory (repeatable; rows are pooled)")
    ap.add_argument("--models", default=None,
                    help="comma list of RL model labels to draw (default: all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = [r for d in args.results for r in read_rows(Path(d) / "results.csv")]
    ext_rows = [r for d in args.results for r in read_rows(Path(d) / "results_ext.csv")]
    if args.models:
        keep = set(args.models.split(","))
        rows = [r for r in rows if r["method"] != "rl" or r["model"] in keep]
        ext_rows = [r for r in ext_rows if r["model"] in keep]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for family in [f for f in FAMILY_NAMES if any(r["family"] == f for r in rows)]:
        series, qubits, contradictions = load_series(rows, ext_rows, family)
        bad += len(contradictions)
        for r in contradictions:
            print(f"NOT IDENTITY {r['file']} {r['variant']} {r['model']} "
                  f"r{r['repeat']}: {r['identity_status']}")
        all_peaks = [p for s in series.values() for d in s["pts"].values()
                     for p in d["peaks"]] or [1]
        draw(series, qubits, family, "peaks", "Peak TDD Nodes (Log Scale)",
             out / f"peak_{family}", marker="o",
             x_y_value=max(all_peaks) * 1.6)
        draw(series, qubits, family, "times", "Total Time [s] (Log Scale)",
             out / f"time_{family}", marker="s",
             x_y_value=TIMEOUT_S, timeout_line=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
