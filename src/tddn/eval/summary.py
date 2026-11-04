"""Summaries of scaling and miter-list results.

Sub-commands:

``scaling``  per model / method and set: instances solved (``ok`` with an
             ``identity`` final TDD) and the geometric-mean peak, overall and
             per family, from ``tddn.eval.scaling`` results.

``fileset``  per model, from ``tddn.eval.fileset`` results: on equivalent
             miters (modes ``combined`` and ``cross_instance_control``) the
             number completed and verified as identity; on non-equivalent
             miters the number completed, correctly rejected (``not_identity``)
             and wrongly accepted (``identity``); geometric-mean peaks over the
             completed rows of each group. Unstructured random miters (C . C^-1)
             are equivalent.

``runs``     robustness across training runs: the arithmetic mean peak of
             each model on the equivalent miters of a non-equivalence result, the
             mean and sample standard deviation across models, and, with
             ``--baselines``, the ratios to the Greedy and Counting mean peaks
             and to the KaHyPar mean time.

CLI:
    python -m tddn.eval.summary scaling results/two_stage/scaling reference_results/baselines/scaling
    python -m tddn.eval.summary fileset results/two_stage/nonequivalence
    python -m tddn.eval.summary runs results/s1000/nonequivalence results/s1001/nonequivalence \\
        --baselines results/baselines/nonequivalence_equivalent
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

EQUIVALENT_MODES = {"combined", "cross_instance_control", "random_unstructured"}


def _rows(dirs):
    out = []
    for d in dirs:
        path = Path(d) / "results.csv"
        if not path.exists():
            raise SystemExit(f"no results.csv in {d}")
        out += list(csv.DictReader(open(path)))
    return out


def _ok(r) -> bool:
    return r["status"] == "ok" and r["identity_status"] == "identity"


def gmean(xs) -> float:
    xs = [max(float(x), 1e-12) for x in xs]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def scaling(dirs) -> None:
    rows = _rows(dirs)
    groups = defaultdict(list)
    for r in rows:
        label = f"RL {r['model']}" if r["method"] == "rl" else r["variant"]
        groups[label].append(r)
    families = sorted({r["family"] for r in rows})
    print(f"{'method':32s} {'set':9s} {'solved':>9s} {'gm peak':>9s}   "
          + " ".join(f"{f:>13s}" for f in families))
    for label, rs in groups.items():
        for setname in sorted({r["set"] for r in rs}):
            per_inst = defaultdict(list)
            for r in rs:
                if r["set"] == setname:
                    per_inst[r["file"]].append(r)
            # An instance counts as solved if any repeat solved it; peak = median of solved repeats.
            solved = {f: st.median(int(x["peak_tdd_nodes"]) for x in v if _ok(x))
                      for f, v in per_inst.items() if any(_ok(x) for x in v)}
            fam_cells = []
            for fam in families:
                fam_files = [f for f, v in per_inst.items() if v[0]["family"] == fam]
                if not fam_files:
                    fam_cells.append(f"{'':>13s}")
                    continue
                peaks = [solved[f] for f in fam_files if f in solved]
                fam_cells.append(f"{gmean(peaks):8.0f} {len(peaks):2d}/{len(fam_files):<2d}")
            print(f"{label:32s} {setname:9s} {len(solved):>4d}/{len(per_inst):<4d} "
                  f"{gmean(solved.values()):9.1f}   " + " ".join(fam_cells))


def fileset(dirs) -> None:
    rows = _rows(dirs)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    print(f"{'model':24s} {'equivalent: identity/total':>27s} {'gm':>7s}   "
          f"{'non-equivalent: rejected / wrongly accepted / completed / total':>63s} {'gm':>7s}")
    for model, rs in by_model.items():
        eq = [r for r in rs if r["mode"] in EQUIVALENT_MODES]
        ne = [r for r in rs if r["mode"] not in EQUIVALENT_MODES]
        eq_ok = [r for r in eq if _ok(r)]
        ne_done = [r for r in ne if r["status"] == "ok"]
        rejected = sum(r["identity_status"] == "not_identity" for r in ne_done)
        accepted = sum(r["identity_status"] == "identity" for r in ne_done)
        print(f"{model:24s} {len(eq_ok):>20d}/{len(eq):<6d} "
              f"{gmean(r['peak_tdd_nodes'] for r in eq_ok):7.1f}   "
              f"{rejected:>40d} / {accepted:>3d} / {len(ne_done):>3d} / {len(ne):<5d} "
              f"{gmean(r['peak_tdd_nodes'] for r in ne_done):7.1f}")


def runs(dirs, baselines=None) -> None:
    rows = _rows(dirs)
    by_model = defaultdict(list)
    for r in rows:
        if r["mode"] in EQUIVALENT_MODES:
            by_model[r["model"]].append(r)
    peaks, times = [], []
    for model, rs in by_model.items():
        bad = sum(not _ok(r) for r in rs)
        p = st.mean(int(r["peak_tdd_nodes"]) for r in rs if _ok(r))
        t = st.mean(float(r["total_time_s"]) for r in rs if _ok(r))
        peaks.append(p)
        times.append(t)
        print(f"  {model:24s} n={len(rs)} not identity={bad}  mean peak {p:7.2f}  "
              f"mean time {t:.4f}s")
    if len(peaks) < 2:
        raise SystemExit("runs needs results of at least two models")
    print(f"\npeak across {len(peaks)} runs: {st.mean(peaks):.2f} +/- {st.stdev(peaks):.2f} "
          f"({100 * st.stdev(peaks) / st.mean(peaks):.1f}% relative)")
    if baselines:
        from tddn.eval.compare import load_baselines
        base = load_baselines(baselines)
        for name, metric, values in (("Greedy", "peak", peaks), ("Counting", "peak", peaks),
                                     ("KaHyPar", "time", times)):
            ref = st.mean(v[metric] for v in base[name]["solved"].values())
            ratios = [ref / v for v in values]
            what = "peak reduction" if metric == "peak" else "speed-up"
            print(f"{what} vs {name:8s}: {st.mean(ratios):.2f}x +/- {st.stdev(ratios):.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scaling", "fileset", "runs"):
        p = sub.add_parser(name)
        p.add_argument("dirs", nargs="+", help="results directories")
        if name == "runs":
            p.add_argument("--baselines", default=None,
                           help="tddn.baselines.runner results on the same equivalent miters")
    args = ap.parse_args(argv)
    if args.cmd == "scaling":
        scaling(args.dirs)
    elif args.cmd == "fileset":
        fileset(args.dirs)
    else:
        runs(args.dirs, args.baselines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
