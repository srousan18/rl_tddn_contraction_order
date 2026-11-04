"""Compare RL and baseline results.

Sub-commands:

``table``        per-method summary over the instances each method solved
                 (``status == ok`` and ``identity_status == identity``):
                 solve count, arithmetic and geometric mean of peak and time,
                 a matched-set line (instances every method solved), and the
                 geometric mean per family. ``--heldout`` keeps only instances
                 that are not in the given training list (BV is never trained on).

``fingerprint``  per-instance peak and status equality of two RL results
                 directories (for example a re-evaluation against
                 ``reference_results``); exit status 1 on any difference.

Inputs are results directories (``results.csv`` inside) written by
``tddn.eval.runner`` (RL) and ``tddn.baselines.runner`` (baselines).
Baseline repeats are reduced per instance by the median. Time: RL
``episode_s``; baselines ``search_s + cont_s``.

CLI:
    python -m tddn.eval.compare table --rl two_stage=results/two_stage/benchmarks \\
        --baselines reference_results/baselines/benchmarks [--heldout lists/benchmark_train.txt]
    python -m tddn.eval.compare fingerprint results/two_stage/benchmarks \\
        reference_results/two_stage/benchmarks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

from tddn.baselines.runner import METHOD_NAMES

FAMILY_RE = re.compile(r"combined_(.+?)_(\d+)q\.qasm$")


def family_of(circuit: str) -> str:
    m = FAMILY_RE.search(circuit)
    return m.group(1) if m else "?"


def _ok(r: dict) -> bool:
    return r["status"] == "ok" and r["identity_status"] == "identity"


def load_rl(results_dir: str | Path) -> dict:
    """circuit -> {qubits, peak, time} for solved rows; plus all circuit names."""
    rows = list(csv.DictReader(open(Path(results_dir) / "results.csv")))
    solved = {r["circuit"]: {"qubits": int(r["qubits"]),
                             "peak": float(r["peak_tdd_nodes"]),
                             "time": float(r["episode_s"])}
              for r in rows if _ok(r)}
    return {"solved": solved, "all": {r["circuit"] for r in rows}}


def load_baselines(results_dir: str | Path) -> dict:
    """method label -> same structure as ``load_rl`` (median over repeats)."""
    rows = list(csv.DictReader(open(Path(results_dir) / "results.csv")))
    per = defaultdict(lambda: defaultdict(list))
    seen = defaultdict(set)
    for r in rows:
        label = METHOD_NAMES.get(r["method"], r["method"])
        seen[label].add(r["circuit"])
        if _ok(r):
            per[label][r["circuit"]].append(r)
    out = {}
    for label in seen:
        solved = {}
        for c, rr in per[label].items():
            solved[c] = {"qubits": int(rr[0]["qubits"]),
                         "peak": median(float(x["peak_tdd_nodes"]) for x in rr),
                         "time": median(float(x["search_s"] or 0) + float(x["cont_s"] or 0)
                                        for x in rr)}
        out[label] = {"solved": solved, "all": seen[label]}
    return out


def heldout_filter(train_list: str | Path):
    """Predicate: the instance is not in the training list (a list file or a run manifest)."""
    if str(train_list).endswith(".json"):
        trained = {Path(p).name for p in json.load(open(train_list))["train_files"]}
    else:
        trained = {Path(line.strip()).name for line in open(train_list) if line.strip()}
    return lambda c: c not in trained or family_of(c) == "bv"


def gmean(xs):
    return math.exp(sum(math.log(max(x, 1e-12)) for x in xs) / len(xs)) if xs else float("nan")


def amean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summary_table(methods: dict, keep=lambda c: True) -> str:
    """methods: label -> load_* structure. Returns a printable report."""
    lines = []
    universe = sorted(set().union(*(m["all"] for m in methods.values())))
    universe = [c for c in universe if keep(c)]
    fams = sorted({family_of(c) for c in universe})
    hdr = (f"{'method':34s} {'solved':>8s} {'peak amean':>11s} {'peak gmean':>11s} "
           f"{'time amean':>11s} {'time gmean':>11s}")
    lines += [f"instances: {len(universe)}", hdr, "-" * len(hdr)]
    for label, m in methods.items():
        s = {c: v for c, v in m["solved"].items() if keep(c)}
        n_all = len([c for c in m["all"] if keep(c)])
        pk = [v["peak"] for v in s.values()]
        tt = [v["time"] for v in s.values()]
        lines.append(f"{label:34s} {len(s):>4d}/{n_all:<3d} {amean(pk):11.1f} "
                     f"{gmean(pk):11.1f} {amean(tt):11.4f} {gmean(tt):11.4f}")

    common = set(universe)
    for m in methods.values():
        common &= set(m["solved"])
    lines += ["", f"matched set (solved by every method): {len(common)} instances"]
    for label, m in methods.items():
        pk = [m["solved"][c]["peak"] for c in common]
        lines.append(f"  {label:32s} peak gmean {gmean(pk):9.1f}   amean {amean(pk):9.1f}")

    lines += ["", "peak gmean per family (own solved set; solved count in brackets)"]
    lines.append(f"{'method':34s} " + " ".join(f"{f:>14s}" for f in fams))
    for label, m in methods.items():
        cells = []
        for f in fams:
            pk = [v["peak"] for c, v in m["solved"].items() if keep(c) and family_of(c) == f]
            cells.append(f"{gmean(pk):9.1f} [{len(pk):2d}]")
        lines.append(f"{label:34s} " + " ".join(f"{x:>14s}" for x in cells))
    return "\n".join(lines)


def fingerprint(new_dir, ref_dir) -> int:
    new = {r["circuit"]: r for r in csv.DictReader(open(Path(new_dir) / "results.csv"))}
    ref = {r["circuit"]: r for r in csv.DictReader(open(Path(ref_dir) / "results.csv"))}
    common = sorted(set(new) & set(ref))
    bad = [(c, ref[c]["peak_tdd_nodes"], new[c]["peak_tdd_nodes"],
            ref[c]["status"], new[c]["status"]) for c in common
           if (ref[c]["peak_tdd_nodes"], ref[c]["status"])
           != (new[c]["peak_tdd_nodes"], new[c]["status"])]
    print(f"fingerprint: {len(common)} common instances, {len(bad)} mismatches "
          f"(new-only {len(set(new) - set(ref))}, reference-only {len(set(ref) - set(new))})")
    for c, rp, np_, rs, ns in bad[:20]:
        print(f"  {c}: reference {rp} ({rs})  new {np_} ({ns})")
    return 1 if bad or not common else 0


def _parse_rl(specs):
    out = {}
    for spec in specs or []:
        label, _, path = spec.partition("=")
        out[label] = load_rl(path)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--rl", action="append", help="label=results_dir (repeatable)")
    t.add_argument("--baselines", default=None, help="baseline results dir")
    t.add_argument("--heldout", default=None,
                   help="training list: keep only instances not in it")
    f = sub.add_parser("fingerprint")
    f.add_argument("new_dir")
    f.add_argument("ref_dir")
    args = ap.parse_args(argv)

    if args.cmd == "fingerprint":
        return fingerprint(args.new_dir, args.ref_dir)
    methods = _parse_rl(args.rl)
    if args.baselines:
        methods.update(load_baselines(args.baselines))
    keep = heldout_filter(args.heldout) if args.heldout else (lambda c: True)
    print(summary_table(methods, keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
