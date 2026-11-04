"""Baseline sweep: cotengra planners and Counting, executed with the TDD backend.

Each row builds the network, searches a path (``search_s``), contracts it in
a fresh TDD session (``cont_s``) and checks the final TDD against the
identity. Comparison time is ``search_s + cont_s``; ``total_s`` also covers
network construction and the identity check.

Every row has a hard time limit and a node cap checked between contractions.
Rows are appended to ``results.csv``; re-running the same command resumes.
cotengra methods are repeated (``--repeats``); Counting is deterministic and
runs once.

CLI:
    python -m tddn.baselines.runner --out results/baselines/benchmarks \\
        --list lists/benchmark_5to23q.txt [--methods kahypar,greedy,betweenness,random_greedy,counting]
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as im
import json
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from tddn.backend.TDD import Clear_TDD, cont
from tddn.baselines.planners import (
    COTENGRA_METHODS,
    cotengra_path,
    counting_path,
    path_from_contraction_order,
)
from tddn.circuits.build import build_network
from tddn.circuits.qasm import read_list
from tddn.network.tdd_table import tdd_table
from tddn.verify.tdd_identity import check_final_tdd

FIELDS = ["circuit", "qubits", "method", "repeat", "steps", "peak_tdd_nodes",
          "search_s", "cont_s", "total_s", "status", "identity_status"]
DEFAULT_METHODS = [*COTENGRA_METHODS, "counting"]
METHOD_NAMES = {"kahypar": "KaHyPar", "greedy": "Greedy", "betweenness": "Betweenness",
                "random_greedy": "Random Greedy", "counting": "Counting"}


class RowTimeout(BaseException):
    """A row exceeded its time limit."""


class NodeCapExceeded(Exception):
    """An intermediate TDD exceeded the node cap."""


def execute_path(net, path, node_cap: int) -> tuple[int, float, object]:
    """Contract along a path of list positions; returns ``(peak, cont_s, final_tdd)``.

    Requires a fresh TDD session (``net.reinit_session()``).
    """

    tdds = [entry[0] for entry in tdd_table(net.tn).values()]
    peak, cont_s = 0, 0.0
    for i, j in path:
        a, b = tdds[i], tdds[j]
        start = time.perf_counter()
        result = cont(a, b)
        cont_s += time.perf_counter() - start
        nodes = result.node_number()
        peak = max(peak, nodes)
        if nodes > node_cap:
            raise NodeCapExceeded(f"{nodes} > {node_cap}")
        tdds.pop(max(i, j))
        tdds.pop(min(i, j))
        tdds.append(result)
    if len(tdds) != 1:
        raise RuntimeError(f"path left {len(tdds)} tensors")
    return peak, cont_s, tdds[0]


def run_row(circuit: str | Path, method: str, repeat: int, timeout_s: int,
            search_max_time: int, node_cap: int) -> dict:
    """One baseline row; always clears the TDD session."""

    row = {"circuit": Path(circuit).name, "qubits": None, "method": method,
           "repeat": repeat, "steps": None, "peak_tdd_nodes": None, "search_s": None,
           "cont_s": None, "total_s": None, "status": "error", "identity_status": None}

    def _raise(signum, frame):
        raise RowTimeout

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(timeout_s)
    start = time.perf_counter()
    try:
        net = build_network(circuit)
        row["qubits"] = net.num_qubits
        t_search = time.perf_counter()
        if method == "counting":
            initial = [f"tdd_{i}" for i in range(len(net.tn.tensors))]
            path = path_from_contraction_order(initial, counting_path(net))
        else:
            path = cotengra_path(net, method, max_time=search_max_time)
        row["search_s"] = round(time.perf_counter() - t_search, 6)
        row["steps"] = len(path)

        net.reinit_session()
        peak, cont_s, final_tdd = execute_path(net, path, node_cap)
        row["peak_tdd_nodes"] = peak
        row["cont_s"] = round(cont_s, 6)
        info = {"fully_contracted": True, "tdds": {"final": [final_tdd, 1, []]}}
        row["identity_status"] = check_final_tdd(info, num_qubits=net.num_qubits)["status"]
        row["status"] = "ok"
    except RowTimeout:
        row["status"] = f"timeout>{timeout_s}s"
    except NodeCapExceeded as exc:
        row["status"] = f"node_cap: {exc}"
    except Exception as exc:
        row["status"] = f"error: {str(exc)[:120]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        row["total_s"] = round(time.perf_counter() - start, 6)
        Clear_TDD()
    return row


def run_sweep(circuits: list, out_dir: str | Path, methods: list[str], repeats: int = 3,
              timeout_s: int = 120, search_max_time: int = 10,
              node_cap: int = 200_000) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    done = set()
    if csv_path.exists():
        with csv_path.open() as fh:
            done = {(r["circuit"], r["method"], r["repeat"]) for r in csv.DictReader(fh)}

    (out_dir / "provenance.json").write_text(json.dumps({
        "methods": methods, "repeats": repeats, "timeout_s": timeout_s,
        "search_max_time": search_max_time, "node_cap": node_cap,
        "num_circuits": len(circuits), "started_at": datetime.now(UTC).isoformat(),
        "versions": {p: im.version(p) for p in ("cotengra", "kahypar", "qiskit", "numpy")},
    }, indent=2))

    contradictions = 0
    jobs = [(c, m, str(r)) for c in circuits for m in methods
            for r in range(1, (1 if m == "counting" else repeats) + 1)]
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for k, (circuit, method, rep) in enumerate(jobs, 1):
            if (Path(circuit).name, method, rep) in done:
                continue
            row = run_row(circuit, method, int(rep), timeout_s, search_max_time, node_cap)
            writer.writerow(row)
            fh.flush()
            if row["status"] == "ok" and row["identity_status"] != "identity":
                contradictions += 1
                print(f"[{k}/{len(jobs)}] NOT IDENTITY {row['circuit']} {method}#{rep}: "
                      f"{row['identity_status']}", flush=True)
            else:
                print(f"[{k}/{len(jobs)}] {row['circuit']} {method}#{rep}: "
                      f"{row['status']} peak={row['peak_tdd_nodes']}", flush=True)

    summary = {"jobs": len(jobs), "not_identity": contradictions, "csv": str(csv_path)}
    print(f"\nbaseline sweep done: {summary}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", default=None, help="file listing the circuits")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120, help="seconds per row")
    ap.add_argument("--search-max-time", type=int, default=10,
                    help="cotengra search budget in seconds")
    ap.add_argument("--node-cap", type=int, default=200_000)
    ap.add_argument("circuits", nargs="*", help="circuit files (in addition to --list)")
    args = ap.parse_args(argv)

    circuits = (read_list(args.list) if args.list else []) + args.circuits
    if not circuits:
        raise SystemExit("give circuits with --list and/or as arguments")
    methods = args.methods.split(",")
    unknown = set(methods) - set(METHOD_NAMES)
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}; choose from {list(METHOD_NAMES)}")
    summary = run_sweep(circuits, args.out, methods, repeats=args.repeats,
                        timeout_s=args.timeout, search_max_time=args.search_max_time,
                        node_cap=args.node_cap)
    return 1 if summary["not_identity"] else 0


if __name__ == "__main__":
    sys.exit(main())
