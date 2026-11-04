"""Evaluate RL checkpoints on a list of miters, equivalent or not.

Used for the non-equivalence set (``lists/nonequivalence.txt``: 304
non-equivalent miters and 114 equivalent ones) and the unstructured random
set (``lists/random_unstructured.txt``). Each row goes through
``tddn.eval.scaling.run_config``: a hard time limit per row (default 120 s),
the Counting prefix if a graph exceeds the policy bounds, and a
classification of the final TDD. Equivalent miters should end ``identity``,
non-equivalent ones ``not_identity``. ``mode`` is the circuit's directory
name (drop_gate, flip_cnot, stray_gate, cross_instance,
cross_instance_control, combined, random_unstructured).

Rows are appended to ``results.csv``; re-running resumes.

CLI:
    python -m tddn.eval.fileset --out results/two_stage/nonequivalence \\
        --model two_stage=models/two_stage.zip --list lists/nonequivalence.txt
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from tddn.circuits.qasm import NAME_RE, read_list
from tddn.eval.scaling import resolve_path, run_config

FIELDS = ["model", "mode", "family", "qubits", "file", "peak_tdd_nodes", "total_time_s",
          "plan_time_s", "contract_time_s", "graph_nodes_before", "graph_edges_before",
          "prefix_activated", "status", "identity_status"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--list", required=True)
    ap.add_argument("--timeout", type=int, default=120, help="seconds per row")
    args = ap.parse_args(argv)

    files = read_list(args.list)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    done = ({(r["model"], r["file"]) for r in csv.DictReader(open(csv_path))}
            if csv_path.exists() else set())
    fresh = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        for spec in args.model:
            label, _, path = spec.partition("=")
            if not Path(path).is_file():
                raise SystemExit(f"checkpoint not found: {path}")
            state = {"model": None, "path": path}
            t0 = time.time()
            for f in files:
                if (label, f) in done:
                    continue
                m = NAME_RE.match(Path(f).name)
                row = run_config(resolve_path(f), "rl", True, state, args.timeout)
                row.update(model=label, file=f, mode=Path(f).parent.name,
                           family=m["family"] if m else "", qubits=m["qubits"] if m else "")
                w.writerow({k: row.get(k, "") for k in FIELDS})
                fh.flush()
            print(f"{label}: {len(files)} files in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
