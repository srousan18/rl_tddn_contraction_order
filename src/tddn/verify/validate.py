"""Equivalence validation of a circuit set with MQT QCEC and the statevector probe.

Modes:

``pairs``     separate pairs ``<root>/alg/<name>.qasm`` and
              ``<root>/compiled/<name>.qasm``, checked as a pair.
``combined``  miters ``C . (C')^-1`` (``<root>/combined`` or explicit files),
              checked against the identity.

QCEC runs with its default configuration and, if inconclusive, once more
without the ZX checker. Each QCEC call runs in a child process that is
stopped at the time budget (recorded as ``no_information``). Up to
``--sv-max-qubits`` the statevector probe runs as well; an overlap below
1 - 1e-6 proves non-equivalence. Results are appended to ``results.csv``
(re-running resumes). Exit status 1 if any instance is proven not equivalent.

CLI:
    python -m tddn.verify.validate pairs --root data/circuits/benchmarks \\
        --out results/validation/benchmark_pairs
    python -m tddn.verify.validate combined --root data/circuits/benchmarks \\
        --max-qubits 23 --out results/validation/benchmark_miters
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from qiskit import QuantumCircuit

from tddn.verify.qcec import (
    VERDICT_NO_INFORMATION,
    VERDICT_NOT_EQUIVALENT,
    qcec_verify,
    qcec_verify_identity,
)
from tddn.verify.statevector import identity_overlap, pair_overlap

FIELDS = ["instance", "qubits", "mode", "verdict", "qcec_verdict", "qcec_raw",
          "qcec_checker", "qcec_config", "qcec_s", "sv_overlap", "files"]
SV_TOL = 1e-6


def _qcec_child(mode, files, timeout, config, conn):
    """Child-process body: one QCEC run; sends the result dict back."""
    try:
        if mode == "pairs":
            res = qcec_verify(files[0], files[1], timeout=timeout, **config)
        else:
            res = qcec_verify_identity(files[0], timeout=timeout, **config)
        conn.send(res)
    except Exception as exc:
        conn.send({"verdict": VERDICT_NO_INFORMATION, "raw_equivalence": f"error: {exc}"[:80],
                   "check_time": None, "decisive_checker": "none"})


def _qcec_guarded(mode, files, timeout, config) -> dict:
    """QCEC in a killable child; hard budget = timeout + 30 s."""
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_qcec_child,
                       args=(mode, [str(f) for f in files], timeout, config, send))
    proc.start()
    if recv.poll(timeout + 30):
        res = recv.recv()
        proc.join()
        return res
    proc.kill()
    proc.join()
    return {"verdict": VERDICT_NO_INFORMATION, "raw_equivalence": f"killed>{timeout + 30}s",
            "check_time": None, "decisive_checker": "none"}


def validate_one(mode: str, files: list[Path], timeout: int, sv_max_qubits: int) -> dict:
    qcs = [QuantumCircuit.from_qasm_file(str(f)) for f in files]
    n = qcs[0].num_qubits
    t = time.perf_counter()
    # Default configuration first; if inconclusive, retry without ZX.
    res, config = _qcec_guarded(mode, files, timeout, {}), "default"
    if res["verdict"] == VERDICT_NO_INFORMATION:
        res, config = _qcec_guarded(mode, files, timeout, {"run_zx_checker": False}), "no_zx"
    qcec_s = time.perf_counter() - t

    overlap = None
    if n <= sv_max_qubits:
        overlap = (pair_overlap(qcs[0], qcs[1]) if mode == "pairs"
                   else identity_overlap(qcs[0]))

    # Zero tolerance: any proof of difference wins; equivalence needs QCEC.
    if res["verdict"] == VERDICT_NOT_EQUIVALENT or (
            overlap is not None and overlap < 1.0 - SV_TOL):
        verdict = VERDICT_NOT_EQUIVALENT
    else:
        verdict = res["verdict"]
    return {"instance": files[0].stem, "qubits": n, "mode": mode,
            "verdict": verdict, "qcec_verdict": res["verdict"],
            "qcec_raw": res["raw_equivalence"],
            "qcec_checker": res["decisive_checker"], "qcec_config": config,
            "qcec_s": round(qcec_s, 4),
            "sv_overlap": None if overlap is None else round(overlap, 10),
            "files": ";".join(str(f) for f in files)}


def _qubits(path: Path) -> int | None:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1].rstrip("q"))
    except (IndexError, ValueError):
        return None


def collect(mode: str, root: Path | None, files: list[str],
            min_q: int | None, max_q: int | None) -> list[list[Path]]:
    if mode == "pairs":
        jobs = []
        for a in sorted((root / "alg").glob("*.qasm")):
            b = root / "compiled" / a.name
            if b.exists():
                jobs.append([a, b])
    else:
        paths = ([Path(f) for f in files] if files
                 else sorted((root / "combined").glob("*.qasm")))
        jobs = [[p] for p in paths]

    def keep(p):
        q = _qubits(p)
        return q is None or ((min_q is None or q >= min_q)
                             and (max_q is None or q <= max_q))
    return [j for j in jobs if keep(j[0])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pairs", "combined"])
    ap.add_argument("--root", default="data/circuits/benchmarks",
                    help="corpus root holding alg/, compiled/ and combined/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=120, help="QCEC seconds per run")
    ap.add_argument("--sv-max-qubits", type=int, default=20)
    ap.add_argument("--min-qubits", type=int, default=None)
    ap.add_argument("--max-qubits", type=int, default=None)
    ap.add_argument("files", nargs="*", help="combined mode: explicit files")
    args = ap.parse_args(argv)

    jobs = collect(args.mode, Path(args.root), args.files,
                   args.min_qubits, args.max_qubits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    done = set()
    if csv_path.exists():
        with csv_path.open() as fh:
            done = {r["instance"] for r in csv.DictReader(fh)}

    import importlib.metadata as im
    (out / "provenance.json").write_text(json.dumps({
        "mode": args.mode, "root": args.root, "timeout_s": args.timeout,
        "sv_max_qubits": args.sv_max_qubits, "sv_tol": SV_TOL,
        "num_instances": len(jobs),
        "started_at": datetime.now(UTC).isoformat(),
        "versions": {p: im.version(p) for p in ("mqt.qcec", "qiskit", "numpy")},
    }, indent=2))

    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for k, files in enumerate(jobs, 1):
            if files[0].stem in done:
                continue
            row = validate_one(args.mode, files, args.timeout, args.sv_max_qubits)
            writer.writerow(row)
            fh.flush()
            print(f"[{k}/{len(jobs)}] {row['instance']} ({row['qubits']}q): "
                  f"{row['verdict']} [qcec {row['qcec_raw']}/{row['qcec_config']}, "
                  f"sv {row['sv_overlap']}]", flush=True)

    with csv_path.open() as fh:
        verdicts = Counter(r["verdict"] for r in csv.DictReader(fh))
    print(f"\nvalidation done ({csv_path}): {dict(verdicts)}", flush=True)
    return 1 if verdicts.get(VERDICT_NOT_EQUIVALENT) else 0


if __name__ == "__main__":
    sys.exit(main())
