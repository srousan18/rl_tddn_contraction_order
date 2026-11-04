"""Saving circuit pairs (C, C') with an MQT QCEC equivalence check.

Layout under a corpus root:

    <root>/alg/<family>_<n>q.qasm                  C
    <root>/compiled/<family>_<n>q.qasm             C'
    <root>/combined/combined_<family>_<n>q.qasm    C . (C')^-1

Each accepted pair appends a record to ``<root>/generation_log.jsonl``.
Requires ``mqt.qcec`` (``pip install -e ".[verify]"``).
"""

from __future__ import annotations

import importlib.metadata as im
import json
from pathlib import Path

import qiskit
from qiskit import QuantumCircuit, qasm2

from tddn.circuits.qasm import combine_circuits
from tddn.verify.qcec import qcec_verify


class PairVerificationError(RuntimeError):
    """The pair is not proven equivalent; its files were removed."""


def save_verified_pair(qc_alg: QuantumCircuit, qc_native: QuantumCircuit, family: str,
                       num_qubits: int, root: str | Path, timeout: int = 120,
                       provenance: dict | None = None) -> dict:
    """Write C and C', check them with QCEC from disk, then write the miter.

    Existing files are never overwritten. On a failed check every file written
    for this pair is deleted and ``PairVerificationError`` is raised.
    """

    root = Path(root)
    base = f"{family}_{num_qubits}q.qasm"
    alg_path = root / "alg" / base
    comp_path = root / "compiled" / base
    combined_path = root / "combined" / f"combined_{base}"

    clashes = [p for p in (alg_path, comp_path, combined_path) if p.exists()]
    if clashes:
        raise FileExistsError(f"refusing to overwrite {', '.join(map(str, clashes))}")

    for p in (alg_path, comp_path, combined_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    alg_path.write_text(qasm2.dumps(qc_alg))
    comp_path.write_text(qasm2.dumps(qc_native))
    written = [alg_path, comp_path]
    try:
        verdict = qcec_verify(alg_path, comp_path, timeout=timeout)
        if verdict["verdict"] != "equivalent":
            raise PairVerificationError(
                f"{family}_{num_qubits}q: QCEC verdict {verdict['verdict']} "
                f"({verdict['raw_equivalence']})")
        combined = combine_circuits(QuantumCircuit.from_qasm_file(str(alg_path)),
                                    QuantumCircuit.from_qasm_file(str(comp_path)))
        combined_path.write_text(qasm2.dumps(combined))
        written.append(combined_path)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise

    record = {
        "family": family,
        "num_qubits": num_qubits,
        **(provenance or {}),
        "verdict": verdict["verdict"],
        "raw_equivalence": verdict["raw_equivalence"],
        "decisive_checker": verdict["decisive_checker"],
        "files": [str(p) for p in written],
        "mqt_qcec": im.version("mqt.qcec"),
        "qiskit": qiskit.__version__,
    }
    with (root / "generation_log.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record
