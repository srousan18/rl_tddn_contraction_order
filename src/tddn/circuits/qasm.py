"""Reading circuits, building miters, and listing benchmark files.

Benchmark files are named ``combined_<family>_<n>q.qasm``; the family and the
qubit count are parsed from that name.
"""

from __future__ import annotations

import re
from pathlib import Path

from qiskit import QuantumCircuit

NAME_RE = re.compile(r"^combined_(?P<family>.+?)_(?P<qubits>\d+)q\.qasm$")


def read_qasm_file(file_path: str | Path) -> QuantumCircuit | None:
    """Parse an OpenQASM 2 file; ``None`` if it cannot be read."""

    try:
        return QuantumCircuit.from_qasm_file(str(file_path))
    except Exception as exc:
        print(f"Error reading QASM file {file_path}: {exc}")
        return None


def combine_circuits(qc1: QuantumCircuit, qc2: QuantumCircuit) -> QuantumCircuit:
    """The miter ``qc1 . qc2^-1``: identity (up to phase) iff qc1 == qc2."""

    num_qubits = max(qc1.num_qubits, qc2.num_qubits)
    combined = QuantumCircuit(num_qubits)
    combined.compose(qc1, inplace=True)
    combined.compose(qc2.inverse(), inplace=True)
    return combined


def family_and_qubits(path: str | Path) -> tuple[str, int] | None:
    """``(family, qubits)`` parsed from a ``combined_<family>_<n>q.qasm`` name."""

    m = NAME_RE.match(Path(path).name)
    return (m["family"], int(m["qubits"])) if m else None


def list_benchmarks(directory: str | Path, families=None, min_qubits=None,
                    max_qubits=None) -> list[str]:
    """Benchmark files in ``directory``, sorted by (qubits, name)."""

    out = []
    for path in Path(directory).glob("combined_*.qasm"):
        parsed = family_and_qubits(path)
        if parsed is None:
            continue
        family, qubits = parsed
        if families is not None and family not in families:
            continue
        if min_qubits is not None and qubits < min_qubits:
            continue
        if max_qubits is not None and qubits > max_qubits:
            continue
        out.append((qubits, path.name, str(path)))
    return [p for _, _, p in sorted(out)]


def read_list(path: str | Path) -> list[str]:
    """Non-empty lines of an instance list file."""

    return [line.strip() for line in open(path) if line.strip()]
