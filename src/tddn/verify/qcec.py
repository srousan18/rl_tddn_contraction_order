"""Equivalence checking with MQT QCEC, independent of the TDD code.

- ``qcec_verify(C, C')`` checks a separate pair;
- ``qcec_verify_identity(miter)`` checks a miter ``C . (C')^-1`` against the
  empty circuit on the same qubits.

Verdicts are ``equivalent``, ``not_equivalent`` or ``no_information``; an
inconclusive run (timeout, ``probably_equivalent``) is never mapped to a
definite verdict, and the raw QCEC result is kept alongside. For
rotation-heavy circuits, ``run_zx_checker=False`` lets the decision-diagram
checker decide.

Requires ``mqt.qcec`` (``pip install -e ".[verify]"``).
"""

from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit

VERDICT_EQUIVALENT = "equivalent"
VERDICT_NOT_EQUIVALENT = "not_equivalent"
VERDICT_NO_INFORMATION = "no_information"

# QCEC equivalence strings that count as proved equivalent / proved not.
_QCEC_POSITIVE = {"equivalent", "equivalent_up_to_global_phase"}
_QCEC_NEGATIVE = {"not_equivalent", "probably_not_equivalent"}

_INSTALL_HINT = 'missing optional dependency: pip install -e ".[verify]"'


def _as_circuit(circ: QuantumCircuit | str | Path) -> QuantumCircuit:
    if isinstance(circ, QuantumCircuit):
        return circ
    return QuantumCircuit.from_qasm_file(str(circ))


def map_qcec_equivalence(raw: str) -> str:
    """Map a QCEC equivalence string to a verdict."""

    if raw in _QCEC_POSITIVE:
        return VERDICT_EQUIVALENT
    if raw in _QCEC_NEGATIVE:
        return VERDICT_NOT_EQUIVALENT
    return VERDICT_NO_INFORMATION


def qcec_verify(circ1: QuantumCircuit | str | Path,
                circ2: QuantumCircuit | str | Path,
                timeout: int = 120, **config) -> dict:
    """Run MQT QCEC on a circuit pair; ``config`` goes to ``qcec.verify``
    (e.g. ``run_zx_checker=False``).

    Returns ``{"verdict", "raw_equivalence", "check_time", "decisive_checker"}``.
    A timeout yields ``verdict == "no_information"``, never a definite answer.
    """

    try:
        from mqt import qcec
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_INSTALL_HINT) from exc

    res = qcec.verify(_as_circuit(circ1), _as_circuit(circ2), timeout=timeout,
                      **config)
    data = res.json()
    raw = data.get("equivalence", VERDICT_NO_INFORMATION)

    decisive_checker = "none"
    for checker in data.get("checkers", []):
        if checker.get("equivalence") not in (None, "no_information"):
            decisive_checker = checker.get("checker", "unknown")
            break

    return {
        "verdict": map_qcec_equivalence(raw),
        "raw_equivalence": raw,
        "check_time": data.get("check_time"),
        "decisive_checker": decisive_checker,
    }


def qcec_verify_identity(combined: QuantumCircuit | str | Path,
                         timeout: int = 120, **config) -> dict:
    """QCEC verdict for a miter ``C . (C')^-1`` against the identity."""

    qc = _as_circuit(combined)
    return qcec_verify(qc, QuantumCircuit(qc.num_qubits), timeout=timeout, **config)
