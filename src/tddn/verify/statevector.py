"""Statevector probe for (non-)equivalence.

If |<psi|U|psi>| < 1 for a Haar-random |psi>, then U is not the identity up
to a global phase: a failing probe proves non-equivalence; a passing probe is
evidence of equivalence. Cost: one 2^n statevector (about 134 MB at 23 qubits).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

DEFAULT_SEED = 20260802


def _flatten(qc: QuantumCircuit) -> QuantumCircuit:
    """Expand composite gates on more than two qubits into their definitions.

    ``Statevector.evolve`` builds a dense operator per instruction, which is
    prohibitive for wide composite gates.
    """

    for _ in range(16):
        big = {inst.operation.name for inst in qc.data
               if inst.operation.num_qubits > 2 and inst.operation.definition is not None}
        if not big:
            break
        qc = qc.decompose(gates_to_decompose=sorted(big))
    return qc


def identity_overlap(qc: QuantumCircuit, seed: int = DEFAULT_SEED) -> float:
    """|<psi|U|psi>| for a Haar-random |psi|; 1.0 iff U acts as a global phase on it."""

    rng = np.random.default_rng(seed)
    dim = 2 ** qc.num_qubits
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec /= np.linalg.norm(vec)
    evolved = Statevector(vec).evolve(_flatten(qc))
    return float(abs(np.vdot(vec, evolved.data)))


def is_provably_nonequivalent(qc: QuantumCircuit, tol: float = 1e-6,
                              seed: int = DEFAULT_SEED) -> bool:
    """True when the combined circuit C.(C')^-1 is proven not to be the identity."""

    return identity_overlap(qc, seed=seed) < 1.0 - tol


def pair_overlap(qc1: QuantumCircuit, qc2: QuantumCircuit,
                 seed: int = DEFAULT_SEED) -> float:
    """|<C' psi|C psi>| for a Haar-random |psi>; 1.0 iff C and C' agree on it up to phase.

    The two circuits are evolved separately and never composed into a miter.
    """

    if qc1.num_qubits != qc2.num_qubits:
        raise ValueError(
            f"qubit-count mismatch: {qc1.num_qubits} vs {qc2.num_qubits}"
        )
    rng = np.random.default_rng(seed)
    dim = 2 ** qc1.num_qubits
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec /= np.linalg.norm(vec)
    start = Statevector(vec)
    return float(abs(np.vdot(start.evolve(_flatten(qc2)).data,
                             start.evolve(_flatten(qc1)).data)))


def is_provably_nonequivalent_pair(qc1: QuantumCircuit, qc2: QuantumCircuit,
                                   tol: float = 1e-6,
                                   seed: int = DEFAULT_SEED) -> bool:
    """True when C and C' are proven not equivalent (up to global phase)."""

    return pair_overlap(qc1, qc2, seed=seed) < 1.0 - tol
