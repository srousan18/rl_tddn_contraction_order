"""Circuit -> tensor network -> contraction graph.

Every consumer (training, evaluation, baselines, verification) builds its
network through this module: read QASM, tensorize, merge low-rank tensors
(``squeezeTN``), open a TDD session for the network's index order, and build
the contraction graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from qiskit import QuantumCircuit, transpile

from tddn.backend.TDD import Clear_TDD, Ini_TDD
from tddn.backend.TDD_Q import cir_2_tn_lbl, get_real_qubit_num, squeezeTN, squeezeTN_ultra
from tddn.backend.TN import TensorNetwork
from tddn.circuits.qasm import combine_circuits, read_qasm_file
from tddn.network.graph import build_networkx_graph

# Basis used to decompose gates the tensorizer does not support.
DECOMPOSITION_BASIS = ["rz", "sx", "x", "cx"]

# Two-qubit gates the tensorizer supports directly: cx, and gates that are
# diagonal and symmetric under exchange of their qubits. Other multi-qubit
# gates are decomposed first.
SUPPORTED_TWO_QUBIT = {"cx", "cz", "cp", "cu1", "rzz"}


def needs_decomposition(qc: QuantumCircuit) -> bool:
    for inst in qc.data:
        n = len(inst.qubits)
        if n > 2 or (n == 2 and inst.operation.name not in SUPPORTED_TWO_QUBIT):
            return True
    return False


@dataclass
class BuiltNetwork:
    """A tensorized circuit with its contraction graph."""

    qc: QuantumCircuit
    tn: TensorNetwork
    index_order: list
    graph: nx.Graph
    num_qubits: int

    def reinit_session(self) -> None:
        """Start a fresh TDD session for this network's index order.

        The TDD backend keeps process-global tables: call this before
        contracting and ``Clear_TDD()`` afterwards; never interleave networks.
        """

        Clear_TDD()
        Ini_TDD(index_order=self.index_order)


def build_network(path: str | Path) -> BuiltNetwork:
    """Network of a miter stored as one QASM file."""

    qc = read_qasm_file(path)
    if qc is None:
        raise ValueError(f"unreadable QASM: {path}")
    return network_from_circuit(qc)


def build_network_from_pair(circ: str | Path | QuantumCircuit,
                            circ_prime: str | Path | QuantumCircuit) -> BuiltNetwork:
    """Network of the miter ``C . (C')^-1`` built from a separate pair.

    The combined circuit is available as ``BuiltNetwork.qc``.
    """

    combined = combine_circuits(_as_circuit(circ), _as_circuit(circ_prime))
    if any(len(inst.qubits) > 2 for inst in combined.data):
        combined = transpile(combined, basis_gates=DECOMPOSITION_BASIS, optimization_level=0)
    return network_from_circuit(combined)


def network_from_circuit(qc: QuantumCircuit) -> BuiltNetwork:
    """Tensorize ``qc`` and build its contraction graph.

    Unsupported gates are decomposed with an optimization-level-0 transpile,
    which translates the basis without changing the qubit layout.
    """

    if needs_decomposition(qc):
        qc = transpile(qc, basis_gates=DECOMPOSITION_BASIS, optimization_level=0)
    tn, index_order, depth = cir_2_tn_lbl(qc)
    n = get_real_qubit_num(qc)
    tensors = squeezeTN(tn.tensors, n, depth)
    tensors = squeezeTN_ultra(tensors, n, depth)
    tn = TensorNetwork(tensors, tn.tn_type, n)
    Clear_TDD()
    Ini_TDD(index_order=index_order)
    graph = build_networkx_graph(tn)
    return BuiltNetwork(qc=qc, tn=tn, index_order=index_order, graph=graph, num_qubits=n)


def _as_circuit(circ: str | Path | QuantumCircuit) -> QuantumCircuit:
    if isinstance(circ, QuantumCircuit):
        return circ
    qc = read_qasm_file(circ)
    if qc is None:
        raise ValueError(f"unreadable QASM: {circ}")
    return qc
