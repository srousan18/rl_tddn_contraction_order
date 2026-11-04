"""Two-qubit gates outside the supported set are decomposed before tensorization."""

from itertools import permutations

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from tddn.backend.TDD import Clear_TDD, cont
from tddn.circuits.build import network_from_circuit
from tddn.circuits.qasm import combine_circuits
from tddn.network.tdd_table import tdd_table
from tddn.verify.tdd_identity import check_final_tdd


def contract(qc):
    net = network_from_circuit(qc)
    entries = list(tdd_table(net.tn).values())
    acc = entries[0][0]
    for e in entries[1:]:
        acc = cont(acc, e[0])
    return net, acc


def assert_matches_operator(qc):
    """The contracted 4-leg tensor equals the operator under some leg ordering (up to phase)."""

    _, acc = contract(qc)
    arr = np.asarray(acc.to_array(), dtype=np.complex128)
    Clear_TDD()
    U = Operator(qc).data

    def phase_match(M, cand):
        pos = int(np.argmax(np.abs(cand)))
        if abs(M.flat[pos]) < 1e-12:
            return False
        s = cand.flat[pos] / M.flat[pos]
        return abs(abs(s) - 1) < 1e-6 and np.allclose(M * s, cand, atol=1e-7)

    assert any(phase_match(arr.transpose(p).reshape(4, 4), cand)
               for p in permutations(range(4)) for cand in (U, U.T)), \
        f"no leg ordering matches the operator of {[i.operation.name for i in qc.data]}"


@pytest.mark.parametrize("build", [
    lambda qc: (qc.x(0), qc.cry(1.1, 0, 1)),
    lambda qc: (qc.cry(1.1, 0, 1), qc.x(0)),
    lambda qc: (qc.h(0), qc.cry(-2.3, 1, 0)),
    lambda qc: (qc.x(0), qc.crz(1.1, 0, 1)),
    lambda qc: (qc.x(0), qc.cp(1.1, 0, 1)),
    lambda qc: (qc.x(0), qc.rzz(0.7, 0, 1)),
])
def test_two_qubit_compositions_match_operator(build):
    qc = QuantumCircuit(2)
    build(qc)
    assert_matches_operator(qc)


def test_cry_cascade_miter_contracts_to_identity():
    alg = QuantumCircuit(4)
    alg.x(0)
    for i in range(3):
        alg.cry(0.9 + 0.3 * i, i, i + 1)
        alg.cx(i + 1, i)
    comp = transpile(alg, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    net, acc = contract(combine_circuits(alg, comp))
    verdict = check_final_tdd({"fully_contracted": True, "tdds": {"final": [acc, 1, []]}},
                              num_qubits=net.num_qubits)
    assert verdict["status"] == "identity", verdict
    assert verdict["dense_agrees"] is True
    Clear_TDD()
