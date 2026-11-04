"""Separate pairs (C, C'): the miter contracts to the identity exactly when the pair is equivalent."""

import time
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from tddn.backend.TDD import Clear_TDD, cont
from tddn.circuits.build import build_network_from_pair
from tddn.config import DATA_DIR
from tddn.network.tdd_table import tdd_table
from tddn.verify.statevector import identity_overlap, pair_overlap
from tddn.verify.tdd_identity import check_final_tdd

PAIRS = Path(__file__).parent / "fixtures" / "pairs"
EQUIVALENT = [("alg/ghz_5q.qasm", "compiled/ghz_5q.qasm"),
              ("alg/dj_6q.qasm", "compiled/dj_6q.qasm")]
NON_EQUIVALENT = [("alg/ghz_5q.qasm", "compiled_mutated/ghz_5q.qasm")]


def contract_fully(net):
    net.reinit_session()
    entries = list(tdd_table(net.tn).values())
    acc = entries[0][0]
    for e in entries[1:]:
        acc = cont(acc, e[0])
    return acc


def load(rel):
    return QuantumCircuit.from_qasm_file(str(PAIRS / rel))


def test_pair_statevector_discriminates():
    for a, b in EQUIVALENT:
        assert pair_overlap(load(a), load(b)) > 1 - 1e-9
    for a, b in NON_EQUIVALENT:
        assert pair_overlap(load(a), load(b)) < 1 - 1e-6


def test_qcec_ground_truth():
    pytest.importorskip("mqt.qcec")
    from tddn.verify.qcec import qcec_verify

    for a, b in EQUIVALENT:
        assert qcec_verify(PAIRS / a, PAIRS / b, timeout=30)["verdict"] == "equivalent"
    for a, b in NON_EQUIVALENT:
        assert qcec_verify(PAIRS / a, PAIRS / b, timeout=30)["verdict"] == "not_equivalent"


def test_miter_of_pair_matches_ground_truth():
    for pairs, expected, overlap_ok in ((EQUIVALENT, "identity", True),
                                        (NON_EQUIVALENT, "not_identity", False)):
        for a, b in pairs:
            net = build_network_from_pair(PAIRS / a, PAIRS / b)
            info = {"fully_contracted": True, "tdds": {"final": [contract_fully(net), 1, []]}}
            assert check_final_tdd(info, num_qubits=net.num_qubits)["status"] == expected
            assert (identity_overlap(net.qc) > 1 - 1e-9) == overlap_ok
            Clear_TDD()


def test_statevector_expands_wide_composite_gates():
    root = DATA_DIR / "circuits" / "benchmarks"
    alg = QuantumCircuit.from_qasm_file(str(root / "alg" / "dj_16q.qasm"))
    assert any(i.operation.num_qubits > 2 for i in alg.data)
    t = time.perf_counter()
    ov = pair_overlap(alg, QuantumCircuit.from_qasm_file(str(root / "compiled" / "dj_16q.qasm")))
    assert ov > 1 - 1e-9 and time.perf_counter() - t < 10
