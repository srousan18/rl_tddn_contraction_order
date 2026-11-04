"""The TDD identity check and the statevector probe agree with ground truth."""

from pathlib import Path

from tddn.backend.TDD import Clear_TDD, cont
from tddn.circuits.build import build_network
from tddn.circuits.qasm import read_qasm_file
from tddn.network.tdd_table import tdd_table
from tddn.verify.statevector import identity_overlap
from tddn.verify.tdd_identity import check_final_tdd

FIX = Path(__file__).parent / "fixtures"


def contract_fully(path):
    net = build_network(path)
    net.reinit_session()
    entries = list(tdd_table(net.tn).values())
    acc = entries[0][0]
    for e in entries[1:]:
        acc = cont(acc, e[0])
    return net, acc


def final_info(tdd):
    return {"fully_contracted": True, "tdds": {"final": [tdd, 1, []]}}


def test_equivalent_pair_is_identity():
    for name in ["combined_ghz_5q.qasm", "combined_dj_6q.qasm"]:
        net, tdd = contract_fully(FIX / name)
        assert check_final_tdd(final_info(tdd), num_qubits=net.num_qubits)["status"] == "identity"
        Clear_TDD()


def test_nonequivalent_pair_is_not_identity():
    for name in ["nonequiv_ghz_5q.qasm", "nonequiv_dj_6q.qasm"]:
        net, tdd = contract_fully(FIX / name)
        assert check_final_tdd(final_info(tdd), num_qubits=net.num_qubits)["status"] == "not_identity"
        Clear_TDD()


def test_statevector_probe_discriminates():
    eq = read_qasm_file(FIX / "combined_ghz_5q.qasm")
    ne = read_qasm_file(FIX / "nonequiv_ghz_5q.qasm")
    assert identity_overlap(eq) > 1 - 1e-9
    assert identity_overlap(ne) < 1 - 1e-6
