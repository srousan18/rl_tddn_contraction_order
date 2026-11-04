"""The declared open indices of a miter are exactly its dangling legs (2n)."""

from collections import Counter

import pytest

from tddn.backend.TDD_Q import TNtoCotInput
from tddn.circuits.build import build_network
from tddn.config import BENCHMARK_DIR

CIRCUITS = ["combined_ghz_6q.qasm", "combined_wstate_8q.qasm", "combined_bv_6q.qasm"]


@pytest.mark.parametrize("name", CIRCUITS)
def test_declared_open_indices_are_the_dangling_legs(name):
    net = build_network(BENCHMARK_DIR / name)
    tensor_list, open_indices, _, _, _ = TNtoCotInput(net.tn, net.num_qubits)
    counts = Counter(i for t in tensor_list for i in t)
    dangling = {i for i, c in counts.items() if c == 1}
    assert set(open_indices) == dangling
    assert len(open_indices) == 2 * net.num_qubits
