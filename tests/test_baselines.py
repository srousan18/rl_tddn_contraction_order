"""Baseline planners: declared open indices and execution of a path."""

from pathlib import Path

import pytest

from tddn.backend.TDD_Q import TNtoCotInput
from tddn.baselines.planners import cot_input
from tddn.baselines.runner import run_row
from tddn.circuits.build import build_network
from tddn.config import BENCHMARK_DIR

FIX = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["combined_ghz_6q.qasm", "combined_wstate_8q.qasm"])
def test_cot_input_declares_all_legs(name):
    net = build_network(BENCHMARK_DIR / name)
    n = net.num_qubits
    _, open_indices, _ = cot_input(net.tn, n)
    assert open_indices == TNtoCotInput(net.tn, n)[1]
    assert set(open_indices) == {f"x{i}" for i in range(n)} | {f"y{i}" for i in range(n)}


def test_counting_row_executes_to_identity():
    row = run_row(BENCHMARK_DIR / "combined_dj_6q.qasm", "counting", 1,
                  timeout_s=60, search_max_time=1, node_cap=200_000)
    assert row["status"] == "ok"
    assert row["identity_status"] == "identity"
    assert row["peak_tdd_nodes"] > 0


def test_counting_row_rejects_nonequivalent():
    row = run_row(FIX / "nonequiv_ghz_5q.qasm", "counting", 1,
                  timeout_s=60, search_max_time=1, node_cap=200_000)
    assert row["status"] == "ok"
    assert row["identity_status"] != "identity"


def test_cotengra_row_executes_to_identity():
    pytest.importorskip("cotengra")
    row = run_row(BENCHMARK_DIR / "combined_ghz_6q.qasm", "greedy", 1, timeout_s=60,
                  search_max_time=1, node_cap=200_000)
    assert row["status"] == "ok", row
    assert row["identity_status"] == "identity"
