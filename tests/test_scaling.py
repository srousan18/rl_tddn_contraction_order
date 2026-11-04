"""Scaling rows reproduce reference peaks (deterministic methods)."""

import pytest

from tddn.config import BENCHMARK_DIR, MODELS
from tddn.eval.scaling import run_config


@pytest.fixture(scope="module")
def model_state():
    path = MODELS["model1_curriculum"]
    if not path.exists():
        pytest.skip("checkpoint not present")
    return {"model": None, "path": str(path)}


@pytest.mark.parametrize("circuit, method, peak, prefix_steps", [
    ("combined_dj_45q.qasm", "rl", 137, 0),         # within the bounds: policy only
    ("combined_dj_45q.qasm", "counting", 475, 0),
    ("combined_wstate_24q.qasm", "rl", 144, 6),     # beyond the bounds: Counting prefix + policy
])
def test_reference_rows(model_state, circuit, method, peak, prefix_steps):
    row = run_config(str(BENCHMARK_DIR / circuit), method, method == "rl", model_state, 120)
    assert row["status"] == "ok"
    assert row["identity_status"] == "identity"
    assert row["peak_tdd_nodes"] == peak
    assert row["prefix_steps"] == prefix_steps
