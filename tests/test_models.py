"""Released checkpoints load and reproduce their reference peaks exactly."""

import csv
from pathlib import Path

import pytest

from tddn.config import BENCHMARK_DIR, M_MAX, MODELS, N_MAX

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ["combined_bv_12q.qasm", "combined_dj_9q.qasm", "combined_ghz_11q.qasm",
          "combined_qnn_8q.qasm", "combined_wstate_10q.qasm", "combined_wstate_17q.qasm"]


@pytest.mark.parametrize("name", list(MODELS))
def test_reference_peaks(name):
    ckpt = MODELS[name]
    if not ckpt.exists():
        pytest.skip(f"{ckpt.name} not present")
    from tddn.agents.loading import load_policy
    from tddn.eval.runner import evaluate_circuit

    ref_csv = ROOT / "reference_results" / name / "benchmarks" / "results.csv"
    expected = {r["circuit"]: r for r in csv.DictReader(open(ref_csv))}
    model = load_policy(ckpt)
    assert model.observation_space["node_features"].shape == (N_MAX, 4)
    assert str(model.action_space) == f"Discrete({M_MAX})"
    for circuit in SAMPLE:
        row = evaluate_circuit(model, BENCHMARK_DIR / circuit, timeout_s=300)
        assert row["status"] == "ok" and row["identity_status"] == "identity", row
        assert str(row["peak_tdd_nodes"]) == expected[circuit]["peak_tdd_nodes"], (
            f"{name} {circuit}: {row['peak_tdd_nodes']} vs {expected[circuit]['peak_tdd_nodes']}")
