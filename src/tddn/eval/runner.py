"""Evaluate a checkpoint on a set of circuits (deterministic policy).

For each circuit: build the network, start a fresh TDD session, roll out the
policy in evaluation mode, and check the final TDD against the identity.
Rows are appended to ``results.csv`` (re-running resumes) and a
``provenance.json`` is written next to it. At the end the first ``--recheck``
circuits are rolled out again and must reproduce their peaks exactly.

Columns:
    peak_tdd_nodes   largest intermediate TDD (node count)
    episode_s        rollout time: policy inference plus contractions
    predict_s        policy inference only
    cont_s           contractions only
    build_s          network construction (not part of episode_s)
    verify_s         identity check (not part of episode_s)
    identity_status  ``identity`` for an equivalent miter

CLI:
    python -m tddn.eval.runner --checkpoint models/two_stage.zip \\
        --out results/two_stage/benchmarks --list lists/benchmark_5to23q.txt
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as im
import json
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from tddn.backend.TDD import Clear_TDD
from tddn.circuits.build import build_network
from tddn.circuits.qasm import read_list
from tddn.config import M_MAX, N_MAX
from tddn.envs.contraction_env import TensorContractionEnv
from tddn.verify.tdd_identity import check_final_tdd

FIELDS = ["circuit", "qubits", "steps", "peak_tdd_nodes", "rollout_s",
          "predict_s", "cont_s", "status", "identity_status", "dense_agrees",
          "build_s", "episode_s", "verify_s"]
MAX_STEPS = 2000


class EvalTimeout(BaseException):
    """A circuit exceeded its time limit."""


def evaluate_circuit(model, circuit_path: str | Path, bounds: tuple[int, int] = (N_MAX, M_MAX),
                     timeout_s: int = 600, cont_timeout_s: int = 60,
                     deterministic: bool = True) -> dict:
    """Roll out ``model`` on one circuit; always clears the TDD session.

    A contraction that exceeds ``cont_timeout_s`` ends the episode, which is
    reported as ``status="incomplete"``.
    """

    row = {"circuit": Path(circuit_path).name, "qubits": None, "steps": 0,
           "peak_tdd_nodes": None, "rollout_s": None, "predict_s": None,
           "cont_s": None, "status": "error", "build_s": None, "episode_s": None,
           "verify_s": None, "identity_status": None, "dense_agrees": None}

    def _raise(signum, frame):
        raise EvalTimeout

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(timeout_s)
    start = time.perf_counter()
    try:
        net = build_network(circuit_path)
        row["qubits"] = net.num_qubits
        nodes = net.graph.number_of_nodes()
        edges = net.graph.number_of_edges()
        if nodes > bounds[0] or edges > bounds[1]:
            row["status"] = f"over_bounds({nodes}n/{edges}e)"
            return row
        net.reinit_session()
        env = TensorContractionEnv(net.graph, net.tn, max_nodes=bounds[0],
                                   max_edges=bounds[1], cont_timeout_s=cont_timeout_s)
        env.set_test_mode(True)
        row["build_s"] = round(time.perf_counter() - start, 6)
        t_episode = time.perf_counter()
        obs, _ = env.reset()
        done = False
        predict_s = 0.0
        while not done and row["steps"] < MAX_STEPS:
            t_pred = time.perf_counter()
            action, _ = model.predict(obs, deterministic=deterministic)
            predict_s += time.perf_counter() - t_pred
            obs, _, done, truncated, _ = env.step(action)
            row["steps"] += 1
            if truncated:
                break
        row["episode_s"] = round(time.perf_counter() - t_episode, 6)
        row["predict_s"] = round(predict_s, 6)
        row["cont_s"] = round(env.total_contraction_time, 6)
        info = env.env_info()
        row["peak_tdd_nodes"] = info["max_tdd_nodes"]
        row["status"] = "ok" if info["fully_contracted"] else "incomplete"
        t_verify = time.perf_counter()
        verdict = check_final_tdd(info, num_qubits=net.num_qubits)
        row["verify_s"] = round(time.perf_counter() - t_verify, 6)
        row["identity_status"] = verdict["status"]
        row["dense_agrees"] = verdict["dense_agrees"]
    except EvalTimeout:
        row["status"] = f"timeout>{timeout_s}s"
    except Exception as exc:
        row["status"] = f"error: {str(exc)[:120]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        row["rollout_s"] = round(time.perf_counter() - start, 3)
        Clear_TDD()
    return row


def run_eval(checkpoint: str | Path, circuits: list, out_dir: str | Path,
             bounds: tuple[int, int] = (N_MAX, M_MAX), timeout_s: int = 600,
             cont_timeout_s: int = 60, recheck: int = 3) -> dict:
    """Evaluate ``checkpoint`` on ``circuits``; rows are appended to ``results.csv``."""

    from tddn.agents.loading import load_policy

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    done_circuits = set()
    if csv_path.exists():
        with csv_path.open() as fh:
            done_circuits = {r["circuit"] for r in csv.DictReader(fh)}

    (out_dir / "provenance.json").write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "bounds": list(bounds),
        "timeout_s": timeout_s,
        "num_circuits": len(circuits),
        "started_at": datetime.now(UTC).isoformat(),
        "versions": {p: im.version(p) for p in ("torch", "stable_baselines3", "qiskit", "numpy")},
    }, indent=2))

    model = load_policy(checkpoint)
    new_file = not csv_path.exists()
    completed, not_identity = [], 0
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, circuit in enumerate(circuits, 1):
            name = Path(circuit).name
            if name in done_circuits:
                continue
            row = evaluate_circuit(model, circuit, bounds, timeout_s,
                                   cont_timeout_s=cont_timeout_s)
            writer.writerow(row)
            fh.flush()
            if row["status"] == "ok":
                completed.append((circuit, row["peak_tdd_nodes"]))
                if row["identity_status"] != "identity":
                    not_identity += 1
            print(f"[{i}/{len(circuits)}] {name}: {row['status']} "
                  f"peak={row['peak_tdd_nodes']} identity={row['identity_status']}", flush=True)

    deterministic_ok = True
    for circuit, first_peak in completed[:recheck]:
        row = evaluate_circuit(model, circuit, bounds, timeout_s, cont_timeout_s=cont_timeout_s)
        if row["peak_tdd_nodes"] != first_peak:
            deterministic_ok = False
            print(f"rerun differs on {Path(circuit).name}: "
                  f"{first_peak} vs {row['peak_tdd_nodes']}", flush=True)

    summary = {"completed": len(completed), "not_identity": not_identity,
               "deterministic": deterministic_ok, "csv": str(csv_path)}
    print(f"\neval done: {summary}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", default=None, help="file listing the circuits")
    ap.add_argument("--bounds", default=f"{N_MAX},{M_MAX}", help="observation bounds 'nodes,edges'")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per circuit")
    ap.add_argument("--cont-timeout", type=int, default=60,
                    help="seconds per contraction; exceeding it ends the episode")
    ap.add_argument("--recheck", type=int, default=3)
    ap.add_argument("circuits", nargs="*", help="circuit files (in addition to --list)")
    args = ap.parse_args(argv)

    circuits = (read_list(args.list) if args.list else []) + args.circuits
    if not circuits:
        raise SystemExit("give circuits with --list and/or as arguments")
    bounds = tuple(int(x) for x in args.bounds.split(","))
    summary = run_eval(args.checkpoint, circuits, args.out, bounds, timeout_s=args.timeout,
                       cont_timeout_s=args.cont_timeout, recheck=args.recheck)
    return 0 if summary["not_identity"] == 0 and summary["deterministic"] else 1


if __name__ == "__main__":
    sys.exit(main())
