"""Structured random miters for the structured-mix curriculum.

Random circuits are added to the benchmark curriculum so that each phase pool
is 100% / 50% / ~30% random while every benchmark training circuit is kept:

    phase 1  17 random                (no benchmark circuits)
    phase 2  34 benchmark + 34 random
    phase 3  52 benchmark + 22 random (22/74 = 29.7%)

C is a generic entangling structure over the benchmarks' source gate set
{h, x, ry, p, cx, cz} with every angle a multiple of pi/4. Four families,
none of them a benchmark algorithm:

    brickwork  rotation layer, then CX/CZ on even and odd neighbour pairs
    cascade    CX/CZ chain along a random qubit order, rotations interleaved
    graph      H on all qubits, CZ on a random graph, final rotation layer
    layered    random layers of one- and two-qubit gates

C' = C compiled at optimization level 3 to {rz, sx, x, cx}, checked equal to
C with the statevector probe. The training miter is C . (C')^-1. Each random
miter is matched in graph size to a benchmark training circuit of its phase,
fits the observation bounds, and passes a Counting contraction under
``--probe-cap`` nodes within ``--probe-seconds``.

The circuits in ``data/circuits/structured_mix`` were generated with:

    python -m tddn.circuits.structured_mix --out data/circuits/structured_mix \\
        --train-list lists/benchmark_train.txt --candidates 1500
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
from pathlib import Path

from qiskit import QuantumCircuit, qasm2, transpile

from tddn.backend.TDD import Clear_TDD
from tddn.baselines.planners import counting_path, path_from_contraction_order
from tddn.baselines.runner import NodeCapExceeded, execute_path
from tddn.circuits.build import build_network
from tddn.circuits.qasm import combine_circuits, read_list
from tddn.config import M_MAX, N_MAX
from tddn.verify.statevector import pair_overlap

NATIVE = ["rz", "sx", "x", "cx"]
ANGLES = [k * math.pi / 4 for k in range(1, 8)]
FAMILIES = ["brickwork", "cascade", "graph", "layered"]
PHASE_COUNTS = {1: 17, 2: 34, 3: 22}


class ProbeTimeout(BaseException):
    """Counting probe exceeded its time budget."""


def _rot(qc, q, rng):
    g = rng.choice(["h", "x", "ry", "p"])
    if g in ("ry", "p"):
        getattr(qc, g)(rng.choice(ANGLES), q)
    else:
        getattr(qc, g)(q)


def _ent(qc, a, b, rng):
    getattr(qc, rng.choice(["cx", "cz"]))(a, b)


def make_circuit(family: str, n: int, depth: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    if family == "brickwork":
        for _ in range(depth):
            for q in range(n):
                _rot(qc, q, rng)
            for start in (0, 1):
                for q in range(start, n - 1, 2):
                    _ent(qc, q, q + 1, rng)
    elif family == "cascade":
        order = list(range(n))
        rng.shuffle(order)
        for _ in range(max(1, depth // 2)):
            for i in range(n - 1):
                if rng.random() < 0.5:
                    _rot(qc, order[i], rng)
                _ent(qc, order[i], order[i + 1], rng)
    elif family == "graph":
        qc.h(range(n))
        edges = {(i, i + 1) for i in range(n - 1)}
        for _ in range(depth):
            a, b = rng.sample(range(n), 2)
            edges.add((min(a, b), max(a, b)))
        for a, b in sorted(edges):
            qc.cz(a, b)
        for q in range(n):
            if rng.random() < 0.5:
                _rot(qc, q, rng)
    else:
        for _ in range(depth):
            qs = list(range(n))
            rng.shuffle(qs)
            i = 0
            while i < n:
                if i + 1 < n and rng.random() < 0.5:
                    _ent(qc, qs[i], qs[i + 1], rng)
                    i += 2
                else:
                    _rot(qc, qs[i], rng)
                    i += 1
    return qc


def compile_verified(C: QuantumCircuit, seed: int):
    """C compiled to the native basis and checked equal to C; ``(None, None)`` if not."""

    for attempt in range(3):
        Cp = transpile(C, basis_gates=NATIVE, optimization_level=3,
                       seed_transpiler=seed + attempt)
        if abs(pair_overlap(C, Cp) - 1.0) < 1e-6:
            return Cp, seed + attempt
    return None, None


def graph_size(path: Path):
    try:
        net = build_network(path)
        return net.graph.number_of_nodes(), net.graph.number_of_edges(), net
    except Exception:
        Clear_TDD()
        return None, None, None


def counting_probe(net, cap: int, seconds: int):
    """Peak of a Counting contraction, or ``None`` over the node cap or time budget."""

    def _raise(signum, frame):
        raise ProbeTimeout

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        init = [f"tdd_{i}" for i in range(len(net.tn.tensors))]
        path = path_from_contraction_order(init, counting_path(net))
        net.reinit_session()
        peak, _, _ = execute_path(net, path, cap)
        return peak
    except (ProbeTimeout, NodeCapExceeded):
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-list", default="lists/benchmark_train.txt",
                    help="benchmark training circuits whose curriculum the pools follow")
    ap.add_argument("--candidates", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=3030)
    ap.add_argument("--probe-cap", type=int, default=20000)
    ap.add_argument("--probe-seconds", type=int, default=10)
    ap.add_argument("--lists", default="lists/structured_mix",
                    help="prefix of the per-phase list files")
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Benchmark curriculum sizes, in the training order (node count, name).
    bench = []
    for p in read_list(args.train_list):
        nodes, _, _ = graph_size(Path(p))
        Clear_TDD()
        bench.append((nodes, Path(p).name))
    bench.sort()
    nodes_sorted = [b[0] for b in bench]
    third = len(bench) // 3
    targets = {1: nodes_sorted[:third], 2: nodes_sorted[:2 * third],
               3: [nodes_sorted[round(i * (len(bench) - 1) / (PHASE_COUNTS[3] - 1))]
                   for i in range(PHASE_COUNTS[3])]}

    cands = []
    for i in range(args.candidates):
        fam = FAMILIES[i % len(FAMILIES)]
        n, depth, seed = rng.randint(5, 23), rng.randint(1, 8), rng.randint(0, 2**31 - 1)
        C = make_circuit(fam, n, depth, seed)
        Cp, tseed = compile_verified(C, seed)
        if Cp is None:
            continue
        path = out / "candidates" / f"combined_sm{fam[:2]}x{i:05d}_{n}q.qasm"
        path.write_text(qasm2.dumps(combine_circuits(C, Cp)))
        nodes, edges, _ = graph_size(path)
        Clear_TDD()
        if nodes is None or nodes > N_MAX or edges > M_MAX:
            path.unlink()
            continue
        cands.append({"path": path, "family": fam, "qubits": n, "depth": depth,
                      "circuit_seed": seed, "transpile_seed": tseed,
                      "nodes": nodes, "edges": edges})
    print(f"candidates kept: {len(cands)} of {args.candidates}", flush=True)

    used = set()
    with open(out / "generation_log.jsonl", "w") as log:
        for phase in (1, 2, 3):
            pdir = out / f"p{phase}"
            pdir.mkdir(exist_ok=True)
            chosen = []
            for k, t in enumerate(targets[phase]):
                fam_pref = FAMILIES[k % len(FAMILIES)]
                order = sorted((c for c in cands if str(c["path"]) not in used),
                               key=lambda c: (abs(c["nodes"] - t), c["family"] != fam_pref))
                for c in order:
                    used.add(str(c["path"]))
                    _, _, net = graph_size(c["path"])
                    peak = counting_probe(net, args.probe_cap, args.probe_seconds) if net else None
                    Clear_TDD()
                    if peak is None:
                        continue
                    dest = pdir / c["path"].name
                    c["path"].rename(dest)
                    chosen.append(str(dest))
                    log.write(json.dumps({**{k2: v for k2, v in c.items() if k2 != "path"},
                                          "file": str(dest), "phase": phase,
                                          "target_nodes": t, "counting_peak": peak,
                                          "angles": "k*pi/4",
                                          "c_prime": "transpile level 3 to rz,sx,x,cx",
                                          "corpus_seed": args.seed}) + "\n")
                    break
            Path(f"{args.lists}_p{phase}.txt").write_text("\n".join(chosen) + "\n")
            print(f"phase {phase}: {len(chosen)} random circuits", flush=True)
    for leftover in (out / "candidates").glob("*.qasm"):
        leftover.unlink()
    (out / "candidates").rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
