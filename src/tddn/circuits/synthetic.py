"""Synthetic circuit families for policy pre-training.

Seeded generators of structural analogues of common circuit classes. None of
them uses a benchmark generator, so no benchmark circuit is seen during
pre-training:

    qaoalike        H layer, then alternating RZZ (chain + random chords) and RX layers
    qnnlike         RY rotation layers with CX entangling ladders
    graphstatelike  H layer and CZ on a random graph
    qftlike         cascading controlled-phase structure with dyadic angles
    randomgrid      random one- and two-qubit gates
    wstatelike      CRY + CX amplitude cascade along a random qubit order
    bvlike          CX fan-in to one ancilla (excluded by default)

All rotation angles are nonzero multiples of pi/16. Each instance draws an
entanglement level (low / mid / high) that scales depth and connectivity.
C' is C compiled at optimization level 3 to {rz, sx, x, cx}; every pair is
checked with MQT QCEC before it is kept (``tddn.circuits.pairs``). Instances
whose graph exceeds ``--bounds`` or whose contraction is too expensive are
regenerated at a smaller width.

Excluded families still draw their instance seeds, so the remaining
instances do not depend on the exclusion list.

The corpus in ``data/circuits/synthetic`` was generated with:

    python -m tddn.circuits.synthetic --root data/circuits/synthetic \\
        --total 1000 --seed 5000 --max-qubits 26
"""

from __future__ import annotations

import argparse
import signal
import string
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile

from tddn.backend.TDD import Clear_TDD, cont
from tddn.circuits.build import build_network_from_pair
from tddn.circuits.pairs import PairVerificationError, save_verified_pair
from tddn.network.tdd_table import tdd_table

BASIS = ["rz", "sx", "x", "cx"]

# Share of the corpus per family (percent).
WEIGHTS = {"qaoalike": 18, "qnnlike": 17, "graphstatelike": 14, "bvlike": 8,
           "qftlike": 5, "randomgrid": 20, "wstatelike": 18}

# Largest qubit count per family.
SIZE_CAPS = {"qaoalike": 16, "qnnlike": 18, "graphstatelike": 26, "bvlike": 26,
             "qftlike": 16, "randomgrid": 14, "wstatelike": 20}

# Contraction-probe node caps that differ from --probe-node-cap.
PROBE_CAPS = {"wstatelike": 200_000}

LEVELS = ("low", "mid", "high")
LEVEL_PROBS = (0.35, 0.40, 0.25)

BOUNDS_RETRIES = 3
DOWNSIZE_STEP = 3


class ProbeTimeout(BaseException):
    """Contraction probe exceeded its time budget."""


def _angle(rng) -> float:
    """A nonzero multiple of pi/16."""

    return float(int(rng.integers(1, 32)) * np.pi / 16)


def _draw_level(rng) -> str:
    return str(rng.choice(LEVELS, p=LEVEL_PROBS))


def _chain_plus_chords(n: int, rng, max_chords: int) -> list[tuple[int, int]]:
    """A random spanning chain plus up to ``max_chords`` extra edges."""

    perm = list(rng.permutation(n))
    edges = [(perm[i], perm[i + 1]) for i in range(n - 1)]
    for _ in range(int(rng.integers(0, max_chords + 1))):
        a, b = rng.choice(n, size=2, replace=False)
        if (a, b) not in edges and (b, a) not in edges:
            edges.append((int(a), int(b)))
    return edges


def qaoalike(n: int, rng, level: str) -> QuantumCircuit:
    layers = {"low": 1, "mid": int(rng.integers(1, 3)), "high": int(rng.integers(2, 4))}[level]
    max_chords = {"low": 0, "mid": max(1, n // 4), "high": max(2, n // 2)}[level]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(layers):
        for a, b in _chain_plus_chords(n, rng, max_chords):
            qc.rzz(_angle(rng), a, b)
        for q in range(n):
            qc.rx(_angle(rng), q)
    return qc


def qnnlike(n: int, rng, level: str) -> QuantumCircuit:
    layers = {"low": 1, "mid": int(rng.integers(1, 3)), "high": int(rng.integers(2, 4))}[level]
    cx_keep = {"low": 0.5, "mid": 0.8, "high": 1.0}[level]
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(_angle(rng), q)
        offset = int(rng.integers(0, 2))
        for start in (offset, 1 - offset):
            for q in range(start, n - 1, 2):
                if rng.random() < cx_keep:
                    qc.cx(q, q + 1)
        if level == "high" and n > 2 and rng.random() < 0.5:
            qc.cx(n - 1, 0)
    return qc


def graphstatelike(n: int, rng, level: str) -> QuantumCircuit:
    max_chords = {"low": 0, "mid": max(1, n // 4), "high": max(2, n // 2)}[level]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for a, b in _chain_plus_chords(n, rng, max_chords):
        qc.cz(a, b)
    return qc


def bvlike(n: int, rng, level: str) -> QuantumCircuit:
    max_ones = {"low": max(1, (n - 1) // 3), "mid": max(1, 2 * (n - 1) // 3),
                "high": n - 1}[level]
    qc = QuantumCircuit(n)
    ancilla = n - 1
    qc.x(ancilla)
    qc.h(range(n))
    ones = rng.choice(n - 1, size=int(rng.integers(1, max_ones + 1)), replace=False)
    for q in ones:
        qc.cx(int(q), ancilla)
    qc.h(range(n - 1))
    return qc


def qftlike(n: int, rng, level: str) -> QuantumCircuit:
    keep = {"low": 0.5, "mid": 0.7, "high": 0.85}[level]
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            if rng.random() < keep:
                exponent = k - j + int(rng.integers(0, 2))
                qc.cp(np.pi / 2**exponent, k, j)
    return qc


_ONE_QUBIT = ("h", "x", "sx", "rz", "ry", "rx")


def randomgrid(n: int, rng, level: str) -> QuantumCircuit:
    p2 = {"low": 0.15, "mid": 0.3, "high": 0.45}[level]
    depth = int(rng.integers(5, 21))
    qc = QuantumCircuit(n)
    for _ in range(depth):
        for q in range(n):
            if rng.random() < 0.7:
                gate = _ONE_QUBIT[int(rng.integers(len(_ONE_QUBIT)))]
                if gate in ("rz", "ry", "rx"):
                    getattr(qc, gate)(_angle(rng), q)
                else:
                    getattr(qc, gate)(q)
        order = list(rng.permutation(n))
        for i in range(0, n - 1, 2):
            if rng.random() < p2:
                qc.cx(order[i], order[i + 1])
    return qc


def wstatelike(n: int, rng, level: str) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    order = list(rng.permutation(n))
    qc.x(order[0])
    keep = {"low": 0.7, "mid": 0.9, "high": 1.0}[level]
    for i in range(n - 1):
        if rng.random() < keep:
            qc.cry(_angle(rng), order[i], order[i + 1])
            qc.cx(order[i + 1], order[i])
    if level == "high":
        for i in range(n - 1, 0, -1):
            if rng.random() < 0.5:
                qc.cry(_angle(rng), order[i], order[i - 1])
                qc.cx(order[i - 1], order[i])
    return qc


GENERATORS = {
    "qaoalike": qaoalike,
    "qnnlike": qnnlike,
    "graphstatelike": graphstatelike,
    "bvlike": bvlike,
    "qftlike": qftlike,
    "randomgrid": randomgrid,
    "wstatelike": wstatelike,
}


def contraction_probe(net, node_cap: int = 50_000, seconds: int = 20) -> tuple[bool, str]:
    """Contract the network in tensor order under a node cap and a time budget.

    Requires the network's TDD session to be active; the caller clears it.
    """

    def _raise(signum, frame):
        raise ProbeTimeout

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        entries = list(tdd_table(net.tn).values())
        acc = entries[0][0]
        for entry in entries[1:]:
            acc = cont(acc, entry[0])
            if acc.node_number() > node_cap:
                return False, f"probe peak > {node_cap} nodes"
        return True, "ok"
    except ProbeTimeout:
        return False, f"probe time > {seconds}s"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _free_variant(root: Path, family: str, n: int) -> str:
    """First unused variant letter for ``<family><letter>_<n>q``."""

    for letter in string.ascii_uppercase:
        if not (root / "alg" / f"{family}{letter}_{n}q.qasm").exists():
            return letter
    raise RuntimeError(f"variant letters exhausted for {family}_{n}q")


def build_corpus(root: str | Path, total: int = 1000, min_qubits: int = 5,
                 max_qubits: int = 26, seed: int = 5000,
                 bounds: tuple[int, int] = (500, 1000), probe_node_cap: int = 50_000,
                 probe_seconds: int = 20, timeout: int = 30,
                 exclude: tuple[str, ...] = ("bvlike",)) -> dict:
    """Generate, size-check, probe, save and QCEC-verify a corpus."""

    root = Path(root)
    rng = np.random.default_rng(seed)
    kept, rejected_qcec, rejected_cost, failed = 0, 0, 0, 0
    level_counts = dict.fromkeys(LEVELS, 0)

    for family, weight in WEIGHTS.items():
        count = round(total * weight / 100)
        generator = GENERATORS[family]
        sizes = list(range(min_qubits, min(max_qubits, SIZE_CAPS[family]) + 1))
        for idx in range(count):
            n = sizes[idx % len(sizes)]
            instance_seed = int(rng.integers(2**31))
            if family in exclude:
                continue
            instance_rng = np.random.default_rng(instance_seed)
            level = _draw_level(instance_rng)
            try:
                reason = ""
                for attempt in range(BOUNDS_RETRIES):
                    qc = generator(n, instance_rng, level)
                    qc_prime = transpile(qc, basis_gates=BASIS, optimization_level=3,
                                         seed_transpiler=instance_seed + attempt)
                    net = build_network_from_pair(qc, qc_prime)
                    nodes = net.graph.number_of_nodes()
                    edges = net.graph.number_of_edges()
                    if nodes <= bounds[0] and edges <= bounds[1]:
                        ok, reason = contraction_probe(
                            net, node_cap=PROBE_CAPS.get(family, probe_node_cap),
                            seconds=probe_seconds)
                        Clear_TDD()
                        if ok:
                            break
                    else:
                        Clear_TDD()
                        reason = f"{nodes}n/{edges}e > bounds"
                    n = max(min_qubits, n - DOWNSIZE_STEP)
                else:
                    rejected_cost += 1
                    print(f"rejected {family} {level} at {n}q ({reason})", flush=True)
                    continue

                name = f"{family}{_free_variant(root, family, n)}"
                save_verified_pair(qc, qc_prime, name, n, root=root, timeout=timeout,
                                   provenance={"level": level, "instance_seed": instance_seed,
                                               "corpus_seed": seed, "opt_level": 3,
                                               "basis": BASIS, "graph_shape": [nodes, edges]})
            except PairVerificationError as exc:
                rejected_qcec += 1
                print(f"rejected by QCEC: {exc}", flush=True)
                continue
            except Exception as exc:
                failed += 1
                print(f"failed {family}_{n}q: {exc}", flush=True)
                continue
            kept += 1
            level_counts[level] += 1
            print(f"ok {name}_{n}q {level} ({nodes}n/{edges}e)", flush=True)

    summary = {"kept": kept, "rejected_qcec": rejected_qcec, "rejected_cost": rejected_cost,
               "failed": failed, "levels": level_counts}
    print(f"\ncorpus: {summary}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/circuits/synthetic_new")
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--min-qubits", type=int, default=5)
    ap.add_argument("--max-qubits", type=int, default=26)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--bounds", default="500,1000", help="max graph nodes,edges")
    ap.add_argument("--probe-node-cap", type=int, default=50_000)
    ap.add_argument("--probe-seconds", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=30, help="QCEC seconds per pair")
    ap.add_argument("--exclude", default="bvlike", help="comma-separated families to skip")
    args = ap.parse_args(argv)

    summary = build_corpus(args.root, total=args.total, min_qubits=args.min_qubits,
                           max_qubits=args.max_qubits, seed=args.seed,
                           bounds=tuple(int(x) for x in args.bounds.split(",")),
                           probe_node_cap=args.probe_node_cap,
                           probe_seconds=args.probe_seconds, timeout=args.timeout,
                           exclude=tuple(x for x in args.exclude.split(",") if x))
    return 1 if summary["kept"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
