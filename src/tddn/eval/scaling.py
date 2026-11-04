"""Scaling beyond the training range: benchmarks at 24-50 qubits.

Every benchmark family at every size from 24 to 50 qubits, with RL policies
and the baselines (Counting and four cotengra optimizers):

  zeroshot  bv / dj / ghz / qnn: the graphs fit the policy bounds (87/151)
            and the policy contracts them directly;
  hybrid    wstate: the graphs exceed the bounds, so a Counting prefix
            reduces them first and the policy contracts the rest.

The prefix is applied only when a graph exceeds the bounds, so one variant
``counting_prefix+rl`` covers both sets. Every method counts the final TDD in
its peak and ends with an identity check of the final TDD. Each row has a hard
time limit (default 120 s); cotengra methods are repeated (randomized
search), RL and Counting are deterministic and run once. ``--ext-timeout``
re-runs RL rows that timed out with a larger limit into ``results_ext.csv``.

CLI:
    python -m tddn.eval.scaling --out results/two_stage/scaling --methods rl \\
        --model two_stage=models/two_stage.zip
    python -m tddn.eval.scaling --out results/baselines/scaling \\
        --methods counting,cotengra_kahypar,cotengra_greedy,cotengra_between,cotengra_rgreedy
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import platform
import signal
import time
from copy import deepcopy
from pathlib import Path

import cotengra as ctg

from tddn.agents.loading import load_policy
from tddn.backend.TDD import Clear_TDD, Ini_TDD, cont
from tddn.backend.TDD_Q import cir_2_tn_lbl, get_real_qubit_num, squeezeTN, squeezeTN_ultra
from tddn.backend.TN import TensorNetwork
from tddn.circuits.qasm import read_qasm_file
from tddn.config import BENCHMARK_DIR, M_MAX, N_MAX, PROJECT_ROOT
from tddn.envs.contraction_env import TensorContractionEnv
from tddn.network.counting import counting_prefix
from tddn.network.graph import build_networkx_graph, merge_nodes
from tddn.network.tdd_table import tdd_table
from tddn.verify.tdd_identity import check_final_tdd

TIMEOUT_S = 120

COTENGRA_METHODS = {
    "cotengra_kahypar": "kahypar",
    "cotengra_greedy": "greedy",
    "cotengra_between": "betweenness",
    "cotengra_rgreedy": "random-greedy",
}
METHODS = ["rl", "counting"] + list(COTENGRA_METHODS)
FAMILIES = ["bv", "dj", "ghz", "qnn", "wstate"]
SIZES = list(range(24, 51))
# cotengra's search is randomized and cannot be seeded: these methods are repeated.
STOCHASTIC = set(COTENGRA_METHODS)
DEFAULT_REPEATS = 3

FIELDNAMES = ["set", "family", "qubits", "method", "model", "variant", "repeat",
              "prefix_activated", "prefix_steps", "prefix_time_s", "prefix_peak_nodes",
              "graph_nodes_before", "graph_edges_before",
              "graph_nodes_after", "graph_edges_after",
              "peak_tdd_nodes", "total_time_s", "plan_time_s", "contract_time_s",
              "status", "identity_status", "file"]


class HardTimeout(BaseException):
    """A row exceeded its time limit."""


@contextlib.contextmanager
def hard_timeout(seconds: float):
    def _fire(signum, frame):
        raise HardTimeout(f"exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def build_pool(path):
    """Circuit -> (graph, TDD table, tensor network, qubit count) in a fresh TDD session."""

    qc = read_qasm_file(path)
    tn, all_indexes, depth = cir_2_tn_lbl(qc)
    n = get_real_qubit_num(qc)
    tensors = squeezeTN(tn.tensors, n, depth)
    tensors = squeezeTN_ultra(tensors, n, depth)
    tn = TensorNetwork(tensors, tn.tn_type, n)

    Clear_TDD()
    Ini_TDD(index_order=all_indexes)
    graph = build_networkx_graph(tn)
    tdds = tdd_table(tn)
    return graph, tdds, tn, n


def tdds_to_cotengra_input(graph, tdds):
    """Index structure of the current TDD pool in cotengra's input format.

    Indices are labelled by ``Index.key``: ``TDD.cont`` sums over keys, so the
    legs of a hyperedge share one label. Open indices are the keys that appear
    on exactly one tensor (the 2n legs x_q / y_q of a miter).
    """

    labels = list(graph.nodes())
    tensor_list = []
    counts: dict[str, int] = {}
    for lbl in labels:
        idx = tuple(dict.fromkeys(i.key for i in tdds[lbl][0].index_set))
        tensor_list.append(idx)
        for i in idx:
            counts[i] = counts.get(i, 0) + 1
    open_indices = tuple(i for i, c in counts.items() if c == 1)
    size_dict = {i: 2 for i in counts}
    return labels, tensor_list, open_indices, size_dict


# ---------------------------------------------------------------------------
# Finishers: plan + contract the current pool
# ---------------------------------------------------------------------------

def join_components(pool_tdds, peak):
    """Contract the remaining disjoint TDDs (disconnected components) into one."""

    contract_s = 0.0
    while len(pool_tdds) > 1:
        a = pool_tdds.pop()
        b = pool_tdds.pop()
        t0 = time.perf_counter()
        merged = cont(a, b)
        contract_s += time.perf_counter() - t0
        peak = max(peak, merged.node_number())
        pool_tdds.append(merged)
    return pool_tdds, peak, contract_s


def finish_counting(graph, tdds, next_label_idx):
    """Contract in FIFO edge order to completion (the Counting heuristic)."""

    from collections import deque

    g = graph.copy()
    pool = tdds
    queue = deque(g.edges())
    last = next_label_idx
    plan_s = contract_s = 0.0
    peak = 0

    while g.number_of_edges() > 0:
        t0 = time.perf_counter()
        if not queue:
            break
        u, v = queue.popleft()
        plan_s += time.perf_counter() - t0
        if u not in g.nodes() or v not in g.nodes() or not g.has_edge(u, v):
            continue

        t0 = time.perf_counter()
        new_tdd = cont(pool[u][0], pool[v][0])
        contract_s += time.perf_counter() - t0

        label = f"tdd_{last}"
        last += 1
        peak = max(peak, new_tdd.node_number())
        pool[label] = [new_tdd, pool[u][1] + pool[v][1], pool[u][2] + pool[v][2]]

        stale = {tuple(sorted(e)) for e in g.edges() if u in e or v in e}
        queue = deque(e for e in queue if tuple(sorted(e)) not in stale)
        g = merge_nodes(g, u, v, label)
        for e in g.edges():
            if label in e:
                queue.append(e)
        del pool[u], pool[v]

    remaining = [pool[l][0] for l in g.nodes()]
    remaining, peak, join_s = join_components(remaining, peak)
    contract_s += join_s
    return remaining, peak, plan_s, contract_s


def finish_cotengra(graph, tdds, method_name, next_label_idx, opt_max_time):
    """Plan with a cotengra optimizer over the pool, then execute the path."""

    labels, tensor_list, open_indices, size_dict = tdds_to_cotengra_input(graph, tdds)

    t0 = time.perf_counter()
    optimizer = ctg.HyperOptimizer(
        methods=[COTENGRA_METHODS[method_name]],
        parallel=False, progbar=False, max_time=opt_max_time,
    )
    tree = optimizer.search(tensor_list, open_indices, size_dict)
    path = tree.get_path()
    plan_s = time.perf_counter() - t0

    # Execute the path (opt_einsum order: remove the higher position first).
    pool = [tdds[l][0] for l in labels]
    contract_s = 0.0
    peak = 0
    for pair in path:
        i, j = sorted(pair)
        a, b = pool[i], pool[j]
        t0 = time.perf_counter()
        merged = cont(a, b)
        contract_s += time.perf_counter() - t0
        peak = max(peak, merged.node_number())
        pool.pop(j)
        pool.pop(i)
        pool.append(merged)

    pool, peak, join_s = join_components(pool, peak)
    return pool, peak, plan_s + 0.0, contract_s + join_s


def finish_rl(graph, tdds, tn, model_state, next_label_idx):
    """Contract the (possibly prefix-reduced) pool with the policy.

    The environment's per-contraction alarm is off (``cont_timeout_s=0``);
    the row's hard time limit applies.
    """

    env = TensorContractionEnv(graph.copy(), tn, max_nodes=N_MAX, max_edges=M_MAX,
                               cont_timeout_s=0)
    env.set_test_mode(True)
    # Start from the given pool instead of the freshly built one.
    env.initial_graph = graph
    env.initial_tdds = deepcopy(tdds)
    env.current_graph = graph.copy()
    env.tdds = deepcopy(tdds)
    env.next_label_idx = next_label_idx

    if model_state["model"] is None:
        model_state["model"] = load_policy(model_state["path"], env=env)
    else:
        model_state["model"].set_env(env)
    model = model_state["model"]

    obs, _ = env.reset()
    # reset() numbers new labels from len(tdds); continue after the prefix's labels.
    env.next_label_idx = next_label_idx

    plan_s = contract_s = 0.0
    done = False
    while not done:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        plan_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        obs, _, done, _, _ = env.step(action)
        contract_s += time.perf_counter() - t0

    info = env.env_info()
    return info, info["max_tdd_nodes"], plan_s, contract_s


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------

def run_config(path, method, use_prefix, model_state, timeout_s):
    row = {
        "method": method,
        "variant": "counting_prefix+" + method if use_prefix else method,
        "prefix_activated": False, "prefix_steps": 0,
        "prefix_time_s": 0.0, "prefix_peak_nodes": 0,
        "peak_tdd_nodes": -1, "total_time_s": 0.0,
        "plan_time_s": 0.0, "contract_time_s": 0.0,
        "status": "ok", "identity_status": "",
    }
    try:
        with hard_timeout(timeout_s):
            graph, tdds, tn, n = build_pool(path)
            row["graph_nodes_before"] = graph.number_of_nodes()
            row["graph_edges_before"] = graph.number_of_edges()

            next_idx = len(tdds)
            prefix_peak = 0
            if use_prefix and (graph.number_of_nodes() > N_MAX
                               or graph.number_of_edges() > M_MAX):
                before = graph.number_of_nodes()
                graph, tdds, h_time, _, prefix_peak, last_idx = counting_prefix(
                    graph, tdds, N_MAX, M_MAX)
                next_idx = last_idx + 1
                row.update(prefix_activated=True,
                           prefix_steps=before - graph.number_of_nodes(),
                           prefix_time_s=round(h_time, 6),
                           prefix_peak_nodes=prefix_peak)

            row["graph_nodes_after"] = graph.number_of_nodes()
            row["graph_edges_after"] = graph.number_of_edges()

            if method == "counting":
                pool, peak, plan_s, contract_s = finish_counting(graph, tdds, next_idx)
                final = {"fully_contracted": len(pool) == 1,
                         "tdds": {"final": [pool[0], 1, []]}} if pool else {}
            elif method == "rl":
                info, peak, plan_s, contract_s = finish_rl(graph, tdds, tn, model_state, next_idx)
                final = info
            else:
                pool, peak, plan_s, contract_s = finish_cotengra(
                    graph, tdds, method, next_idx,
                    int(timeout_s * 0.5) if timeout_s > 10 else int(timeout_s * 0.8))
                final = {"fully_contracted": len(pool) == 1,
                         "tdds": {"final": [pool[0], 1, []]}} if pool else {}

            row["peak_tdd_nodes"] = max(peak, prefix_peak)
            row["plan_time_s"] = round(plan_s, 6)
            row["contract_time_s"] = round(contract_s + row["prefix_time_s"], 6)
            row["total_time_s"] = round(plan_s + contract_s + row["prefix_time_s"], 6)
            row["identity_status"] = check_final_tdd(final, num_qubits=n)["status"]
    except HardTimeout:
        row["status"] = "timeout"
    except Exception as exc:
        row["status"] = f"error:{type(exc).__name__}"
    finally:
        Clear_TDD()
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def instance_list(families, sizes):
    """(set, family, qubits, path): zeroshot = within bounds, hybrid = wstate."""

    out = []
    for q in sizes:
        for fam in families:
            p = BENCHMARK_DIR / f"combined_{fam}_{q}q.qasm"
            if p.exists():
                out.append(("hybrid" if fam == "wstate" else "zeroshot", fam, q, str(p)))
    return out


def jobs(instances, methods, models, repeats):
    """(set, family, q, path, method, model_label, use_prefix, repeat)."""

    out = []
    for setname, fam, q, path in instances:
        for method in methods:
            if method == "rl":
                for label in models:
                    out.append((setname, fam, q, path, "rl", label, True, 1))
            else:
                n = repeats if method in STOCHASTIC else 1
                out += [(setname, fam, q, path, method, "", False, r) for r in range(1, n + 1)]
    return out


def _key(r):
    return (r["file"], r["method"], r["model"], r["variant"], str(r["repeat"]))


def run_jobs(todo, out_csv, models, timeout_s, model_states):
    fresh = not out_csv.exists()
    with open(out_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if fresh:
            writer.writeheader()
            fh.flush()
        t_start = time.time()
        for setname, fam, q, path, method, label, use_prefix, rep in todo:
            state = model_states.setdefault(label, {"model": None, "path": models.get(label)})
            row = run_config(resolve_path(path), method, use_prefix, state, timeout_s)
            row.update(set=setname, family=fam, qubits=q, file=display_path(path),
                       repeat=rep, model=label)
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
            fh.flush()
            tag = "ok " if row["status"] == "ok" else "!! "
            name = f"{row['variant']}[{label}]" if label else row["variant"]
            print(f"[{tag}] {setname:8s} {fam:6s} n={q:2d} r{rep} {name:30s} "
                  f"peak={row['peak_tdd_nodes']:>9} t={row['total_time_s']:8.3f}s "
                  f"{row['status']:12s} {row['identity_status']} "
                  f"(elapsed {time.time() - t_start:6.0f}s)", flush=True)


def display_path(path) -> str:
    """Path relative to the project root when inside it (as written to the CSV)."""
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def resolve_path(path) -> str:
    path = Path(path)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def provenance(models, args):
    import numpy, qiskit
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "qiskit": qiskit.__version__, "cotengra": getattr(ctg, "__version__", "?"),
            "models": models, "N_max": N_MAX, "M_max": M_MAX,
            "timeout_s": args.timeout, "ext_timeout_s": args.ext_timeout,
            "repeats": args.repeats, "methods": args.methods,
            "families": args.families, "sizes": args.sizes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH",
                    help="RL checkpoint by explicit path (repeatable)")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--sizes", default=None, help="comma list (default 24..50)")
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S)
    ap.add_argument("--ext-timeout", type=int, default=0,
                    help="re-run timed-out RL rows with this budget (results_ext.csv)")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                    help="repeats of the cotengra methods")
    args = ap.parse_args(argv)
    args.methods = args.methods.split(",")
    args.families = args.families.split(",")
    args.sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else SIZES

    models = {}
    for spec in args.model:
        label, _, path = spec.partition("=")
        if not Path(path).is_file():
            raise SystemExit(f"checkpoint not found: {path}")
        models[label] = path
    if "rl" in args.methods and not models:
        raise SystemExit("method rl needs at least one --model LABEL=PATH")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / "results.csv"
    done = {_key(r) for r in csv.DictReader(open(out_csv))} if out_csv.exists() else set()

    instances = instance_list(args.families, args.sizes)
    todo = [j for j in jobs(instances, args.methods, models, args.repeats)
            if (display_path(j[3]), j[4], j[5], "counting_prefix+rl" if j[6] else j[4],
                str(j[7])) not in done]
    print(f"{len(instances)} instance(s); {len(todo)} run(s) to do; "
          f"{len(done)} already done", flush=True)
    with open(out / "provenance.json", "w") as pf:
        json.dump(provenance(models, args), pf, indent=2)

    model_states = {}
    run_jobs(todo, out_csv, models, args.timeout, model_states)

    if args.ext_timeout and models:
        ext_csv = out / "results_ext.csv"
        ext_done = ({_key(r) for r in csv.DictReader(open(ext_csv))}
                    if ext_csv.exists() else set())
        timed_out = [r for r in csv.DictReader(open(out_csv))
                     if r["method"] == "rl" and r["status"] == "timeout"
                     and r["model"] in models and _key(r) not in ext_done]
        ext = [(r["set"], r["family"], int(r["qubits"]), r["file"], "rl", r["model"],
                True, 1) for r in timed_out]
        print(f"\nextended budget {args.ext_timeout}s: {len(ext)} RL row(s)", flush=True)
        run_jobs(ext, ext_csv, models, args.ext_timeout, model_states)
    print(f"\n  results {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
