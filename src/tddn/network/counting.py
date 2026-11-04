"""Counting-heuristic prefix for networks larger than the policy's bounds.

Contracts graph edges in FIFO order until the graph fits ``(max_nodes,
max_edges)``; the policy (or another planner) then contracts the rest.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from copy import deepcopy

from tddn.backend.TDD import cont
from tddn.network.graph import merge_nodes
from tddn.network.tdd_table import estimate_tdd_contraction_flops


def counting_prefix(graph, tdds, max_nodes, max_edges):
    """Reduce ``graph`` with FIFO contractions until it fits the bounds.

    Returns ``(graph, tdds, time_s, flops, peak_tdd_nodes, last_label_idx)``.
    New TDDs are labelled ``tdd_<k>`` with ``k`` continuing after the largest
    existing label; the caller continues numbering at ``last_label_idx + 1``.
    """

    new_g = graph.copy()
    reduced_tdds = deepcopy(tdds)

    existing_indices = []
    for key in reduced_tdds:
        if key.startswith("tdd_"):
            try:
                existing_indices.append(int(key.split("_")[1]))
            except ValueError:
                pass
    last_added = max(existing_indices) if existing_indices else len(reduced_tdds) - 1

    fifo_queue = deque(new_g.edges())
    time_s = 0.0
    flops_total = 0.0
    max_tdd_nodes = 0

    while new_g.number_of_edges() > 0:
        if new_g.number_of_nodes() <= max_nodes and new_g.number_of_edges() <= max_edges:
            break
        if not fifo_queue:
            break

        u, v = fifo_queue.popleft()
        if u not in new_g.nodes() or v not in new_g.nodes():
            continue
        if not new_g.has_edge(u, v):
            continue

        start_t = time.perf_counter()
        flops_total += estimate_tdd_contraction_flops(reduced_tdds[u][0], reduced_tdds[v][0])
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                new_tdd = cont(reduced_tdds[u][0], reduced_tdds[v][0])
        except Exception as exc:
            print(f"  counting prefix: contraction ({u}, {v}) failed: {exc}")
            continue

        last_added += 1
        label = f"tdd_{last_added}"
        max_tdd_nodes = max(max_tdd_nodes, new_tdd.node_number())

        tc_u = reduced_tdds[u][1] if len(reduced_tdds[u]) > 1 else 1
        tc_v = reduced_tdds[v][1] if len(reduced_tdds[v]) > 1 else 1
        gates_u = reduced_tdds[u][2] if len(reduced_tdds[u]) > 2 else []
        gates_v = reduced_tdds[v][2] if len(reduced_tdds[v]) > 2 else []
        reduced_tdds[label] = [new_tdd, tc_u + tc_v, gates_u + gates_v]

        stale = {tuple(sorted(e)) for e in new_g.edges() if v in e or u in e}
        fifo_queue = deque(e for e in fifo_queue if tuple(sorted(e)) not in stale)
        new_g = merge_nodes(new_g, u, v, label)
        fifo_queue.extend(e for e in new_g.edges() if label in e)

        del reduced_tdds[u]
        del reduced_tdds[v]
        time_s += time.perf_counter() - start_t

    return new_g, reduced_tdds, time_s, flops_total, max_tdd_nodes, last_added
