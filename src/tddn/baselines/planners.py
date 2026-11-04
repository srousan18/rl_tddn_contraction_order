"""Baseline contraction planners: cotengra optimizers and the Counting heuristic.

- ``cotengra_path``: a contraction path from ``cotengra.HyperOptimizer`` with
  one method (KaHyPar, Greedy, Betweenness or Random Greedy). The search is
  randomized and cannot be seeded, so sweeps repeat each method and report
  per-instance medians.
- ``counting_path``: the Counting heuristic, a deterministic FIFO edge queue
  over the contraction graph. Pop the oldest edge, contract it, drop queued
  edges touching the merged pair, and append the merged node's edges;
  disconnected components left at the end are joined pairwise.

The network's open indices are all 2n legs of the miter (n inputs ``x`` and
n outputs ``y``).
"""

from __future__ import annotations

from collections import deque

from tddn.backend.TDD_Q import TNtoCotInput
from tddn.network.graph import merge_nodes

COTENGRA_METHODS = {
    "kahypar": "kahypar",
    "greedy": "greedy",
    "betweenness": "betweenness",
    "random_greedy": "random-greedy",
}


def cot_input(tn, num_qubits: int):
    """``(tensor_list, open_indices, size_dict)`` for cotengra."""

    tensor_list, open_indices, size_dict, _, _ = TNtoCotInput(tn, num_qubits)
    return tensor_list, open_indices, size_dict


def cotengra_path(net, method: str, max_time: int = 10) -> tuple:
    """Contraction path for ``net`` found by one cotengra method."""

    import cotengra as ctg

    optimizer = ctg.HyperOptimizer(methods=[COTENGRA_METHODS[method]], progbar=False,
                                   parallel=False, max_time=max_time)
    tensor_list, open_indices, size_dict = cot_input(net.tn, net.num_qubits)
    tree = optimizer.search(tensor_list, open_indices, size_dict)
    return tuple(tree.get_path())


def counting_path(net) -> list:
    """Counting contraction order as ``(u, v)`` label pairs."""

    graph = net.graph.copy()
    last = len(net.tn.tensors) - 1
    fifo = deque(graph.edges())
    order = []

    while graph.number_of_edges() > 0:
        u, v = fifo.popleft()
        order.append((u, v))
        last += 1
        label = f"tdd_{last}"
        stale = {tuple(sorted(e)) for e in graph.edges() if u in e or v in e}
        fifo = deque(e for e in fifo if tuple(sorted(e)) not in stale)
        graph = merge_nodes(graph, u, v, label)
        fifo.extend(e for e in graph.edges() if label in e)

    leftovers = list(graph.nodes)
    while len(leftovers) > 1:
        a = leftovers.pop()
        b = leftovers.pop()
        order.append((b, a))
        last += 1
        leftovers.append(f"tdd_{last}")
    return order


def path_from_contraction_order(initial_labels, contraction_order) -> tuple:
    """Convert ``(u, v)`` label pairs into a path of ``(i, j)`` list positions.

    The label list starts as ``initial_labels``; each step removes both
    operands and appends ``tdd_<k>``, with ``k`` counting up from
    ``len(initial_labels)``.
    """

    labels = list(initial_labels)
    counter = len(labels)
    path = []
    for u, v in contraction_order:
        path.append((labels.index(u), labels.index(v)))
        labels.remove(u)
        labels.remove(v)
        labels.append(f"tdd_{counter}")
        counter += 1
    return tuple(path)
