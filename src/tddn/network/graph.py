"""Contraction graph of a tensor network.

One node per tensor (labelled ``tdd_<i>``), one edge per pair of tensors that
share at least one index. The edge attribute ``shared_indices`` lists the
shared ``(key, idx)`` index pairs.
"""

from __future__ import annotations

import networkx as nx


def build_networkx_graph(tn) -> nx.Graph:
    """Contraction graph of ``tn`` (a ``tddn.backend.TN.TensorNetwork``)."""

    G = nx.Graph()
    index_to_tensors = {}

    for i in range(len(tn.tensors)):
        tensor_node = f"tdd_{i}"
        G.add_node(tensor_node, qubits=tn.tensors[i].qubits)
        for index_obj in tn.tensors[i].index_set:
            index_tuple = (index_obj.key, index_obj.idx)
            if index_tuple not in index_to_tensors:
                index_to_tensors[index_tuple] = set()
            index_to_tensors[index_tuple].add(tensor_node)

    for index, tensors in index_to_tensors.items():
        tensor_list = list(tensors)
        if len(tensor_list) > 1:
            for i in range(len(tensor_list)):
                for j in range(i + 1, len(tensor_list)):
                    if G.has_edge(tensor_list[i], tensor_list[j]):
                        G[tensor_list[i]][tensor_list[j]]["shared_indices"].append(index)
                    else:
                        G.add_edge(tensor_list[i], tensor_list[j], shared_indices=[index])
    return G


def merge_nodes(G: nx.Graph, u, v, new_node) -> nx.Graph:
    """Replace nodes ``u`` and ``v`` by ``new_node`` (in place).

    ``new_node`` is connected to every neighbour of ``u`` or ``v``; the shared
    indices of parallel edges are united.
    """

    if u not in G or v not in G:
        raise ValueError("One or both nodes to merge are not in the graph.")

    G.add_node(new_node)
    new_edges = {}
    for neighbor, edge_data in G[u].items():
        if neighbor not in (u, v):
            if neighbor not in new_edges:
                new_edges[neighbor] = set()
            new_edges[neighbor].update(edge_data.get("shared_indices", []))
    for neighbor, edge_data in G[v].items():
        if neighbor not in (u, v):
            if neighbor not in new_edges:
                new_edges[neighbor] = set()
            new_edges[neighbor].update(edge_data.get("shared_indices", []))

    for neighbor, shared_indices in new_edges.items():
        G.add_edge(new_node, neighbor, shared_indices=list(shared_indices))

    G.remove_node(u)
    G.remove_node(v)
    return G
