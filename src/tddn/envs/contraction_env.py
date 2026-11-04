"""Contraction environment: one episode contracts one tensor network to a single TDD.

* **State**: the contraction graph (one node per tensor, one edge per pair of
  tensors sharing indices) and the TDD of every tensor.
* **Action**: an index into the current edge list; the two tensors at its
  ends are contracted with the TDD backend.
* **Reward**: each contraction pays ``-(10 / (N - 1)) * (log10 |TDD|)^beta``,
  where ``|TDD|`` is the node count of the new TDD and ``N`` the initial
  number of tensors; completing the episode adds +5. ``beta = 1.5``.
* **Observation**, padded to fixed ``(max_nodes, max_edges)`` bounds:

  ===============  ======================================================
  node (4)         tensor count, log10 TDD size, effective degree / N_MAX,
                   rank / 10
  edge (2)         shared indices / 5, (deg_u + deg_v) / (2 N_MAX)
  global (5)       nodes / N_MAX, edges / M_MAX, mean and max log10 size,
                   steps / 50
  ===============  ======================================================

Safeguards: a single contraction runs under an alarm of ``cont_timeout_s``
seconds and the episode ends if it is exceeded (the TDD tables are
process-global and are not reliable after an interrupted contraction). A
contraction that produces more than ``NODE_GUARD`` TDD nodes, or raises, is
refused and its edge is masked for the rest of the episode.
"""

import math
import signal
import time
import warnings
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

from tddn.backend.TDD import cont
from tddn.network.graph import merge_nodes
from tddn.network.tdd_table import tdd_table

# Largest TDD (node count) a single contraction may produce.
NODE_GUARD = 100_000


class ContractionTimeout(BaseException):
    """A single contraction exceeded ``cont_timeout_s``."""


class TensorContractionEnv(gym.Env):
    def __init__(
        self,
        graph,
        tn,
        max_nodes=40,
        max_edges=100,
        reward_exponent=1.5,
        bond_norm=5.0,
        rank_norm_factor=10.0,
        cont_timeout_s=10,
    ):
        super().__init__()
        self.cont_timeout_s = cont_timeout_s

        self.initial_graph = graph
        self.current_graph = graph.copy()

        # Label -> [tdd, tensor_count, gate_names]
        self.tdds = tdd_table(tn)
        for entry in self.tdds.values():
            entry[2] = self._normalize_gate_list(entry[2])
        # reset() deep-copies this table for every episode.
        self.initial_tdds = self.tdds
        self._tdd_size_cache = {}
        self.next_label_idx = len(self.tdds)
        self.last_label = None

        self.max_tdd_nodes = 0
        self.total_contraction_time = 0.0
        self.reward = 0.0
        self.contraction_order = []

        self.MAX_NODES = max_nodes
        self.MAX_EDGES = max_edges
        self.test_mode = False
        self.banned_edges = set()
        self.reward_exponent = reward_exponent
        self.bond_norm = float(bond_norm)
        self.rank_norm_factor = float(rank_norm_factor)
        self._deg_cache = None

        self.NODE_FEATURES = 4
        self.EDGE_FEATURES = 2
        self.GLOBAL_FEATURES = 5

        self.action_space = Discrete(self.MAX_EDGES)
        self.observation_space = Dict({
            "node_features": Box(-np.inf, np.inf, (self.MAX_NODES, self.NODE_FEATURES), dtype=np.float32),
            "edge_index": Box(-1, self.MAX_NODES, (2, self.MAX_EDGES), dtype=np.int64),
            "edge_features": Box(-np.inf, np.inf, (self.MAX_EDGES, self.EDGE_FEATURES), dtype=np.float32),
            "global_features": Box(-np.inf, np.inf, (1, self.GLOBAL_FEATURES), dtype=np.float32),
            "num_nodes": Box(0, self.MAX_NODES, (1,), dtype=np.int64),
            "num_edges": Box(0, self.MAX_EDGES, (1,), dtype=np.int64),
            "feasible_actions": Box(0, 1, (self.MAX_EDGES,), dtype=np.float32),
        })
        self._steps_in_ep = 0

    # ---------- Helpers ----------
    def _normalize_gate_list(self, gate_field):
        def flatten_any(obj):
            if isinstance(obj, str):
                return obj.split("-")
            if isinstance(obj, list):
                out = []
                for item in obj:
                    out.extend(flatten_any(item))
                return out
            return []
        return flatten_any(gate_field)

    def _timed_cont(self, tdd_a, tdd_b):
        """``cont()`` under an alarm; raises ``ContractionTimeout`` on overrun.

        Without an alarm (``cont_timeout_s`` of 0, or outside the main thread)
        the contraction runs unguarded.
        """

        if not self.cont_timeout_s:
            return cont(tdd_a, tdd_b)

        def _raise(signum, frame):
            raise ContractionTimeout

        try:
            previous = signal.signal(signal.SIGALRM, _raise)
        except ValueError:
            return cont(tdd_a, tdd_b)
        signal.alarm(int(self.cont_timeout_s))
        try:
            return cont(tdd_a, tdd_b)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    def _tdd_size(self, label):
        size = self._tdd_size_cache.get(label)
        if size is None:
            size = self.tdds[label][0].node_number()
            self._tdd_size_cache[label] = size
        return size

    def _rebuild_graph_cache(self):
        """Effective degree of every node, computed in one pass over the edges."""

        deg = {}
        for u, v, data in self.current_graph.edges(data=True):
            si = data.get("shared_indices") if data else None
            w = len(si) if si is not None else 1
            deg[u] = deg.get(u, 0) + w
            deg[v] = deg.get(v, 0) + w
        for n in self.current_graph.nodes:
            deg.setdefault(n, 0)
        self._deg_cache = deg

    def _get_tensor_rank(self, node):
        """Number of neighbouring tensors."""
        return len(self.current_graph[node])

    def _effective_degree(self, node):
        """Total number of indices shared with neighbouring tensors."""
        if self._deg_cache is not None:
            return self._deg_cache.get(node, 0)
        total = 0
        for nbr in self.current_graph.neighbors(node):
            data = self.current_graph.get_edge_data(node, nbr)
            if data and "shared_indices" in data:
                total += len(data["shared_indices"])
            else:
                total += 1
        return total

    # ---------- Features ----------
    def _build_node_features_single(self, node):
        tensor_count = self.tdds[node][1]
        internal_size = self._tdd_size(node)
        internal_log10 = math.log10(internal_size) if internal_size > 0 else 0.0
        deg_norm = self._effective_degree(node) / float(self.MAX_NODES)
        rank_norm = self._get_tensor_rank(node) / self.rank_norm_factor
        return np.array([float(tensor_count), float(internal_log10),
                         float(deg_norm), float(rank_norm)], dtype=np.float32)

    def _build_edge_features_single(self, u, v):
        data = self.current_graph.get_edge_data(u, v)
        if data and "shared_indices" in data:
            shared_count = len(data["shared_indices"])
        else:
            shared_count = 1
        shared_norm = shared_count / self.bond_norm
        deg_sum_norm = ((self._effective_degree(u) + self._effective_degree(v))
                        / (2.0 * self.MAX_NODES))
        return np.array([shared_norm, deg_sum_norm], dtype=np.float32)

    def _build_all_nodes_features(self):
        out = np.zeros((self.MAX_NODES, self.NODE_FEATURES), dtype=np.float32)
        for i, node in enumerate(self.current_graph.nodes):
            if i >= self.MAX_NODES:
                break
            out[i] = self._build_node_features_single(node)
        return out

    def _build_all_edges_features(self):
        out = np.zeros((self.MAX_EDGES, self.EDGE_FEATURES), dtype=np.float32)
        for i, (u, v) in enumerate(self.current_graph.edges):
            if i >= self.MAX_EDGES:
                break
            out[i] = self._build_edge_features_single(u, v)
        return out

    def _build_edge_index(self):
        edge_index = np.full((2, self.MAX_EDGES), -1, dtype=np.int64)
        node_to_idx = {n: idx for idx, n in enumerate(self.current_graph.nodes)}
        for i, (u, v) in enumerate(self.current_graph.edges):
            if i >= self.MAX_EDGES:
                break
            edge_index[0, i] = node_to_idx[u]
            edge_index[1, i] = node_to_idx[v]
        return edge_index

    def _build_global_features(self, node_features=None):
        num_nodes = len(self.current_graph.nodes)
        num_edges = len(self.current_graph.edges)
        if node_features is None:
            node_features = self._build_all_nodes_features()
        if num_nodes > 0:
            # Column 1 holds log10 of the TDD size.
            avg_size = float(np.mean(node_features[:num_nodes, 1]))
            max_size = float(np.max(node_features[:num_nodes, 1]))
        else:
            avg_size, max_size = 0.0, 0.0
        return np.array([
            float(num_nodes) / self.MAX_NODES,
            float(num_edges) / self.MAX_EDGES,
            avg_size,
            max_size,
            float(self._steps_in_ep) / 50.0,
        ], dtype=np.float32).reshape(1, -1)

    def _compute_action_mask(self):
        mask = np.zeros(self.MAX_EDGES, dtype=np.float32)
        for i, (u, v) in enumerate(self.current_graph.edges):
            if i >= self.MAX_EDGES:
                break
            if frozenset({u, v}) not in self.banned_edges:
                mask[i] = 1.0
        return mask

    def get_observation(self):
        self._rebuild_graph_cache()
        node_feats = self._build_all_nodes_features()
        edge_feats = self._build_all_edges_features()
        edge_idx = self._build_edge_index()
        global_vec = self._build_global_features(node_feats)

        # Graphs larger than the bounds: drop edges to nodes beyond MAX_NODES.
        if len(self.current_graph.nodes) > self.MAX_NODES:
            invalid = ~((edge_idx[0] < self.MAX_NODES) & (edge_idx[1] < self.MAX_NODES))
            if np.any(invalid):
                edge_idx[0, invalid] = 0
                edge_idx[1, invalid] = 0
                edge_feats[invalid] = 0

        return {
            "node_features": node_feats,
            "edge_index": edge_idx,
            "edge_features": edge_feats,
            "global_features": global_vec,
            "num_nodes": np.array([min(len(self.current_graph.nodes), self.MAX_NODES)], dtype=np.int64),
            "num_edges": np.array([len(self.current_graph.edges)], dtype=np.int64),
            "feasible_actions": self._compute_action_mask(),
        }

    # ---------- Episode ----------
    def _compute_reward(self, new_tdd):
        new_size = new_tdd.node_number()
        mem_log = math.log10(new_size) if new_size > 0 else 0.0
        return float(-1.0 * (mem_log ** self.reward_exponent) * self.reward_scale)

    def set_test_mode(self, mode: bool):
        """Evaluation mode: no rewards, no completion bonus, no per-step time limit."""
        self.test_mode = mode

    def env_info(self):
        num_nodes_remaining = len(self.current_graph.nodes)
        num_edges_remaining = len(self.current_graph.edges)
        return {
            "graph": self.current_graph,
            "contraction_order": self.contraction_order,
            "tdds": self.tdds,
            "num_nodes_remaining": num_nodes_remaining,
            "num_edges_remaining": num_edges_remaining,
            "fully_contracted": num_nodes_remaining == 1 and num_edges_remaining == 0,
            "max_tdd_nodes": self.max_tdd_nodes,
        }

    def reset(self, seed=None, options=None):
        del self.current_graph
        del self.tdds
        self.current_graph = self.initial_graph.copy()
        self.tdds = deepcopy(self.initial_tdds)
        self._tdd_size_cache.clear()

        # An episode has N - 1 contractions; scaling by 10 / (N - 1) keeps the
        # episode return independent of the network size.
        n_initial = len(self.current_graph.nodes)
        self.reward_scale = 10.0 / max(1, n_initial - 1)

        self.contraction_order = []
        self.total_contraction_time = 0.0
        self.reward = 0.0
        self.next_label_idx = len(self.tdds)
        self.last_label = None
        self.banned_edges.clear()
        self._steps_in_ep = 0
        self.max_tdd_nodes = 0
        return self.get_observation(), {}

    def _record_merge(self, u, v, new_tdd, n_nodes):
        """Store the TDD of contracting ``u`` and ``v`` under a new label."""
        new_label = f"tdd_{self.next_label_idx}"
        self.next_label_idx += 1
        self.tdds[new_label] = [new_tdd, self.tdds[u][1] + self.tdds[v][1],
                                self.tdds[u][2] + self.tdds[v][2]]
        self._tdd_size_cache[new_label] = n_nodes
        self._tdd_size_cache.pop(u, None)
        self._tdd_size_cache.pop(v, None)
        return new_label

    def step(self, action):
        start_t_total = time.perf_counter()
        if isinstance(action, (np.ndarray, list)):
            action = int(np.squeeze(action))

        edges = list(self.current_graph.edges)

        # 1. Select the edge (fallback: the first edge that is not banned).
        if edges and action < len(edges) and frozenset(edges[action]) not in self.banned_edges:
            u, v = edges[action]
        else:
            u, v = None, None
            for e in edges:
                if frozenset(e) not in self.banned_edges:
                    u, v = e
                    break
            if u is None and edges:
                return self.get_observation(), -100.0, True, False, {"reason": "all_edges_banned"}

        step_reward = 0.0

        # 2. Contract.
        if u is not None and v is not None:
            try:
                start_t = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    new_tdd = self._timed_cont(self.tdds[u][0], self.tdds[v][0])
                cont_time_s = time.perf_counter() - start_t
                if new_tdd.node_number() > NODE_GUARD:
                    raise ValueError(f"TDD larger than {NODE_GUARD} nodes")
            except ContractionTimeout:
                return self.get_observation(), -10.0, False, True, {
                    "reason": "contraction_timeout",
                    "error": f"contraction exceeded {self.cont_timeout_s}s; episode ended",
                }
            except Exception as e:
                self.banned_edges.add(frozenset({u, v}))
                return self.get_observation(), -10.0, False, False, {
                    "reason": "contraction_failed", "error": str(e)}

            # 3. Update the state.
            self.contraction_order.append((u, v))
            self.total_contraction_time += cont_time_s
            self._steps_in_ep += 1
            n_nodes = new_tdd.node_number()
            self.max_tdd_nodes = max(self.max_tdd_nodes, n_nodes)

            new_label = self._record_merge(u, v, new_tdd, n_nodes)
            self.current_graph = merge_nodes(self.current_graph, u, v, new_label)
            del self.tdds[u]
            del self.tdds[v]
            self.last_label = new_label
            self.banned_edges = {k for k in self.banned_edges
                                 if k.issubset(self.current_graph.nodes)}

            # 4. Reward.
            if not self.test_mode:
                step_reward = self._compute_reward(new_tdd)
                self.reward = step_reward

        # 5. No edges left: join the disconnected components.
        if self.current_graph.number_of_edges() == 0:
            remaining = list(self.current_graph.nodes)
            while len(remaining) > 1:
                n1, n2 = remaining[0], remaining[1]
                try:
                    res_tdd = self._timed_cont(self.tdds[n1][0], self.tdds[n2][0])
                    res_nodes = res_tdd.node_number()
                    self.max_tdd_nodes = max(self.max_tdd_nodes, res_nodes)
                    self.contraction_order.append((n1, n2))
                    new_lbl = self._record_merge(n1, n2, res_tdd, res_nodes)
                    del self.tdds[n1]
                    del self.tdds[n2]
                    self.last_label = new_lbl
                    self.current_graph.add_node(new_lbl)
                    self.current_graph.remove_node(n1)
                    self.current_graph.remove_node(n2)
                    remaining = list(self.current_graph.nodes)
                except (ContractionTimeout, Exception) as e:
                    return self.get_observation(), -10.0, True, False, {
                        "reason": "cleanup_failed", "error": str(e)}

            final_bonus = 5.0 if not self.test_mode else 0.0
            return (self.get_observation(), step_reward + final_bonus, True, False,
                    {"reason": "finished"})

        obs = self.get_observation()

        # 6. Training only: per-step wall-clock limit.
        if time.perf_counter() - start_t_total > 0.5 and not self.test_mode:
            return obs, -10.0, True, True, {"reason": "timeout"}

        return obs, step_reward, False, False, {}
