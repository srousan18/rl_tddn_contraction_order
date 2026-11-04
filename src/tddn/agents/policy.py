"""Graph policy: a GNN encoder with an edge-scoring actor head.

Encoder: node and edge embeddings, three GINE layers, two GAT layers and a
TransformerConv layer (residual connections), mean pooling, and an MLP over
the global features. The actor scores every edge from the embeddings of its
two end nodes, the edge embedding and the global embedding; infeasible edges
are masked. The critic is a linear head on the graph embedding.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GINEConv, TransformerConv, global_mean_pool


class HybridGraphFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, *, hidden_dim=128, heads=4,
                 n_gine=3, n_gat=2, dropout=0.05, residual=True,
                 features_dim=256):
        super().__init__(observation_space, features_dim)

        node_in = observation_space.spaces["node_features"].shape[-1]
        edge_in = observation_space.spaces["edge_features"].shape[-1]
        global_in = observation_space.spaces["global_features"].shape[-1]

        self.hidden_dim = hidden_dim
        self.residual = residual
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.node_emb = nn.Linear(node_in, hidden_dim)
        self.edge_emb = nn.Linear(edge_in, hidden_dim)

        def make_gine_mlp():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.gine_convs = nn.ModuleList([
            GINEConv(make_gine_mlp(), edge_dim=hidden_dim) for _ in range(n_gine)
        ])
        self.gat_convs = nn.ModuleList([
            GATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads,
                    heads=heads, concat=True, dropout=dropout)
            for _ in range(n_gat)
        ])
        self.global_att = TransformerConv(
            in_channels=hidden_dim, out_channels=hidden_dim, heads=heads,
            concat=False, dropout=dropout, edge_dim=hidden_dim,
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(global_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.final_lin = nn.Linear(hidden_dim + hidden_dim, features_dim)

        # Node embeddings of the last forward pass, read by the actor head.
        self.last_batch_struct = None

    def forward(self, obs):
        B = obs["node_features"].shape[0]
        data_list = []
        batch_edge_slices = []
        start_edge = 0

        for i in range(B):
            n = int(obs["num_nodes"][i])
            e = int(obs["num_edges"][i])
            x = self.node_emb(obs["node_features"][i, :n])
            ei = obs["edge_index"][i, :, :e].long()
            ea = self.edge_emb(obs["edge_features"][i, :e])
            data_list.append(Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=n))
            batch_edge_slices.append((start_edge, start_edge + e))
            start_edge += e

        batch = Batch.from_data_list(data_list).to(obs["node_features"].device)
        h = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        for conv in self.gine_convs:
            h_new = torch.relu(conv(h, edge_index, edge_attr=edge_attr))
            h = h + h_new if self.residual else h_new
            h = self.dropout(h)

        for conv in self.gat_convs:
            h_new = torch.relu(conv(h, edge_index))
            h = h + h_new if self.residual else h_new
            h = self.dropout(h)

        h_trans = torch.relu(self.global_att(h, edge_index, edge_attr=edge_attr))
        h = h + h_trans if self.residual else h_trans
        h = self.dropout(h)

        h_g = global_mean_pool(h, batch.batch)

        g_raw = obs["global_features"]
        if g_raw.dim() == 3:
            g_raw = g_raw.squeeze(1)
        g_proc = self.global_mlp(g_raw)

        graph_embed = torch.relu(self.final_lin(torch.cat([h_g, g_proc], dim=-1)))

        self.last_batch_struct = dict(
            h_nodes=h,
            batch=batch,
            edge_attr=edge_attr,
            batch_edge_slices=batch_edge_slices,
            global_proc=g_proc,
        )
        return graph_embed


class IdentityMlpExtractor(nn.Module):
    def forward(self, x):
        return (x, x)

    def forward_actor(self, x):
        return x

    def forward_critic(self, x):
        return x


class HybridGraphActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, hidden_dim: int = 256,
                 n_gine: int = 3, n_gat: int = 2, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=HybridGraphFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=hidden_dim,
                hidden_dim=hidden_dim,
                n_gine=n_gine,
                n_gat=n_gat,
                heads=4,
                dropout=0.05,
                residual=True,
            ),
        )

        self.mlp_extractor = IdentityMlpExtractor()
        latent_dim = self.features_extractor.features_dim

        # Critic head.
        self.value_net = nn.Linear(latent_dim, 1)

        # Actor head: one score per edge.
        hidden_dim = self.features_extractor.features_dim
        hidden_edge = 128
        self.edge_scorer = nn.Sequential(
            nn.Linear((hidden_dim * 2) + hidden_dim + hidden_dim, hidden_edge),
            nn.ReLU(),
            nn.Linear(hidden_edge, 1),
        )
        self.action_net = self.edge_scorer

    def _edge_logits_from_extractor_state(self, obs_shape):
        """Edge logits from the node embeddings of the last encoder pass."""
        st = self.features_extractor.last_batch_struct
        h_nodes = st["h_nodes"]
        edge_attr = st["edge_attr"]
        batch = st["batch"]
        batch_edge_slices = st["batch_edge_slices"]
        global_proc = st["global_proc"]
        edge_index = batch.edge_index

        B = obs_shape[0]
        MAX_EDGES = obs_shape[1]

        all_logits = []
        for b in range(B):
            e_start, e_end = batch_edge_slices[b]
            if e_end <= e_start:
                all_logits.append(torch.full((MAX_EDGES,), -1e9, device=h_nodes.device))
                continue

            these_edges = torch.arange(e_start, e_end, device=h_nodes.device)
            u_idx = edge_index[0, these_edges]
            v_idx = edge_index[1, these_edges]
            h_u = h_nodes[u_idx]
            h_v = h_nodes[v_idx]
            e_a = edge_attr[these_edges]
            g_b = global_proc[b].expand_as(h_u)

            edge_inp = torch.cat([h_u, h_v, e_a, g_b], dim=-1)
            logits_local = self.edge_scorer(edge_inp).squeeze(-1)

            pad_len = MAX_EDGES - logits_local.shape[0]
            if pad_len > 0:
                logits_b = torch.cat([logits_local,
                                      torch.full((pad_len,), -1e9, device=h_nodes.device)], dim=0)
            else:
                logits_b = logits_local[:MAX_EDGES]
            all_logits.append(logits_b)

        return torch.stack(all_logits, dim=0)

    def _masked_logits(self, obs, graph_embed):
        max_edges = obs["edge_features"].shape[1]
        logits = self._edge_logits_from_extractor_state((graph_embed.shape[0], max_edges))
        feas = obs["feasible_actions"].bool()
        return logits.masked_fill(~feas, -1e9)

    def forward(self, obs, deterministic=False):
        graph_embed = self.extract_features(obs)
        value = self.value_net(graph_embed)
        dist = self._get_action_dist_from_logits(self._masked_logits(obs, graph_embed))
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, value, log_prob

    def evaluate_actions(self, obs, actions):
        graph_embed = self.extract_features(obs)
        value = self.value_net(graph_embed)
        dist = self._get_action_dist_from_logits(self._masked_logits(obs, graph_embed))
        return value, dist.log_prob(actions), dist.entropy()

    def predict_values(self, obs):
        return self.value_net(self.extract_features(obs))

    def _predict(self, observation, deterministic=False):
        action, _, _ = self.forward(observation, deterministic=deterministic)
        return action

    def _get_action_dist_from_logits(self, logits, deterministic=False):
        dist = CategoricalDistribution(self.action_space.n)
        dist.proba_distribution(action_logits=logits)
        return dist
