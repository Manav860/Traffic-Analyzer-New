"""
Graph Neural Network (GNN) World Model.
Learns graph-level and node-level transition dynamics P(G_t+1 | G_t) using
spatial message-passing Graph Convolutions combined with temporal recurrent dynamics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class GraphConvolution(nn.Module):
    """
    Spatial Graph Convolution layer implementing degree-normalized message passing:
    H^(l+1) = ReLU( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^(l) W )
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Num_Nodes, In_Features]
        adj: [Batch, Num_Nodes, Num_Nodes] (adjacency matrix)
        """
        # Add self-loops to adjacency matrix
        batch_size, num_nodes, _ = adj.shape
        eye = torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_tilde = adj + eye

        # Degree normalization: D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
        deg = torch.sum(adj_tilde, dim=-1) # [Batch, Num_Nodes]
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
        deg_mat = torch.diag_embed(deg_inv_sqrt) # [Batch, Num_Nodes, Num_Nodes]
        norm_adj = torch.bmm(torch.bmm(deg_mat, adj_tilde), deg_mat)

        # Message passing: A_norm * X * W
        support = torch.matmul(x, self.weight) # [Batch, Num_Nodes, Out_Features]
        output = torch.bmm(norm_adj, support)  # [Batch, Num_Nodes, Out_Features]
        if self.bias is not None:
            output = output + self.bias
        return F.relu(output)


class GNNWorldModel(nn.Module):
    """
    Spatio-Temporal Graph Neural Network World Model:
    1. Spatial GCN Encoder: extracts host communication structural features.
    2. Global Graph Readout Pooling: summarizes topological network state into latent vector z_t.
    3. Recurrent Transition Dynamics (GRU): models temporal progression of network state.
    4. Multi-Task Heads: P(S_t+1 | S_t), Infiltration Risk, and MITRE ATT&CK Stage classification.
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        gnn_hidden_dim: int = 64,
        latent_dim: int = 128,
        num_stages: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.latent_dim = latent_dim
        self.num_stages = num_stages

        # Spatial Message Passing
        self.gcn1 = GraphConvolution(node_in_dim, gnn_hidden_dim)
        self.gcn2 = GraphConvolution(gnn_hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

        # Temporal Transition Dynamics
        self.temporal_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True
        )

        # Head 1: Future Graph Latent State Dynamics P(z_t+1 | z_t)
        self.dynamics_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim)
        )

        # Head 2: Infiltration Risk Probability Head
        self.infil_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )

        # Head 3: MITRE ATT&CK Kill-Chain Stage Head
        self.stage_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_stages)
        )

    def encode_graph(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Encodes node features and adjacency into a graph-level latent embedding.
        node_feats: [Batch, Num_Nodes, Node_In_Dim]
        adj: [Batch, Num_Nodes, Num_Nodes]
        Returns: [Batch, Latent_Dim]
        """
        h1 = self.gcn1(node_feats, adj)
        h1 = self.dropout(h1)
        h2 = self.gcn2(h1, adj)
        # Global graph readout pooling (mean + max pooling)
        mean_pool = torch.mean(h2, dim=1)
        max_pool = torch.max(h2, dim=1)[0]
        graph_latent = (mean_pool + max_pool) * 0.5
        return graph_latent

    def forward(
        self,
        node_seq: torch.Tensor,
        adj_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        node_seq: [Batch, Seq_Len, Num_Nodes, Node_In_Dim]
        adj_seq: [Batch, Seq_Len, Num_Nodes, Num_Nodes]
        """
        batch_size, seq_len, num_nodes, _ = node_seq.shape
        latent_list = []

        for t in range(seq_len):
            n_t = node_seq[:, t, :, :]
            a_t = adj_seq[:, t, :, :]
            z_t = self.encode_graph(n_t, a_t)
            latent_list.append(z_t)

        # [Batch, Seq_Len, Latent_Dim]
        temporal_latents = torch.stack(latent_list, dim=1)
        gru_out, h_n = self.temporal_gru(temporal_latents, hidden)
        last_context = gru_out[:, -1, :] # [Batch, Latent_Dim]

        # Predictions
        next_latent = self.dynamics_head(last_context)
        infil_prob = self.infil_head(last_context)
        stage_logits = self.stage_head(last_context)

        return {
            "predicted_next_latent": next_latent,
            "infiltration_probability": infil_prob,
            "stage_logits": stage_logits,
            "context_vector": last_context,
            "gru_hidden": h_n
        }
