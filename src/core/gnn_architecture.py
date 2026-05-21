"""
EGFR GNN regressor (extracted from model_arch.ipynb)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool


class EGFR_GNN_Regressor(nn.Module):
    def __init__(
        self,
        node_features: int = 8,
        edge_features: int = 5,
        global_features: int = 20,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 4,
        dropout_rate: float = 0.3,
        fingerprint_dim: int = 1024,
        fingerprint_hidden: int = 128,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.node_encoder = nn.Linear(node_features, hidden_dim)
        # We always append a normalized distance scalar to edge features (if pos provided)
        self.edge_input_dim = edge_features + 1
        self.edge_encoder = nn.Linear(self.edge_input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp))

        self.dropout_rate = dropout_rate
        self.global_features = global_features
        self.fingerprint_dim = fingerprint_dim
        self.fingerprint_hidden = fingerprint_hidden if fingerprint_dim and fingerprint_dim > 0 else 0

        # Fingerprint reducer (strong regularization to avoid memorization)
        if self.fingerprint_dim and self.fingerprint_hidden:
            self.fp_reducer = nn.Sequential(
                nn.Linear(self.fingerprint_dim, self.fingerprint_hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
            )
        else:
            self.fp_reducer = None

        # Multi-strategy pooling: concatenate mean + max (2 * hidden_dim) + global features + fingerprint_hidden
        mlp_input = 2 * hidden_dim + global_features + (self.fingerprint_hidden if self.fp_reducer is not None else 0)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        fingerprint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("edge_attr is required for EGFR_GNN_Regressor")

        # Calculate 3D Euclidean distances if pos is provided
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            if pos is not None:
                distances = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)
                distances = distances / 3.0
                distances = torch.clamp(distances, 0, 1)
            else:
                distances = torch.zeros((edge_attr.size(0), 1), device=edge_attr.device, dtype=edge_attr.dtype)
            # Concatenate normalized distance to edge attributes (ensures consistent shape)
            edge_attr = torch.cat([edge_attr, distances], dim=1)

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Multi-strategy pooling: concatenate mean + max for robustness
        graph_emb_mean = global_mean_pool(x, batch)
        graph_emb_max = global_max_pool(x, batch)
        graph_emb = torch.cat([graph_emb_mean, graph_emb_max], dim=1)

        # Prepare global features and optional fingerprint
        if global_features is None:
            gf = graph_emb_mean.new_zeros((graph_emb_mean.size(0), self.global_features))
        else:
            gf = global_features.view(graph_emb_mean.size(0), -1).to(graph_emb_mean.dtype)

        if fingerprint is not None and self.fp_reducer is not None:
            fp_reduced = self.fp_reducer(fingerprint)
            out = torch.cat([graph_emb, gf, fp_reduced], dim=-1)
        else:
            out = torch.cat([graph_emb, gf], dim=-1)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.mlp(out)
        return out.view(-1)


if __name__ == "__main__":
    model = EGFR_GNN_Regressor()
    print(model)
