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
        global_features: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp))

        self.dropout = nn.Dropout(dropout)
        self.global_features = global_features
        # Multi-strategy pooling: concatenate mean + max (2 * hidden_dim) + global features
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("edge_attr is required for EGFR_GNN_Regressor")

        # Calculate 3D Euclidean distances if pos is provided
        if pos is not None and edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            distances = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)
            # Normalize distances to [0, 1] range (typical molecular bonds: 1-3 Å)
            distances = distances / 3.0
            distances = torch.clamp(distances, 0, 1)
            # Concatenate normalized distance to edge attributes
            edge_attr = torch.cat([edge_attr, distances], dim=1)

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)

        # Multi-strategy pooling: concatenate mean + max for robustness
        graph_emb_mean = global_mean_pool(x, batch)
        graph_emb_max = global_max_pool(x, batch)
        graph_emb = torch.cat([graph_emb_mean, graph_emb_max], dim=1)

        if global_features is None:
            global_features = graph_emb_mean.new_zeros((graph_emb_mean.size(0), self.global_features))
        else:
            global_features = global_features.view(graph_emb_mean.size(0), -1).to(graph_emb_mean.dtype)

        out = torch.cat([graph_emb, global_features], dim=-1)
        out = self.mlp(out)
        return out.view(-1)


if __name__ == "__main__":
    model = EGFR_GNN_Regressor()
    print(model)
