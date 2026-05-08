"""
EGFR GNN regressor (extracted from model_arch.ipynb)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool


class EGFR_GNN_Regressor(nn.Module):
    def __init__(
        self,
        node_features: int = 6,
        edge_features: int = 4,
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
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_features, hidden_dim),
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
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("edge_attr is required for EGFR_GNN_Regressor")

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)

        graph_emb = global_mean_pool(x, batch)

        if global_features is None:
            global_features = graph_emb.new_zeros((graph_emb.size(0), self.global_features))
        else:
            global_features = global_features.view(graph_emb.size(0), -1).to(graph_emb.dtype)

        out = torch.cat([graph_emb, global_features], dim=-1)
        out = self.mlp(out)
        return out.view(-1)


if __name__ == "__main__":
    model = EGFR_GNN_Regressor()
    print(model)
