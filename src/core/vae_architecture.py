"""
GraphVAE architecture (extracted from generative_model.ipynb)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_features: int = 6,
        edge_features: int = 4,
        global_features: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
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
        self.latent_dim = latent_dim

        self.to_mu = nn.Linear(hidden_dim + global_features, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim + global_features, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)

        graph_emb = global_mean_pool(x, batch)

        num_graphs = graph_emb.size(0)
        if global_features.size(0) != num_graphs:
            global_features = global_features.view(num_graphs, -1)
        elif global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)

        graph_emb = torch.cat([graph_emb, global_features], dim=-1)

        mu = self.to_mu(graph_emb)
        logvar = self.to_logvar(graph_emb)
        return mu, logvar


class GraphDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        pocket_dim: int = 128,
        max_nodes: int = 50,
        node_features: int = 6,
        edge_features: int = 4,
        dropout: float = 0.2,
        use_pocket_conditioning: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pocket_dim = pocket_dim
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.use_pocket_conditioning = use_pocket_conditioning
        self.node_features = node_features
        self.edge_features = edge_features

        input_dim = latent_dim

        self.mlp_expand = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_pocket_conditioning:
            self.pocket_projector = nn.Linear(pocket_dim, hidden_dim // 2)
            self.mlp_expand[0] = nn.Linear(input_dim + hidden_dim // 2, hidden_dim)

        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * node_features),
        )

        self.edge_adjacency_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * max_nodes),
        )

        self.edge_type_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * max_nodes * edge_features),
        )

    def forward(self, z: torch.Tensor, pocket_embedding: Optional[torch.Tensor] = None):
        batch_size = z.size(0)

        if self.use_pocket_conditioning and pocket_embedding is not None:
            pocket_proj = self.pocket_projector(pocket_embedding)
            h_input = torch.cat([z, pocket_proj], dim=-1)
            h = self.mlp_expand(h_input)
        else:
            h = self.mlp_expand(z)

        node_logits_flat = self.node_decoder(h)
        node_logits = node_logits_flat.view(batch_size, self.max_nodes, self.node_features)

        edge_adj_flat = self.edge_adjacency_decoder(h)
        edge_adjacency = edge_adj_flat.view(batch_size, self.max_nodes, self.max_nodes)

        edge_type_flat = self.edge_type_decoder(h)
        edge_type_logits = edge_type_flat.view(batch_size, self.max_nodes, self.max_nodes, self.edge_features)

        return node_logits, edge_adjacency, edge_type_logits


class GraphVAE(nn.Module):
    def __init__(
        self,
        node_features: int = 6,
        edge_features: int = 4,
        global_features: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_encoder_layers: int = 3,
        max_nodes: int = 50,
        pocket_dim: int = 128,
        dropout: float = 0.2,
        beta: float = 1.0,
        use_pocket_conditioning: bool = True,
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            node_features=node_features,
            edge_features=edge_features,
            global_features=global_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            pocket_dim=pocket_dim,
            max_nodes=max_nodes,
            node_features=node_features,
            edge_features=edge_features,
            dropout=dropout,
            use_pocket_conditioning=use_pocket_conditioning,
        )
        self.latent_dim = latent_dim
        self.beta = beta

    def encode(self, x, edge_index, edge_attr, batch, global_features):
        return self.encoder(x, edge_index, edge_attr, batch, global_features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor, pocket_embedding: Optional[torch.Tensor] = None):
        return self.decoder(z, pocket_embedding=pocket_embedding)

    def forward(self, x, edge_index, edge_attr, batch, global_features, pocket_embedding: Optional[torch.Tensor] = None):
        mu, logvar = self.encode(x, edge_index, edge_attr, batch, global_features)
        z = self.reparameterize(mu, logvar)
        node_logits, edge_adjacency, edge_type_logits = self.decode(z, pocket_embedding)
        return node_logits, edge_adjacency, edge_type_logits, mu, logvar

    def generate(self, num_samples: int, device: str = 'cpu', pocket_embedding: Optional[torch.Tensor] = None):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, pocket_embedding)
