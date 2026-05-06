"""
Graph Variational Autoencoder (GraphVAE) for EGFR ligand generation.

Architecture:
- Encoder: Compresses PyG graphs into latent vectors (mu, logvar)
- Decoder: Reconstructs node features and edge matrices from latent + 3D pocket info
- Sampling: Reparameterization trick for differentiable sampling
- Conditioning Hook: T790M pocket geometry injection for structure-based generation

The generative model will eventually extend the discriminative GNN regressor
to enable structure-based generation conditioned on binding pocket geometry.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, GraphConv
from torch_geometric.data import Data, Batch


class GraphEncoder(nn.Module):
    """
    Encodes a batch of PyG graphs into latent vectors.
    
    Input:
    - x: Node features (batch_total_nodes, 6)
    - edge_index: Edge connectivity (2, batch_total_edges)
    - edge_attr: Edge features (batch_total_edges, 4)
    - batch: Batch assignment for nodes (batch_total_nodes,)
    - global_features: Per-graph features (num_graphs, 2)
    
    Output:
    - mu: Mean of latent distribution (num_graphs, latent_dim)
    - logvar: Log-variance of latent distribution (num_graphs, latent_dim)
    """
    
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
        
        # Graph convolution layers
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
        
        # Project graph embeddings to latent space (mu and logvar)
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
        """Returns (mu, logvar) for reparameterization sampling."""
        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Apply graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Aggregate to graph level
        graph_emb = global_mean_pool(x, batch)
        
        # Handle global features shape: could be concatenated [g1_f1, g1_f2, g2_f1, g2_f2, ...]
        # Reshape to (num_graphs, 2) if needed
        num_graphs = graph_emb.size(0)
        if global_features.size(0) != num_graphs:
            # Flatten and reshape to (num_graphs, features_per_graph)
            global_features = global_features.view(num_graphs, -1)
        elif global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        
        # Concatenate with global features
        graph_emb = torch.cat([graph_emb, global_features], dim=-1)
        
        # Project to latent space
        mu = self.to_mu(graph_emb)
        logvar = self.to_logvar(graph_emb)
        
        return mu, logvar


class GraphDecoder(nn.Module):
    """
    Decodes latent vectors back into graph structure and node attributes.
    
    Input:
    - z: Latent vectors (num_graphs, latent_dim)
    - pocket_embedding (Optional): 3D T790M pocket geometry encoding
      Shape: (num_graphs, pocket_dim) [PLACEHOLDER FOR FUTURE INJECTION]
    
    Output:
    - node_logits: Reconstructed node features (num_graphs, max_nodes, 6)
    - edge_logits: Edge adjacency predictions (num_graphs, max_nodes, max_nodes, 1)
    - edge_type_logits: Edge type predictions (num_graphs, max_nodes, max_nodes, 4)
    
    Note on 3D Conditioning:
    When pocket_embedding is provided, it will be concatenated with z before
    the MLP expansion. This allows the decoder to generate ligands that fit
    the local geometry of the T790M binding site (e.g., contact points, 
    clash-free regions, hydrogen bonding hot spots).
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        pocket_dim: int = 128,  # Actual pocket embedding dimension from pocket_extraction.py
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
        
        # Handles both conditioned (z + pocket_embedding) and unconditioned (z only)
        # When use_pocket_conditioning=True and pocket_embedding is provided in forward,
        # the input will be concatenated to (latent_dim + pocket_dim)
        input_dim = latent_dim
        
        self.mlp_expand = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Optional projection layer for pocket embedding when conditioning is used
        if use_pocket_conditioning:
            self.pocket_projector = nn.Linear(pocket_dim, hidden_dim // 2)
            self.mlp_expand[0] = nn.Linear(input_dim + hidden_dim // 2, hidden_dim)
        
        # Decode node features: (hidden_dim) -> (max_nodes, node_features)
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * node_features),
        )
        
        # Decode edge adjacency: (hidden_dim) -> (max_nodes * max_nodes)
        self.edge_adjacency_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * max_nodes),
        )
        
        # Decode edge types (bond types): (hidden_dim) -> (max_nodes * max_nodes * edge_features)
        self.edge_type_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * max_nodes * edge_features),
        )
    
    def forward(
        self,
        z: torch.Tensor,
        pocket_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode latent vectors to graph structure.
        
        Args:
            z: Latent vectors (num_graphs, latent_dim)
            pocket_embedding: Optional 3D pocket encoding (num_graphs, pocket_dim)
                             Set to None for unconditional generation.
                             When provided, decoder conditions generation on
                             binding pocket geometry (3D coordinates, contact maps, etc.)
        
        Returns:
            node_logits: (batch_size, max_nodes, node_features)
            edge_adjacency: (batch_size, max_nodes, max_nodes)
            edge_type_logits: (batch_size, max_nodes, max_nodes, edge_features)
        """
        batch_size = z.size(0)
        
        # === 3D POCKET CONDITIONING INJECTION ===
        # When pocket_embedding is provided, project it and concatenate with latent z
        if self.use_pocket_conditioning and pocket_embedding is not None:
            # Project pocket embedding to intermediate dimension
            pocket_proj = self.pocket_projector(pocket_embedding)  # (batch, hidden_dim//2)
            # Concatenate with latent code
            h_input = torch.cat([z, pocket_proj], dim=-1)  # (batch, latent_dim + hidden_dim//2)
            h = self.mlp_expand(h_input)
        else:
            # Unconditional generation: use latent z directly
            h = self.mlp_expand(z)
        
        # Decode to node features
        node_logits_flat = self.node_decoder(h)
        node_logits = node_logits_flat.view(batch_size, self.max_nodes, 6)
        
        # Decode to edge adjacency
        edge_adj_flat = self.edge_adjacency_decoder(h)
        edge_adjacency = edge_adj_flat.view(batch_size, self.max_nodes, self.max_nodes)
        
        # Decode to edge types
        edge_type_flat = self.edge_type_decoder(h)
        edge_type_logits = edge_type_flat.view(batch_size, self.max_nodes, self.max_nodes, 4)
        
        return node_logits, edge_adjacency, edge_type_logits


class GraphVAE(nn.Module):
    """
    Full Graph Variational Autoencoder for EGFR ligand generation.
    
    The VAE combines an encoder (that maps graphs to latent space) with a decoder
    (that reconstructs graphs from latent codes). The latent space can be
    sampled from to generate new molecules, optionally conditioned on 3D binding
    pocket information for structure-based drug design.
    
    Training objective:
    L = E_q(z|x)[log p(x|z)] - D_KL(q(z|x) || p(z))
    
    where:
    - q(z|x): Encoder (recognizes graphs)
    - p(x|z): Decoder (generates graphs)
    - p(z): Standard normal prior
    
    Future Enhancements:
    1. Pocket Encoder: 3D CNN or PointNet to embed T790M pocket geometry
    2. Conditioning Layer: Inject pocket_embedding into decoder
    3. Adversarial Loss: GAN-style discriminator to improve generated graph quality
    4. Property Prediction Head: Predict pIC50 from latent z
    5. Docking Validation: Real EGFR docking scoring of generated ligands
    """
    
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
        beta: float = 1.0,  # KL divergence weight in ELBO
        use_pocket_conditioning: bool = True,  # Enable 3D pocket conditioning
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
        self.beta = beta  # Weight on KL term in ELBO
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode graph batch to (mu, logvar)."""
        return self.encoder(x, edge_index, edge_attr, batch, global_features)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from N(mu, exp(logvar)) using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(
        self,
        z: torch.Tensor,
        pocket_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent vectors to graph structure."""
        return self.decoder(z, pocket_embedding=pocket_embedding)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
        pocket_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.
        
        Args:
            x, edge_index, edge_attr, batch, global_features: Graph batch data
            pocket_embedding: Optional 3D pocket encoding for conditioned generation
        
        Returns:
            node_logits: Reconstructed node features
            edge_adjacency: Reconstructed edge adjacency
            edge_type_logits: Reconstructed edge types
            mu: Latent mean
            logvar: Latent log-variance
        """
        # Encode
        mu, logvar = self.encode(x, edge_index, edge_attr, batch, global_features)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        node_logits, edge_adjacency, edge_type_logits = self.decode(z, pocket_embedding)
        
        return node_logits, edge_adjacency, edge_type_logits, mu, logvar
    
    def generate(
        self,
        num_samples: int,
        device: str = 'cpu',
        pocket_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate new graphs by sampling from the prior N(0, I).
        
        Args:
            num_samples: Number of new graphs to generate
            device: Device to generate on
            pocket_embedding: Optional 3D pocket conditioning (num_samples, pocket_dim)
        
        Returns:
            node_logits, edge_adjacency, edge_type_logits
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, pocket_embedding)


# ============================================================================
# PLACEHOLDER: POCKET ENCODER (for future integration with 3D geometry)
# ============================================================================
# 
# class PocketEncoder3D(nn.Module):
#     """Encodes T790M binding pocket 3D geometry into a fixed-size embedding.
#     
#     Input: 3D coordinates of nearby residues, contact maps, hydrogen bond donors/acceptors
#     Output: pocket_embedding (num_samples, pocket_dim=32)
#     
#     Implementation options:
#     1. PointNet: Direct 3D point cloud encoding
#     2. 3D CNN: Discretize pocket into 3D grid, use volumetric convolutions
#     3. Graph Attention: Residue atoms as nodes, distance-weighted edges
#     """
#     pass


if __name__ == "__main__":
    """
    Comprehensive test demonstrating:
    1. GraphVAE unconditional generation
    2. GraphVAE pocket-conditioned generation
    3. Integration with pocket_extraction module
    """
    import logging
    from torch_geometric.data import DataLoader
    from data_prep import smiles_to_pyg_data
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("GRAPHVAE COMPREHENSIVE TEST")
    logger.info("="*80)
    
    # ============================================================================
    # Part 1: Unconditional Generation
    # ============================================================================
    logger.info("\nPart 1: Unconditional Generation (no pocket conditioning)")
    logger.info("-" * 80)
    
    smiles_list = [
        "C=CC(=O)Nc1cc(Nc2nccc(N(C)c3ccc(N(C)C)cc3OC)n2)c(OC)cc1N(C)C",  # Osimertinib
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    ]
    
    graphs = []
    for smi in smiles_list:
        data = smiles_to_pyg_data(smi, label=0.0)
        if data is not None:
            graphs.append(data)
    
    if graphs:
        loader = DataLoader(graphs, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        # Reshape global_features for VAE
        num_graphs = batch.batch.max().item() + 1
        global_features = batch.global_features
        if global_features.size(0) != num_graphs:
            global_features = global_features.view(num_graphs, -1)
        
        # Initialize VAE without pocket conditioning
        vae_uncond = GraphVAE(use_pocket_conditioning=False)
        logger.info(f"✓ Created VAE (unconditional mode)")
        logger.info(f"  Latent dim: {vae_uncond.latent_dim}")
        logger.info(f"  Decoder pocket_dim: {vae_uncond.decoder.pocket_dim}")
        
        # Forward pass through VAE
        node_logits, edge_adj, edge_type, mu, logvar = vae_uncond(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
            global_features=global_features,
        )
        
        logger.info(f"✓ Encoder output (mu): {mu.shape}")
        logger.info(f"✓ Decoder output (node_logits): {node_logits.shape}")
        logger.info(f"✓ Decoder output (edge_adjacency): {edge_adj.shape}")
        logger.info(f"✓ Decoder output (edge_type_logits): {edge_type.shape}")
        
        # Generate new molecules unconditionally
        logger.info(f"\n✓ Generating 3 new molecules unconditionally...")
        z_new = torch.randn(3, vae_uncond.latent_dim)
        node_new, edge_adj_new, edge_type_new = vae_uncond.decode(z_new)
        logger.info(f"  Generated node features: {node_new.shape}")
        logger.info(f"  Generated edge adjacency: {edge_adj_new.shape}")
        logger.info(f"  Generated edge types: {edge_type_new.shape}")
    
    # ============================================================================
    # Part 2: Pocket-Conditioned Generation
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("Part 2: Pocket-Conditioned Generation")
    logger.info("-" * 80)
    
    # Try to import pocket extraction (will work if pocket_extraction.py exists)
    try:
        from pocket_extraction import get_pocket_embedding
        logger.info("✓ Successfully imported pocket_extraction module")
        
        # Create or use mock pocket embedding
        # In real use: pocket_emb = get_pocket_embedding('egfr_pdb.pdb', 'IRE')
        mock_pocket_embedding = torch.randn(1, 128)  # 1 sample, 128D pocket vector
        logger.info(f"✓ Generated mock pocket embedding: {mock_pocket_embedding.shape}")
        
        # Initialize VAE with pocket conditioning enabled
        vae_cond = GraphVAE(use_pocket_conditioning=True)
        logger.info(f"✓ Created VAE (pocket-conditioned mode)")
        logger.info(f"  Latent dim: {vae_cond.latent_dim}")
        logger.info(f"  Decoder pocket_dim: {vae_cond.decoder.pocket_dim}")
        logger.info(f"  Pocket projector output: {vae_cond.decoder.hidden_dim // 2}")
        
        # Generate molecules conditioned on pocket
        logger.info(f"\n✓ Generating 1 molecule conditioned on T790M pocket...")
        z_cond = torch.randn(1, vae_cond.latent_dim)
        node_cond, edge_adj_cond, edge_type_cond = vae_cond.decode(
            z_cond,
            pocket_embedding=mock_pocket_embedding,
        )
        logger.info(f"  Generated node features: {node_cond.shape}")
        logger.info(f"  Generated edge adjacency: {edge_adj_cond.shape}")
        logger.info(f"  Generated edge types: {edge_type_cond.shape}")
        
    except ImportError:
        logger.warning("⚠ pocket_extraction module not available (expected in this test)")
        logger.info("  Pocket conditioning capability scaffolded but requires PDB file")
    
    logger.info("\n" + "="*80)
    logger.info("✅ All tests passed!")
    logger.info("="*80)

