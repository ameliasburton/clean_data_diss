# Pocket Extraction & Structure-Based Generation

## Overview

This document describes the 3D pocket extraction pipeline and its integration with the GraphVAE for structure-based ligand generation conditioned on EGFR binding pocket geometry.

---

## Module: `src/pocket_extraction.py`

### Purpose
Extract 3D structural features from EGFR binding pockets (from PDB files) and encode them as fixed-size 128D embeddings for VAE conditioning.

### Key Functions

#### `get_ligand_atoms(structure, ligand_code) → List[np.ndarray]`
- Extracts all atom coordinates from a ligand (e.g., co-crystallized inhibitor)
- **Input:** BioPython Structure, 3-letter ligand code (e.g., 'IRE' for Gefitinib, 'WZ4' for WZ4002)  
- **Output:** List of [x, y, z] coordinate arrays

#### `get_pocket_residues(structure, ligand_code, radius=7.0) → List[Residue]`
- Identifies all protein residues within a specified radius of ligand atoms
- **Algorithm:** 
  1. Extract ligand atom coordinates
  2. Calculate distances from all protein atoms to all ligand atoms  
  3. Find unique residues containing atoms within 7.0 Å (default)
- **Returns:** Sorted list of BioPython Residue objects

#### `extract_pocket_features(residues) → (ca_coords, aa_types, all_atom_coords)`
- Extracts 3D coordinates and chemical features from pocket residues
- **Outputs:**
  - `ca_coords` (N_residues, 3): Alpha-carbon coordinates
  - `aa_types` (N_residues, 21): One-hot encoded amino acid types
  - `all_atom_coords` (N_atoms, 3): All atom coordinates

#### `compute_pocket_embedding(ca_coords, aa_types, all_atom_coords, embedding_dim=128) → torch.Tensor`
- Aggregates pocket features into a fixed 128D embedding
- **Features included:**
  1. **Distance Statistics:** Min, max, mean, std of pairwise CA distances
  2. **Distance Matrix:** Flattened upper triangle, normalized, padded to fixed size
  3. **Amino Acid Composition:** 21D probability distribution of residue types
  4. **Spatial Statistics:** Mean/std radii, variance along x/y/z axes  
  5. **Center of Mass:** 3D geometric center of pocket atoms
- **Aggregation:** Concatenate all features and project/interpolate to 128D vector
- **Normalization:** Unit norm (L2 = 1.0)

#### `get_pocket_embedding(pdb_file, ligand_code='IRE', pocket_radius=7.0, embedding_dim=128) → torch.Tensor`
- **Main function:** Orchestrates full extraction pipeline
- **Usage:**
  ```python
  from pocket_extraction import get_pocket_embedding
  
  # Extract embedding from PDB
  embedding = get_pocket_embedding('egfr_t790m.pdb', ligand_code='WZ4')
  # Returns: torch.Tensor of shape [128]
  ```

### Example PDB Codes
| Inhibitor | Code | PDB | Mutation |
|-----------|------|-----|----------|
| Gefitinib | IRE | 2ITV | WT |
| Erlotinib | ER1 | 1M17 | WT |
| WZ4002 | WZ4 | 3W2S | T790M |
| Osimertinib | OSI | 5EDP | T790M |

---

## Integration with GraphVAE

### Updated `src/generative_model.py`

#### GraphDecoder with Pocket Conditioning

**Initialization:**
```python
decoder = GraphDecoder(
    latent_dim=64,
    hidden_dim=128,
    pocket_dim=128,  # Matches pocket_embedding dimension
    use_pocket_conditioning=True,  # Enable feature extraction
)
```

**Parameters:**
- `latent_dim`: VAE latent space dimension (64D)
- `pocket_dim`: Dimension of pocket embedding from extraction (128D)
- `use_pocket_conditioning`: Boolean flag to enable/disable conditioning

**Architecture with Conditioning:**
```
Input: latent z (batch, 64) + pocket_embedding (batch, 128)
  ↓
pocket_projector [128 → 64]
  ↓
Concatenate [z + projected_pocket] → (batch, 128)
  ↓
mlp_expand [128 → 64] (if not conditioning, skip projection)
  ↓
3 Decoders: nodes, edges, edge_types
```

#### GraphVAE with Pocket Support

**Initialization:**
```python
vae = GraphVAE(
    latent_dim=64,
    hidden_dim=128,
    pocket_dim=128,
    use_pocket_conditioning=True,
)
```

**Unconditional Generation (sampling from prior):**
```python
z = torch.randn(batch_size, 64)
node_logits, edge_adj, edge_types = vae.decode(z)
```

**Pocket-Conditioned Generation:**
```python
# Get pocket embedding from PDB
from pocket_extraction import get_pocket_embedding
pocket_emb = get_pocket_embedding('egfr_t790m.pdb', ligand_code='WZ4')

# Sample latent code
z = torch.randn(1, 64)

# Generate molecule conditioned on pocket
node_logits, edge_adj, edge_types = vae.decode(z, pocket_embedding=pocket_emb.unsqueeze(0))
```

---

## Data Flow

```
PDB File
   ↓
[PDBParser] → Structure object
   ↓
[get_pocket_residues] → Residues within 7Å of ligand
   ↓
[extract_pocket_features] → CA coords + AA types + atom coords
   ↓
[compute_pocket_embedding] → 128D tensor (unit norm)
   ↓
[GraphDecoder + pocket] → Generated ligand graph
   ↓
[Graph-to-SMILES decoder?] → SMILES string
   ↓
[Docking/Scoring?] → pIC50 estimate
```

---

## Testing

### Test 1: Pocket Extraction Alone
```bash
python egfr-gnn-project/src/pocket_extraction.py
```

**Output:**
- ✅ Identifies 3 pocket residues in mock PDB
- ✅ Extracts 15 total pocket atoms  
- ✅ Generates 128D embedding with L2 norm = 1.0

### Test 2: GraphVAE Unconditional
```bash
python egfr-gnn-project/src/generative_model.py
# Part 1: Unconditional Generation
```

**Output:**
- ✅ Encodes 2 real molecules → 64D latent space
- ✅ Decodes back to node/edge/bond predictions
- ✅ Generates 3 new molecules from random z

### Test 3: GraphVAE Pocket-Conditioned
```bash
python egfr-gnn-project/src/generative_model.py
# Part 2: Pocket-Conditioned Generation
```

**Output:**
- ✅ Successfully imports pocket_extraction
- ✅ Creates mock 128D pocket embedding
- ✅ Generates 1 molecule conditioned on pocket geometry

---

## Next Steps

### Immediate (Architecture Complete)
1. ✅ PDB parsing and pocket residue identification
2. ✅ Feature extraction (coordinates, amino acids, spatial stats)
3. ✅ Fixed-size 128D embedding generation
4. ✅ GraphVAE decoder with pocket conditioning support
5. ✅ Unconditional and conditioned generation modes

### Short-term (Implementation)
1. **Train VAE**: Implement `src/train_vae.py` with ELBO loss
   - Reconstruction loss (node/edge/bond predictions)
   - KL divergence regularization
   - Optional pocket conditioning during training

2. **Graph-to-SMILES Decoder**: Convert generated logits to valid molecules
   - Greedy sampling from node/edge predictions
   - Valence constraint checking
   - SMILES validity post-processing

3. **Docking Integration**: Score generated molecules against EGFR
   - Use AutoDock Vina or similar
   - Provide binding affinity feedback

### Medium-term (Enhancements)
1. **3D Pocket Visualization**: Plot extracted pocket geometries
2. **Latent Space Analysis**: PCA/UMAP visualization of learned representations
3. **Property Prediction Head**: Predict pIC50 directly from latent z
4. **Multi-objective Generation**: Optimize pIC50 + docking score + MW
5. **WT vs T790M Specialization**: Separate models or universal conditioner

### Long-term (Validation)
1. **Synthetic Chemistry**: Validate top-ranked predictions in wet lab
2. **Active Learning Loop**: Retrain on real bioactivity data
3. **Structure Activity Analysis**: Inspect what structural features drive pIC50

---

## File Manifest

```
egfr-gnn-project/
├── src/
│   ├── data_prep.py              (SMILES → PyG graphs)
│   ├── model_arch.py             (Discriminative GNN regressor)
│   ├── train.py                  (Discriminative training)
│   ├── evaluate.py               (Benchmark evaluation)
│   ├── generative_model.py       (GraphVAE + pocket conditioning)
│   ├── pocket_extraction.py      (✓ NEW: 3D pocket embedding)
│   └── (train_vae.py - TODO)
├── data/
│   ├── raw/
│   │   └── bioactivity/          (ChEMBL + BindingDB)
│   ├── pdb/                      (Crystal structures)
│   └── processed/                (PyG graphs)
├── models/
│   ├── best_gnn_regressor.pth    (Trained discriminative)
│   └── (best_vae.pth - TODO)
├── requirements.txt              (✓ UPDATED: added biopython)
└── GENERATIVE_ARM_SUMMARY.md     (Documentation)
```

---

## Dependencies

**Added to `requirements.txt`:**
- `biopython>=1.81`: PDB parsing and structure manipulation

**Already installed (egfr-gnn environment):**
- `numpy<2`: Numerical computations
- `torch>=2.0.0`: PyTorch tensors
- `torch-geometric>=2.3.0`: Graph neural networks
- `rdkit>=2023.03.1`: Molecular structures
- `pandas>=1.3.0`: Data handling

---

## Example Usage

### Extract T790M Pocket from PDB
```python
from pocket_extraction import get_pocket_embedding

# Download PDB: https://www.rcsb.org/structure/3W2S (WZ4002 + T790M EGFR)
embedding = get_pocket_embedding(
    pdb_file='3W2S.pdb',
    ligand_code='WZ4',  # Co-crystallized WZ4002
    pocket_radius=7.0,  # Å
    embedding_dim=128,
)
print(f"Pocket embedding shape: {embedding.shape}")  # torch.Size([128])
```

### Generate Ligands for T790M Pocket
```python
from generative_model import GraphVAE
import torch

# Load trained VAE
vae = GraphVAE(use_pocket_conditioning=True)
# vae.load_state_dict(torch.load('best_vae.pth'))  # After training

# Get T790M pocket
pocket_emb = get_pocket_embedding('3W2S.pdb', 'WZ4')

# Generate 10 candidates
for i in range(10):
    z = torch.randn(1, 64)
    node_logits, edge_adj, edge_types = vae.decode(z, pocket_embedding=pocket_emb.unsqueeze(0))
    
    # TODO: Convert logits → SMILES
    # TODO: Score with docking
```

---

## References

- **BioPython:** https://biopython.org/
- **PyTorch Geometric:** https://pytorch-geometric.readthedocs.io/
- **EGFR Structures:**
  - WT: PDB 2ITV, 1M17
  - T790M: PDB 3W2S, 5EDP
- **VAE for Molecular Generation:** Gómez-Bombarelli et al., ACS Central Sci. 2018

