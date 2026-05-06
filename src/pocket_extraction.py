"""
Pocket Extraction and 3D Embedding Generator

Extracts 3D structural features from EGFR binding pockets (PDB files) and
encodes them as fixed-size embeddings for VAE conditioning.

Features extracted:
- Pocket residue identification (within 7Å of ligand)
- Spatial coordinates and atom types
- Distance matrices and statistical moments
- Aggregation into 128D fixed-size tensor

Usage:
    from pocket_extraction import get_pocket_embedding
    embedding = get_pocket_embedding('structure.pdb', ligand_code='IRE')
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from Bio.PDB import PDBParser, NeighborSearch, PPBuilder
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore', category=DeprecationWarning)


# Standard amino acid one-hot encoding
AMINO_ACIDS = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20  # Unknown
}
NUM_AA_TYPES = 21


def get_ligand_atoms(structure: Structure, ligand_code: str) -> List[np.ndarray]:
    """
    Extract all atom coordinates from a ligand in the structure.
    
    Args:
        structure: BioPython Structure object
        ligand_code: 3-letter code for ligand (e.g., 'IRE', 'WZ4', 'HEM')
    
    Returns:
        List of atom coordinates as [x, y, z] arrays
    """
    ligand_atoms = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == ligand_code:
                    for atom in residue:
                        coords = atom.get_coord()
                        ligand_atoms.append(coords)
    
    if not ligand_atoms:
        logger.warning(f"No ligand atoms found for code '{ligand_code}' in structure")
    
    return ligand_atoms


def get_pocket_residues(
    structure: Structure,
    ligand_code: str,
    radius: float = 7.0,
) -> List[Residue]:
    """
    Identify all protein residues within a specified radius of ligand atoms.
    
    Args:
        structure: BioPython Structure object
        ligand_code: 3-letter code for ligand
        radius: Distance threshold in Ångströms (default 7.0)
    
    Returns:
        List of Residue objects near the ligand
    """
    ligand_atoms = get_ligand_atoms(structure, ligand_code)
    
    if not ligand_atoms:
        logger.error(f"Cannot identify pocket: no ligand '{ligand_code}' found")
        return []
    
    # Convert to numpy array for efficient distance calculation
    ligand_coords = np.array(ligand_atoms)
    
    # Find all protein atoms
    protein_atoms = []
    atom_to_residue = {}
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip non-standard residues and ligands
                if residue.get_id()[0] != ' ':
                    continue
                if residue.get_resname() in [ligand_code, 'HOH', 'WAT']:
                    continue
                
                for atom in residue:
                    coords = atom.get_coord()
                    atom_idx = len(protein_atoms)
                    protein_atoms.append(coords)
                    atom_to_residue[atom_idx] = residue
    
    protein_coords = np.array(protein_atoms)
    
    # Calculate distances from all protein atoms to all ligand atoms
    # Shape: (num_protein_atoms, num_ligand_atoms)
    distances = np.linalg.norm(
        protein_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :],
        axis=2
    )
    
    # Find atoms within radius
    close_atoms = np.where(np.min(distances, axis=1) <= radius)[0]
    
    # Collect unique residues
    pocket_residues_set = set()
    for atom_idx in close_atoms:
        if atom_idx in atom_to_residue:
            pocket_residues_set.add(atom_to_residue[atom_idx])
    
    pocket_residues = sorted(list(pocket_residues_set), key=lambda r: r.get_id())
    logger.info(f"Identified {len(pocket_residues)} pocket residues within {radius}Å of '{ligand_code}'")
    
    return pocket_residues


def extract_pocket_features(
    residues: List[Residue],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D coordinates, amino acid types, and alpha-carbon coordinates.
    
    Args:
        residues: List of Residue objects
    
    Returns:
        Tuple of:
        - ca_coords: Alpha-carbon coordinates (N_residues, 3)
        - aa_types: Amino acid one-hot encodings (N_residues, 21)
        - all_atom_coords: All atom coordinates (N_atoms, 3)
    """
    ca_coords_list = []
    aa_types_list = []
    all_atom_coords_list = []
    
    for residue in residues:
        aa_type = residue.get_resname()
        aa_idx = AMINO_ACIDS.get(aa_type, AMINO_ACIDS['UNK'])
        one_hot = np.zeros(NUM_AA_TYPES)
        one_hot[aa_idx] = 1.0
        aa_types_list.append(one_hot)
        
        # Try to get alpha carbon
        if 'CA' in residue:
            ca_atom = residue['CA']
            ca_coords = ca_atom.get_coord()
            ca_coords_list.append(ca_coords)
        else:
            # Fallback: use center of mass of residue
            try:
                center_of_mass = np.mean([a.get_coord() for a in residue], axis=0)
                ca_coords_list.append(center_of_mass)
            except Exception:
                logger.warning(f"Could not get coordinates for residue {residue}")
                continue
        
        # Collect all atom coordinates
        for atom in residue:
            all_atom_coords_list.append(atom.get_coord())
    
    ca_coords = np.array(ca_coords_list) if ca_coords_list else np.zeros((0, 3))
    aa_types = np.array(aa_types_list) if aa_types_list else np.zeros((0, NUM_AA_TYPES))
    all_atom_coords = np.array(all_atom_coords_list) if all_atom_coords_list else np.zeros((0, 3))
    
    logger.info(f"Extracted {len(ca_coords)} residues, {len(all_atom_coords)} total atoms")
    
    return ca_coords, aa_types, all_atom_coords


def compute_pocket_embedding(
    ca_coords: np.ndarray,
    aa_types: np.ndarray,
    all_atom_coords: np.ndarray,
    embedding_dim: int = 128,
) -> torch.Tensor:
    """
    Aggregate 3D pocket features into a fixed-size 128D embedding.
    
    Strategy:
    1. Compute pairwise distances between alpha carbons → distance matrix features
    2. Compute amino acid composition (flattened one-hot histogram)
    3. Compute spatial statistics (center of mass, inertia tensor moments)
    4. Concatenate and project to embedding_dim
    
    Args:
        ca_coords: Alpha-carbon coordinates (N, 3)
        aa_types: Amino acid types one-hot (N, 21)
        all_atom_coords: All atom coordinates (M, 3)
        embedding_dim: Output dimension (default 128)
    
    Returns:
        Fixed-size embedding tensor of shape (embedding_dim,)
    """
    features_list = []
    
    # === 1. Distance Matrix Features ===
    if len(ca_coords) > 1:
        # Compute pairwise CA distances
        distances = np.linalg.norm(ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :], axis=2)
        # Extract upper triangle and flatten
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        # Statistics of distances
        dist_stats = np.array([
            np.min(upper_triangle) if len(upper_triangle) > 0 else 0,
            np.max(upper_triangle) if len(upper_triangle) > 0 else 0,
            np.mean(upper_triangle) if len(upper_triangle) > 0 else 0,
            np.std(upper_triangle) if len(upper_triangle) > 0 else 0,
        ])
        features_list.append(dist_stats)
        
        # Flattened distance matrix (with max clipping to 100 residues)
        max_res = 100
        if len(ca_coords) <= max_res:
            dist_flat = upper_triangle
        else:
            # Sample evenly if too many
            n_sample = max_res * (max_res - 1) // 2
            indices = np.linspace(0, len(upper_triangle) - 1, n_sample, dtype=int)
            dist_flat = upper_triangle[indices]
        
        # Normalize and pad/truncate to fixed size
        if len(dist_flat) > 0:
            dist_flat = (dist_flat - np.mean(dist_flat) + 1e-6) / (np.std(dist_flat) + 1e-6)
        features_list.append(dist_flat)
    else:
        features_list.append(np.zeros(4))
        features_list.append(np.zeros(1))
    
    # === 2. Amino Acid Composition ===
    aa_comp = np.sum(aa_types, axis=0)  # Sum one-hot vectors → counts
    aa_comp = aa_comp / (np.sum(aa_comp) + 1e-6)  # Normalize to probabilities
    features_list.append(aa_comp)
    
    # === 3. Spatial Statistics ===
    if len(all_atom_coords) > 0:
        center_of_mass = np.mean(all_atom_coords, axis=0)
        # Translate to origin
        centered_coords = all_atom_coords - center_of_mass
        # Compute moments (mean distance, variance along axes)
        radii = np.linalg.norm(centered_coords, axis=1)
        spatial_stats = np.array([
            np.mean(radii),
            np.std(radii),
            np.max(radii),
            centered_coords[:, 0].std(),  # variance in x
            centered_coords[:, 1].std(),  # variance in y
            centered_coords[:, 2].std(),  # variance in z
        ])
        features_list.append(spatial_stats)
        features_list.append(center_of_mass)  # 3D center of mass
    else:
        features_list.append(np.zeros(6))
        features_list.append(np.zeros(3))
    
    # === 4. Concatenate and Project ===
    all_features = np.concatenate(features_list)
    
    # Project to embedding_dim using simple linear interpolation
    current_dim = len(all_features)
    
    if current_dim >= embedding_dim:
        # Truncate by taking every nth element
        indices = np.linspace(0, current_dim - 1, embedding_dim, dtype=int)
        embedding = all_features[indices]
    else:
        # Pad with zeros then replicate to reach embedding_dim
        padding = embedding_dim - current_dim
        embedding = np.concatenate([all_features, np.zeros(padding)])
    
    # Normalize to unit norm
    embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
    
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
    
    return embedding_tensor


def get_pocket_embedding(
    pdb_file: str,
    ligand_code: str = 'IRE',
    pocket_radius: float = 7.0,
    embedding_dim: int = 128,
) -> torch.Tensor:
    """
    Main function: Extract 3D pocket embedding from PDB file.
    
    Args:
        pdb_file: Path to PDB file
        ligand_code: 3-letter ligand code (default 'IRE' for Gefitinib)
        pocket_radius: Residue selection radius in Ångströms (default 7.0)
        embedding_dim: Output embedding dimension (default 128)
    
    Returns:
        Fixed-size embedding tensor of shape (embedding_dim,)
        Default is (128,)
    
    Raises:
        FileNotFoundError: If PDB file does not exist
        ValueError: If no pocket residues found
    """
    pdb_path = Path(pdb_file)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    logger.info(f"Loading PDB: {pdb_file}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_path))
    
    # Extract pocket residues
    pocket_residues = get_pocket_residues(
        structure,
        ligand_code=ligand_code,
        radius=pocket_radius,
    )
    
    if not pocket_residues:
        raise ValueError(f"No pocket residues found for ligand '{ligand_code}' within {pocket_radius}Å")
    
    # Extract features
    ca_coords, aa_types, all_atom_coords = extract_pocket_features(pocket_residues)
    
    # Compute embedding
    embedding = compute_pocket_embedding(ca_coords, aa_types, all_atom_coords, embedding_dim)
    
    logger.info(f"Generated pocket embedding: {embedding.shape}")
    
    return embedding


if __name__ == "__main__":
    """
    Test with mock data or example PDB file.
    """
    import tempfile
    
    # Create a minimal mock PDB structure for testing
    mock_pdb_content = """HEADER    TRANSFERASE                     10-MAY-21   6LTT              
ATOM      1  N   ALA A   1      20.154  29.699   5.276  1.00 39.89           N
ATOM      2  CA  ALA A   1      20.580  29.276   3.796  1.00 39.89           C
ATOM      3  C   ALA A   1      21.250  27.915   3.750  1.00 39.89           C
ATOM      4  O   ALA A   1      20.604  26.849   3.918  1.00 39.89           O
ATOM      5  CB  ALA A   1      19.376  29.210   2.845  1.00 39.89           C
ATOM      6  N   ARG A   2      22.472  27.994   3.559  1.00 37.33           N
ATOM      7  CA  ARG A   2      23.289  26.787   3.414  1.00 37.33           C
ATOM      8  C   ARG A   2      23.027  26.059   2.062  1.00 37.33           C
ATOM      9  O   ARG A   2      23.942  25.526   1.414  1.00 37.33           O
ATOM     10  CB  ARG A   2      24.769  27.157   3.453  1.00 37.33           C
ATOM     11  N   GLU A   3      21.799  26.013   1.592  1.00 34.33           N
ATOM     12  CA  GLU A   3      21.349  25.388   0.388  1.00 34.33           C
ATOM     13  C   GLU A   3      21.998  24.029   0.179  1.00 34.33           C
ATOM     14  O   GLU A   3      21.646  23.387  -0.816  1.00 34.33           O
ATOM     15  CB  GLU A   3      19.838  25.206   0.483  1.00 34.33           C
HETATM   16  C1  IRE A 201      20.500  25.000   2.500  1.00 50.00           C
HETATM   17  C2  IRE A 201      20.600  24.900   2.400  1.00 50.00           C
HETATM   18  C3  IRE A 201      20.700  26.000   3.500  1.00 50.00           C
HETATM   19  C4  IRE A 201      21.000  25.500   2.000  1.00 50.00           C
HETATM   20  C5  IRE A 201      21.200  23.800   2.800  1.00 50.00           C
END
"""
    
    # Write mock PDB to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(mock_pdb_content)
        temp_pdb = f.name
    
    try:
        logger.info("\n" + "="*80)
        logger.info("POCKET EXTRACTION TEST")
        logger.info("="*80)
        
        # Test extraction
        embedding = get_pocket_embedding(temp_pdb, ligand_code='IRE', pocket_radius=7.0)
        
        logger.info(f"\n✓ Pocket embedding generated successfully")
        logger.info(f"  Shape: {embedding.shape}")
        logger.info(f"  Dtype: {embedding.dtype}")
        logger.info(f"  L2 norm: {torch.norm(embedding, p=2):.4f}")
        logger.info(f"  Sample values: {embedding[:5].tolist()}")
        
        # Verify output
        assert embedding.shape == (128,), f"Expected shape (128,), got {embedding.shape}"
        assert embedding.dtype == torch.float32, f"Expected dtype float32, got {embedding.dtype}"
        logger.info("\n✅ All tests passed!")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        raise
    
    finally:
        # Cleanup
        import os
        os.unlink(temp_pdb)
