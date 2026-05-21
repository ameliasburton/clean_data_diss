"""
Pocket extraction utilities (extracted from pocket_extraction.ipynb)
"""
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import ConvexHull

# Kyte-Doolittle hydrophobicity scale (per-residue)
_KD_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Simple residue charge approximation at physiological pH
_RES_CHARGE = {
    'ASP': -1.0, 'GLU': -1.0, 'LYS': 1.0, 'ARG': 1.0, 'HIS': 0.1
}


def extract_pocket_embedding(pdb_path: str, ligand_resname: str = 'LIG', pocket_radius: float = 7.0, embedding_dim: int = 133) -> np.ndarray:
    """
    Extracts the 3D coordinates of protein atoms within a specified radius of a ligand,
    and converts them into a fixed-size embedding vector.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_path)
    
    # 1. Collect all ligand atoms and all protein atoms
    ligand_atoms = []
    protein_atoms = []
    pocket_residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if it's the target ligand
                if residue.get_resname() == ligand_resname:
                    ligand_atoms.extend(residue.get_atoms())
                # Otherwise, it's part of the protein
                elif residue.id[0] == ' ':  # ' ' means standard amino acid
                    protein_atoms.extend(residue.get_atoms())
                    pocket_residues.append(residue)
                    
    if not ligand_atoms:
        print(f"Warning: Ligand {ligand_resname} not found in {pdb_path}. Returning zero vector.")
        return np.zeros(embedding_dim)

    # 2. Find protein atoms and residues within pocket_radius of ANY ligand atom
    pocket_coords = []
    pocket_residue_list = []
    for residue in pocket_residues:
        residue_in_pocket = False
        for p_atom in residue.get_atoms():
            for l_atom in ligand_atoms:
                distance = p_atom - l_atom
                if distance <= pocket_radius:
                    pocket_coords.append(p_atom.get_coord())
                    residue_in_pocket = True
                    break
            if residue_in_pocket:
                break
        if residue_in_pocket:
            pocket_residue_list.append(residue)

    coords = np.array(pocket_coords)
    
    if coords.size == 0:
        return np.zeros(embedding_dim)
        
    # 3. Calculate simple spatial moments (centroid, standard deviation, max spread)
    centroid = coords.mean(axis=0)
    std_dev = coords.std(axis=0)
    max_spread = coords.max(axis=0) - coords.min(axis=0)
    feat = np.concatenate([centroid, std_dev, max_spread])

    # 4. Residue-level physicochemical summaries (hydrophobicity mean/std, net charge)
    hydrophobicities = []
    net_charge = 0.0
    for res in pocket_residue_list:
        resname = res.get_resname().upper()
        # Map 3-letter to 1-letter if possible for hydrophobicity
        try:
            aa = Chem.GetResidueName(resname) if hasattr(Chem, 'GetResidueName') else None
        except Exception:
            aa = None
        # Best-effort mapping: use first letter of resname when standard
        if aa is None:
            one = resname[0] if len(resname) == 3 else resname[0]
        else:
            one = aa
        one = one.upper()
        if one in _KD_SCALE:
            hydrophobicities.append(_KD_SCALE[one])
        # Residue charge (simple lookup on 3-letter code)
        net_charge += _RES_CHARGE.get(resname, 0.0)

    if hydrophobicities:
        hyd_mean = float(np.mean(hydrophobicities))
        hyd_std = float(np.std(hydrophobicities))
    else:
        hyd_mean, hyd_std = 0.0, 0.0

    # 5. Pocket convex-hull volume and pocket depth (max distance to centroid)
    try:
        if coords.shape[0] >= 4:
            hull = ConvexHull(coords)
            hull_volume = float(hull.volume)
        else:
            hull_volume = 0.0
    except Exception:
        hull_volume = 0.0

    pocket_depth = float(np.max(np.linalg.norm(coords - centroid, axis=1))) if coords.size else 0.0

    # Append new scalar features
    extra_scalars = np.array([hyd_mean, hyd_std, net_charge, hull_volume, pocket_depth], dtype=float)
    feat = np.concatenate([feat, extra_scalars])
    
    # Pad to the required embedding dimension (embedding_dim)
    feat = np.pad(feat, (0, max(0, embedding_dim - feat.size)), mode='constant')[:embedding_dim]
    return feat


def get_pocket_embedding(pdb_path: str, ligand_code: str = 'LIG', pocket_radius: float = 7.0, embedding_dim: int = 128) -> np.ndarray:
    return extract_pocket_embedding(pdb_path, ligand_resname=ligand_code, pocket_radius=pocket_radius, embedding_dim=embedding_dim)


if __name__ == "__main__":
    print("pocket_extractor utilities available")
