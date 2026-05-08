"""
Pocket extraction utilities (extracted from pocket_extraction.ipynb)
"""
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem


def extract_pocket_embedding(pdb_path: str, ligand_resname: str = 'LIG', pocket_radius: float = 7.0, embedding_dim: int = 128) -> np.ndarray:
    """
    Extracts the 3D coordinates of protein atoms within a specified radius of a ligand,
    and converts them into a fixed-size embedding vector.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_path)
    
    # 1. Collect all ligand atoms and all protein atoms
    ligand_atoms = []
    protein_atoms = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if it's the target ligand
                if residue.get_resname() == ligand_resname:
                    ligand_atoms.extend(residue.get_atoms())
                # Otherwise, it's part of the protein
                elif residue.id[0] == ' ':  # ' ' means standard amino acid
                    protein_atoms.extend(residue.get_atoms())
                    
    if not ligand_atoms:
        print(f"Warning: Ligand {ligand_resname} not found in {pdb_path}. Returning zero vector.")
        return np.zeros(embedding_dim)

    # 2. Find protein atoms within pocket_radius of ANY ligand atom
    pocket_coords = []
    for p_atom in protein_atoms:
        for l_atom in ligand_atoms:
            distance = p_atom - l_atom  # BioPython overloads the minus operator to calculate distance
            if distance <= pocket_radius:
                pocket_coords.append(p_atom.get_coord())
                break  # Move to the next protein atom once it's confirmed in the pocket

    coords = np.array(pocket_coords)
    
    if coords.size == 0:
        return np.zeros(embedding_dim)
        
    # 3. Calculate simple spatial moments (centroid, standard deviation, max spread)
    centroid = coords.mean(axis=0)
    std_dev = coords.std(axis=0)
    max_spread = coords.max(axis=0) - coords.min(axis=0)
    
    feat = np.concatenate([centroid, std_dev, max_spread])
    
    # Pad to the required embedding dimension (128)
    feat = np.pad(feat, (0, max(0, embedding_dim - feat.size)), mode='constant')[:embedding_dim]
    return feat


def get_pocket_embedding(pdb_path: str, ligand_code: str = 'LIG', pocket_radius: float = 7.0, embedding_dim: int = 128) -> np.ndarray:
    return extract_pocket_embedding(pdb_path, ligand_resname=ligand_code, pocket_radius=pocket_radius, embedding_dim=embedding_dim)


if __name__ == "__main__":
    print("pocket_extractor utilities available")
