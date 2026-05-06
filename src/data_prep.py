"""
Data preparation utilities for EGFR GNN project.
Converts SMILES -> PyG Data objects and saves processed dataset.

This version produces:
- Node features: 6 floats per atom
- Edge features: 4 floats per bond (one-hot bond type)
- Global features: 2 floats per graph (num_atoms, num_bonds)
"""
from pathlib import Path
import logging
import warnings
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore')


def get_atomic_features(atom: Chem.Atom) -> np.ndarray:
    """Return 6 node features as floats:
    [atomic_num, degree, formal_charge, hybridization_ordinal, is_aromatic, num_explicit_hs]
    """
    atomic_num = float(atom.GetAtomicNum())
    degree = float(atom.GetDegree())
    formal_charge = float(atom.GetFormalCharge())
    try:
        hyb = float(int(atom.GetHybridization()))
    except Exception:
        hyb = 0.0
    is_aromatic = float(atom.GetIsAromatic())
    num_explicit_h = float(atom.GetNumExplicitHs())
    return np.array([atomic_num, degree, formal_charge, hyb, is_aromatic, num_explicit_h], dtype=np.float32)


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """Return 4-dim one-hot bond type: [single,double,triple,aromatic]"""
    bt = bond.GetBondType()
    is_single = 1.0 if bt == Chem.BondType.SINGLE else 0.0
    is_double = 1.0 if bt == Chem.BondType.DOUBLE else 0.0
    is_triple = 1.0 if bt == Chem.BondType.TRIPLE else 0.0
    is_aromatic = 1.0 if bt == Chem.BondType.AROMATIC else 0.0
    return np.array([is_single, is_double, is_triple, is_aromatic], dtype=np.float32)


def smiles_to_pyg_data(smiles: str, label: float, mol_id: Optional[str] = None) -> Optional[Data]:
    """Convert a SMILES to a PyG Data object with specified feature sizes.

    Returns None if RDKit cannot parse or sanitize the molecule.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None
        Chem.SanitizeMol(mol)

        # Node features
        node_feats = [get_atomic_features(a) for a in mol.GetAtoms()]
        if len(node_feats) == 0:
            return None
        x = torch.tensor(np.vstack(node_feats), dtype=torch.float32)

        # Edge connectivity and features (undirected -> both directions)
        edges = []
        edge_attrs = []
        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            feat = get_bond_features(b)
            edges.append([i, j])
            edges.append([j, i])
            edge_attrs.append(feat)
            edge_attrs.append(feat)

        if len(edges) > 0:
            edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
            edge_attr = torch.tensor(np.vstack(edge_attrs), dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float32)

        # Label and global features
        y = torch.tensor([label], dtype=torch.float32)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        global_features = torch.tensor([float(num_atoms), float(num_bonds)], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            smiles=smiles,
            mol_id=mol_id or "",
            global_features=global_features,
        )
        return data

    except Exception as e:
        logger.warning(f"Failed to convert SMILES {smiles}: {e}")
        return None


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'Smiles' not in df.columns or 'pChEMBL_Value' not in df.columns:
        raise ValueError("CSV must contain 'Smiles' and 'pChEMBL_Value' columns")
    df = df.dropna(subset=['Smiles', 'pChEMBL_Value'])
    df = df.drop_duplicates(subset=['Smiles'])
    df['pChEMBL_Value'] = pd.to_numeric(df['pChEMBL_Value'], errors='coerce')
    df = df.dropna(subset=['pChEMBL_Value']).reset_index(drop=True)
    return df


def process(csv_path: str, out_path: str, max_mols: Optional[int] = None) -> Tuple[List[Data], dict]:
    df = load_and_clean(csv_path)
    if max_mols:
        df = df.head(max_mols)
    graphs: List[Data] = []
    failed = 0
    for i, row in df.iterrows():
        data = smiles_to_pyg_data(row['Smiles'], float(row['pChEMBL_Value']), mol_id=str(i))
        if data is None:
            failed += 1
        else:
            graphs.append(data)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, out_path)
    stats = {'total': len(df), 'success': len(graphs), 'failed': failed, 'out': out_path}
    return graphs, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw/raw_chembl_egfr.csv')
    parser.add_argument('--output', default='data/processed/graph_data.pt')
    parser.add_argument('--max', type=int, default=None)
    args = parser.parse_args()

    graphs, stats = process(args.input, args.output, args.max)
    logger.info(f"Processed: {stats}")
