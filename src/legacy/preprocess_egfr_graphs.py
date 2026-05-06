"""
EGFR Drug Discovery Pipeline - PyTorch Geometric Graph Generation
==================================================================

This script preprocesses bioactivity data (ChEMBL/BindingDB CSV format) 
and converts molecule SMILES strings into PyTorch Geometric Data objects
suitable for Graph Neural Network training.

Input: CSV file with 'Smiles' and 'pChEMBL_Value' columns
Output: processed_egfr_graphs.pt containing a list of torch_geometric.data.Data objects
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pathlib import Path
import logging
from typing import Optional, Tuple, List
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress RDKit warnings
warnings.filterwarnings('ignore', category=UserWarning)
Chem.SanitizeMol.__defaults__ = (True,)


# ============================================================================
# ATOMIC AND BOND FEATURE MAPPINGS
# ============================================================================

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # Atomic numbers 1-118
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, 0, 1],
    'hybridization': [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    ],
    'is_aromatic': [False, True],
}

BOND_FEATURES = {
    'bond_type': [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ],
    'is_conjugated': [False, True],
    'is_aromatic': [False, True],
}


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def get_atomic_features(atom: Chem.Atom) -> np.ndarray:
    """
    Extract categorical features for an atom and one-hot encode them.
    
    Features:
    - Atomic number (118-dim)
    - Degree (6-dim)
    - Formal charge (3-dim)
    - Hybridization (5-dim)
    - Is aromatic (2-dim)
    
    Args:
        atom: RDKit Atom object
        
    Returns:
        One-hot encoded feature vector of shape [136]
    """
    features = []
    
    # Atomic number (1-118)
    atomic_num = atom.GetAtomicNum()
    atomic_num_features = [0] * len(ATOM_FEATURES['atomic_num'])
    if atomic_num in ATOM_FEATURES['atomic_num']:
        atomic_num_features[ATOM_FEATURES['atomic_num'].index(atomic_num)] = 1
    features.extend(atomic_num_features)
    
    # Degree (0-5)
    degree = atom.GetDegree()
    degree_features = [0] * len(ATOM_FEATURES['degree'])
    if degree in ATOM_FEATURES['degree']:
        degree_features[ATOM_FEATURES['degree'].index(degree)] = 1
    else:
        # Clamp degree to max if exceeded
        degree_features[-1] = 1
    features.extend(degree_features)
    
    # Formal charge (-1, 0, 1)
    formal_charge = atom.GetFormalCharge()
    formal_charge_features = [0] * len(ATOM_FEATURES['formal_charge'])
    if formal_charge in ATOM_FEATURES['formal_charge']:
        formal_charge_features[ATOM_FEATURES['formal_charge'].index(formal_charge)] = 1
    else:
        formal_charge_features[1] = 1  # Default to neutral
    features.extend(formal_charge_features)
    
    # Hybridization
    hybridization = atom.GetHybridization()
    hybridization_features = [0] * len(ATOM_FEATURES['hybridization'])
    if hybridization in ATOM_FEATURES['hybridization']:
        hybridization_features[ATOM_FEATURES['hybridization'].index(hybridization)] = 1
    features.extend(hybridization_features)
    
    # Is aromatic
    is_aromatic = int(atom.GetIsAromatic())
    features.extend([1 - is_aromatic, is_aromatic])
    
    return np.array(features, dtype=np.float32)


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """
    Extract categorical features for a bond and one-hot encode them.
    
    Features:
    - Bond type (4-dim): SINGLE, DOUBLE, TRIPLE, AROMATIC
    - Is conjugated (2-dim)
    - Is aromatic (2-dim)
    
    Args:
        bond: RDKit Bond object
        
    Returns:
        One-hot encoded feature vector of shape [8]
    """
    features = []
    
    # Bond type
    bond_type = bond.GetBondType()
    bond_type_features = [0] * len(BOND_FEATURES['bond_type'])
    if bond_type in BOND_FEATURES['bond_type']:
        bond_type_features[BOND_FEATURES['bond_type'].index(bond_type)] = 1
    features.extend(bond_type_features)
    
    # Is conjugated
    is_conjugated = int(bond.GetIsConjugated())
    features.extend([1 - is_conjugated, is_conjugated])
    
    # Is aromatic
    is_aromatic = int(bond.GetIsAromatic())
    features.extend([1 - is_aromatic, is_aromatic])
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# MOLECULAR GRAPH CONVERSION
# ============================================================================

def smiles_to_pyg_data(
    smiles: str,
    label: float,
    mol_id: Optional[str] = None
) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string representation of molecule
        label: Target value (pChEMBL_Value / pIC50)
        mol_id: Optional molecule identifier
        
    Returns:
        torch_geometric.data.Data object or None if conversion fails
    """
    try:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None
        
        # Sanitize molecule
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.warning(f"Sanitization failed for SMILES {smiles}: {e}")
            return None
        
        # Add hydrogens for more realistic molecular representation
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        # ====== EXTRACT NODE FEATURES ======
        num_atoms = mol.GetNumAtoms()
        node_features = []
        
        for atom in mol.GetAtoms():
            atom_features = get_atomic_features(atom)
            node_features.append(atom_features)
        
        x = torch.tensor(np.array(node_features), dtype=torch.float32)
        
        # ====== EXTRACT EDGE CONNECTIVITY AND FEATURES ======
        edges = []
        edge_features = []
        
        for bond in mol.GetBonds():
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            
            # Get bond features
            bond_feat = get_bond_features(bond)
            
            # Add edge in both directions (undirected graph)
            edges.append([begin_atom_idx, end_atom_idx])
            edges.append([end_atom_idx, begin_atom_idx])
            
            # Replicate bond features for both directions
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)
        
        # Convert to tensors
        if len(edges) > 0:
            edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)
        else:
            # Handle case with no edges (single atom)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 8), dtype=torch.float32)
        
        # ====== CREATE PYTORCH GEOMETRIC DATA OBJECT ======
        y = torch.tensor([label], dtype=torch.float32)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            smiles=smiles,
            mol_id=mol_id if mol_id is not None else ""
        )
        
        return data
        
    except Exception as e:
        logger.error(f"Error processing SMILES {smiles}: {e}")
        return None


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV file and perform basic data cleaning.
    
    Args:
        csv_path: Path to input CSV file
        
    Returns:
        Cleaned pandas DataFrame
    """
    logger.info(f"Loading data from {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    # Check for required columns
    if 'Smiles' not in df.columns or 'pChEMBL_Value' not in df.columns:
        raise ValueError("CSV must contain 'Smiles' and 'pChEMBL_Value' columns")
    
    # Remove rows with missing values
    df = df.dropna(subset=['Smiles', 'pChEMBL_Value'])
    removed_missing = initial_count - len(df)
    logger.info(f"Removed {removed_missing} rows with missing values")
    
    # Remove duplicates based on SMILES
    initial_count = len(df)
    df = df.drop_duplicates(subset=['Smiles'], keep='first')
    removed_duplicates = initial_count - len(df)
    logger.info(f"Removed {removed_duplicates} duplicate SMILES strings")
    
    # Ensure pChEMBL_Value is numeric
    df['pChEMBL_Value'] = pd.to_numeric(df['pChEMBL_Value'], errors='coerce')
    df = df.dropna(subset=['pChEMBL_Value'])
    
    logger.info(f"Final dataset size: {len(df)} molecules")
    
    return df.reset_index(drop=True)


def process_dataset(
    csv_path: str,
    output_path: str = "processed_egfr_graphs.pt",
    max_molecules: Optional[int] = None
) -> Tuple[List[Data], dict]:
    """
    Process entire dataset and convert to PyG Data objects.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to save processed graphs
        max_molecules: Optional limit on number of molecules to process
        
    Returns:
        Tuple of (list of Data objects, statistics dict)
    """
    # Load and clean data
    df = load_and_clean_data(csv_path)
    
    if max_molecules is not None:
        df = df.head(max_molecules)
        logger.info(f"Processing limited to {max_molecules} molecules")
    
    # Process molecules
    graph_objects = []
    failed_count = 0
    
    logger.info(f"Processing {len(df)} molecules...")
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} molecules")
        
        smiles = row['Smiles']
        label = float(row['pChEMBL_Value'])
        mol_id = row.get('Molecule ID', None) or str(idx)
        
        # Convert to PyG Data object
        data = smiles_to_pyg_data(smiles, label, mol_id)
        
        if data is not None:
            graph_objects.append(data)
        else:
            failed_count += 1
    
    # Save to file
    logger.info(f"Saving {len(graph_objects)} graphs to {output_path}")
    torch.save(graph_objects, output_path)
    
    # Compute statistics
    stats = {
        'total_molecules': len(df),
        'successful_conversions': len(graph_objects),
        'failed_conversions': failed_count,
        'success_rate': (len(graph_objects) / len(df) * 100) if len(df) > 0 else 0,
        'output_file': output_path,
    }
    
    return graph_objects, stats


def print_statistics(graph_objects: List[Data], stats: dict):
    """
    Print dataset statistics and sample data object information.
    
    Args:
        graph_objects: List of PyG Data objects
        stats: Statistics dictionary from process_dataset
    """
    logger.info("=" * 60)
    logger.info("DATASET PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total molecules in CSV: {stats['total_molecules']}")
    logger.info(f"Successful conversions: {stats['successful_conversions']}")
    logger.info(f"Failed conversions: {stats['failed_conversions']}")
    logger.info(f"Success rate: {stats['success_rate']:.2f}%")
    logger.info(f"Output file: {stats['output_file']}")
    
    if len(graph_objects) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("SAMPLE DATA OBJECT PROPERTIES")
        logger.info("=" * 60)
        sample = graph_objects[0]
        logger.info(f"Node features shape (x): {sample.x.shape}")
        logger.info(f"Edge index shape: {sample.edge_index.shape}")
        logger.info(f"Edge attributes shape: {sample.edge_attr.shape}")
        logger.info(f"Target value (y): {sample.y.item():.4f}")
        logger.info(f"SMILES: {sample.smiles}")
        
        # Statistics across dataset
        logger.info("\n" + "=" * 60)
        logger.info("DATASET-WIDE STATISTICS")
        logger.info("=" * 60)
        
        num_nodes = [data.x.shape[0] for data in graph_objects]
        num_edges = [data.edge_index.shape[1] for data in graph_objects]
        target_values = [data.y.item() for data in graph_objects]
        
        logger.info(f"Nodes per molecule - Mean: {np.mean(num_nodes):.2f}, "
                   f"Min: {np.min(num_nodes)}, Max: {np.max(num_nodes)}")
        logger.info(f"Edges per molecule - Mean: {np.mean(num_edges):.2f}, "
                   f"Min: {np.min(num_edges)}, Max: {np.max(num_edges)}")
        logger.info(f"Target values (pChEMBL) - Mean: {np.mean(target_values):.2f}, "
                   f"Min: {np.min(target_values):.2f}, Max: {np.max(target_values):.2f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_CSV = "raw_chembl_egfr.csv"  # Change this to your actual CSV filename
    OUTPUT_FILE = "processed_egfr_graphs.pt"
    
    # Check if input file exists
    if not Path(INPUT_CSV).exists():
        logger.error(f"Input file not found: {INPUT_CSV}")
        logger.info("Please ensure the CSV file is in the current directory")
        exit(1)
    
    # Process dataset
    try:
        graph_objects, stats = process_dataset(
            csv_path=INPUT_CSV,
            output_path=OUTPUT_FILE,
            max_molecules=None  # Set to an integer to limit for testing
        )
        
        # Print statistics
        print_statistics(graph_objects, stats)
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        exit(1)
