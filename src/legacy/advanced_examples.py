"""
Advanced Configuration and Customization Examples
==================================================

This file provides advanced usage patterns and customization options
for the EGFR drug discovery preprocessing pipeline.
"""


# ============================================================================
# Example 1: Processing Multiple CSV Files with Different Bioactivity Sources
# ============================================================================

def process_multiple_sources():
    """
    Process bioactivity data from multiple sources (ChEMBL, BindingDB, etc.)
    and combine into a single dataset.
    """
    from preprocess_egfr_graphs import process_dataset
    import torch
    
    sources = {
        'chembl': 'raw_chembl_egfr.csv',
        'bindingdb': 'raw_bindingdb_egfr.csv',
        'pubchem': 'raw_pubchem_egfr.csv',
    }
    
    all_graphs = []
    source_stats = {}
    
    for source_name, csv_file in sources.items():
        print(f"\nProcessing {source_name}...")
        try:
            graphs, stats = process_dataset(
                csv_path=csv_file,
                output_path=f"processed_{source_name}_graphs.pt"
            )
            all_graphs.extend(graphs)
            source_stats[source_name] = stats
            print(f"  Added {len(graphs)} graphs from {source_name}")
        except FileNotFoundError:
            print(f"  Skipping {source_name}: file not found")
    
    # Save combined dataset
    combined_file = "processed_combined_egfr_graphs.pt"
    torch.save(all_graphs, combined_file)
    print(f"\nCombined dataset: {len(all_graphs)} graphs saved to {combined_file}")
    
    return all_graphs, source_stats


# ============================================================================
# Example 2: Filtering Graphs by Molecular Properties
# ============================================================================

def filter_graphs_by_properties(graphs, **filters):
    """
    Filter processed graphs based on molecular properties.
    
    Args:
        graphs: List of torch_geometric.data.Data objects
        **filters: Keyword arguments specifying filtering criteria
                   - min_atoms: Minimum number of atoms
                   - max_atoms: Maximum number of atoms
                   - min_target: Minimum target value (pChEMBL)
                   - max_target: Maximum target value (pChEMBL)
    
    Returns:
        Tuple of (filtered_graphs, filter_stats)
    """
    import numpy as np
    
    filtered_graphs = graphs.copy()
    initial_count = len(filtered_graphs)
    
    # Filter by atom count
    if 'min_atoms' in filters:
        filtered_graphs = [g for g in filtered_graphs if g.x.shape[0] >= filters['min_atoms']]
        removed = initial_count - len(filtered_graphs)
        print(f"  Removed {removed} graphs with <{filters['min_atoms']} atoms")
        initial_count = len(filtered_graphs)
    
    if 'max_atoms' in filters:
        filtered_graphs = [g for g in filtered_graphs if g.x.shape[0] <= filters['max_atoms']]
        removed = initial_count - len(filtered_graphs)
        print(f"  Removed {removed} graphs with >{filters['max_atoms']} atoms")
        initial_count = len(filtered_graphs)
    
    # Filter by target value
    if 'min_target' in filters:
        filtered_graphs = [g for g in filtered_graphs if g.y.item() >= filters['min_target']]
        removed = initial_count - len(filtered_graphs)
        print(f"  Removed {removed} graphs with pChEMBL <{filters['min_target']}")
        initial_count = len(filtered_graphs)
    
    if 'max_target' in filters:
        filtered_graphs = [g for g in filtered_graphs if g.y.item() <= filters['max_target']]
        removed = initial_count - len(filtered_graphs)
        print(f"  Removed {removed} graphs with pChEMBL >{filters['max_target']}")
    
    stats = {
        'original_count': len(graphs),
        'filtered_count': len(filtered_graphs),
        'removed': len(graphs) - len(filtered_graphs),
        'retention_rate': (len(filtered_graphs) / len(graphs) * 100) if len(graphs) > 0 else 0
    }
    
    return filtered_graphs, stats


# ============================================================================
# Example 3: Data Normalization for Model Training
# ============================================================================

def normalize_target_values(graphs, method='standardization'):
    """
    Normalize target values (pChEMBL) for improved model training.
    
    Args:
        graphs: List of torch_geometric.data.Data objects
        method: Normalization method ('standardization', 'minmax', 'robust')
    
    Returns:
        Tuple of (normalized_graphs, normalization_params)
    """
    import numpy as np
    import torch
    
    # Extract target values
    targets = np.array([g.y.item() for g in graphs])
    
    normalization_params = {
        'method': method,
        'original_mean': float(np.mean(targets)),
        'original_std': float(np.std(targets)),
        'original_min': float(np.min(targets)),
        'original_max': float(np.max(targets)),
    }
    
    if method == 'standardization':
        # Z-score normalization: (x - mean) / std
        mean = np.mean(targets)
        std = np.std(targets)
        normalized_targets = (targets - mean) / (std + 1e-6)
        normalization_params['mean'] = mean
        normalization_params['std'] = std
        
    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(targets)
        max_val = np.max(targets)
        normalized_targets = (targets - min_val) / (max_val - min_val + 1e-6)
        normalization_params['min'] = min_val
        normalization_params['max'] = max_val
        
    elif method == 'robust':
        # Robust normalization using median and IQR
        median = np.median(targets)
        q25 = np.percentile(targets, 25)
        q75 = np.percentile(targets, 75)
        iqr = q75 - q25
        normalized_targets = (targets - median) / (iqr + 1e-6)
        normalization_params['median'] = float(median)
        normalization_params['iqr'] = float(iqr)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Update graphs with normalized targets
    normalized_graphs = []
    for graph, new_target in zip(graphs, normalized_targets):
        graph.y = torch.tensor([new_target], dtype=torch.float32)
        normalized_graphs.append(graph)
    
    return normalized_graphs, normalization_params


def denormalize_predictions(predictions, normalization_params):
    """
    Denormalize model predictions back to original scale.
    
    Args:
        predictions: Model predictions (numpy array or torch tensor)
        normalization_params: Dictionary from normalize_target_values
    
    Returns:
        Denormalized predictions
    """
    import numpy as np
    
    method = normalization_params['method']
    
    if method == 'standardization':
        mean = normalization_params['mean']
        std = normalization_params['std']
        denormalized = predictions * std + mean
        
    elif method == 'minmax':
        min_val = normalization_params['min']
        max_val = normalization_params['max']
        denormalized = predictions * (max_val - min_val) + min_val
        
    elif method == 'robust':
        median = normalization_params['median']
        iqr = normalization_params['iqr']
        denormalized = predictions * iqr + median
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return denormalized


# ============================================================================
# Example 4: Custom Data Augmentation
# ============================================================================

def augment_dataset_with_smiles_variants(graphs):
    """
    Augment dataset by generating alternative SMILES representations
    (e.g., different atom ordering from canonical forms).
    
    This leverages RDKit's ability to generate multiple valid SMILES
    for the same molecule, providing implicit data augmentation.
    
    Args:
        graphs: List of torch_geometric.data.Data objects
    
    Returns:
        Augmented list of Data objects (2x original size)
    """
    from rdkit import Chem
    from preprocess_egfr_graphs import smiles_to_pyg_data
    import logging
    
    logger = logging.getLogger(__name__)
    augmented_graphs = graphs.copy()
    new_graphs_count = 0
    
    for graph in graphs:
        try:
            original_smiles = graph.smiles
            mol = Chem.MolFromSmiles(original_smiles)
            
            if mol is None:
                continue
            
            # Generate alternative SMILES (randomized atom ordering)
            alt_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=0)
            
            if alt_smiles != original_smiles:
                # Create new graph with alternative SMILES
                new_graph = smiles_to_pyg_data(alt_smiles, graph.y.item())
                
                if new_graph is not None:
                    augmented_graphs.append(new_graph)
                    new_graphs_count += 1
        
        except Exception as e:
            logger.warning(f"Augmentation failed for {graph.smiles}: {e}")
            continue
    
    logger.info(f"Dataset augmented: added {new_graphs_count} new graphs from SMILES variants")
    return augmented_graphs


# ============================================================================
# Example 5: Graph-Level Property Analysis
# ============================================================================

def analyze_graph_complexity(graphs):
    """
    Analyze and categorize graphs by their structural complexity.
    
    Args:
        graphs: List of torch_geometric.data.Data objects
    
    Returns:
        Dictionary with complexity analysis
    """
    import numpy as np
    
    num_atoms = np.array([g.x.shape[0] for g in graphs])
    num_edges = np.array([g.edge_index.shape[1] for g in graphs])
    
    # Calculate graph density (edges / max_possible_edges)
    max_edges = num_atoms * (num_atoms - 1)
    graph_density = num_edges / (max_edges + 1e-6)
    
    # Define complexity categories
    complexity = {
        'small': sum(num_atoms <= 20),
        'medium': sum((num_atoms > 20) & (num_atoms <= 50)),
        'large': sum(num_atoms > 50),
    }
    
    analysis = {
        'complexity_distribution': complexity,
        'avg_atoms': float(np.mean(num_atoms)),
        'avg_edges': float(np.mean(num_edges)),
        'avg_density': float(np.mean(graph_density)),
        'min_density': float(np.min(graph_density)),
        'max_density': float(np.max(graph_density)),
    }
    
    return analysis


# ============================================================================
# Example 6: Train/Val/Test Split with Stratification
# ============================================================================

def stratified_split_by_target_value(graphs, train_ratio=0.7, val_ratio=0.15, n_bins=5):
    """
    Split dataset using stratification by target value range.
    Ensures each split has similar target value distributions.
    
    Args:
        graphs: List of torch_geometric.data.Data objects
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        n_bins: Number of bins for stratification
    
    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Get target values
    targets = np.array([g.y.item() for g in graphs])
    
    # Create bins for stratification
    _, bins = np.histogram(targets, bins=n_bins)
    target_bins = np.digitize(targets, bins)
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        np.arange(len(graphs)),
        train_size=train_ratio,
        stratify=target_bins,
        random_state=42
    )
    
    # Second split: val vs test
    temp_targets = target_bins[temp_idx]
    val_idx_relative, test_idx_relative = train_test_split(
        np.arange(len(temp_idx)),
        train_size=val_ratio / (val_ratio + test_ratio),
        stratify=temp_targets,
        random_state=42
    )
    
    val_idx = temp_idx[val_idx_relative]
    test_idx = temp_idx[test_idx_relative]
    
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    return train_graphs, val_graphs, test_graphs


# ============================================================================
# Example 7: Creating Mini-Batches for Memory-Efficient Processing
# ============================================================================

def process_large_dataset_in_batches(csv_path, batch_size=100):
    """
    Process very large CSV files in batches to manage memory usage.
    
    Args:
        csv_path: Path to input CSV
        batch_size: Number of molecules per batch
    
    Yields:
        torch_geometric.data.Data objects from each batch
    """
    import pandas as pd
    from preprocess_egfr_graphs import load_and_clean_data, smiles_to_pyg_data
    
    df = load_and_clean_data(csv_path)
    
    for start_idx in range(0, len(df), batch_size):
        batch_df = df.iloc[start_idx:start_idx + batch_size]
        batch_graphs = []
        
        for _, row in batch_df.iterrows():
            graph = smiles_to_pyg_data(
                row['Smiles'],
                float(row['pChEMBL_Value']),
                mol_id=row.get('Molecule ID', None)
            )
            if graph is not None:
                batch_graphs.append(graph)
        
        yield batch_graphs


# ============================================================================
# Example 8: Handling Class Imbalance in Binding Affinity
# ============================================================================

def identify_binding_affinity_classes(graphs, thresholds=None):
    """
    Classify molecules into binding affinity categories (weak, moderate, strong).
    Useful for classification-based approaches or handling class imbalance.
    
    Args:
        graphs: List of torch_geometric.data.Data objects
        thresholds: Tuple of (low_threshold, high_threshold) for pChEMBL values
                    Default: (5.0, 7.0) - weak (<5), moderate (5-7), strong (>7)
    
    Returns:
        Categorized graphs and distribution
    """
    import numpy as np
    
    if thresholds is None:
        thresholds = (5.0, 7.0)
    
    low_thresh, high_thresh = thresholds
    
    weak_binders = []
    moderate_binders = []
    strong_binders = []
    
    for graph in graphs:
        pchembl = graph.y.item()
        if pchembl < low_thresh:
            weak_binders.append(graph)
        elif pchembl < high_thresh:
            moderate_binders.append(graph)
        else:
            strong_binders.append(graph)
    
    distribution = {
        'weak': len(weak_binders),
        'moderate': len(moderate_binders),
        'strong': len(strong_binders),
    }
    
    return {
        'weak': weak_binders,
        'moderate': moderate_binders,
        'strong': strong_binders,
    }, distribution


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Process multiple sources
    # all_graphs, stats = process_multiple_sources()
    
    # Example 2: Filter graphs
    # from graph_dataset_utils import load_processed_graphs
    # graphs = load_processed_graphs("processed_egfr_graphs.pt")
    # filtered_graphs, filter_stats = filter_graphs_by_properties(
    #     graphs,
    #     min_atoms=10,
    #     max_atoms=100,
    #     min_target=5.0,
    #     max_target=9.0
    # )
    
    # Example 3: Normalize and save
    # normalized_graphs, norm_params = normalize_target_values(
    #     filtered_graphs,
    #     method='standardization'
    # )
    # import torch
    # torch.save(normalized_graphs, "processed_egfr_graphs_normalized.pt")
    # import json
    # with open("normalization_params.json", "w") as f:
    #     json.dump(norm_params, f, indent=2)
    
    # Example 4: Stratified split
    # train, val, test = stratified_split_by_target_value(graphs)
    
    # Example 5: Analyze complexity
    # complexity = analyze_graph_complexity(graphs)
    # print(complexity)
    
    # Example 6: Binding affinity classes
    # categorized, dist = identify_binding_affinity_classes(graphs)
    # print(f"Binding affinity distribution: {dist}")
    
    print("Advanced examples are provided as functions in this file.")
    print("Uncomment examples in the __main__ block to run them.")
