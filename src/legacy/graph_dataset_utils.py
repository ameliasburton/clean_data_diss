"""
EGFR Drug Discovery Pipeline - Graph Dataset Utilities
======================================================

Utility functions for loading, inspecting, and working with processed
PyTorch Geometric graph datasets.
"""

import torch
from torch_geometric.data import Data
from pathlib import Path
import logging
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ============================================================================
# DATASET LOADING AND INSPECTION
# ============================================================================

def load_processed_graphs(filepath: str) -> List[Data]:
    """
    Load processed graph objects from disk.
    
    Args:
        filepath: Path to saved .pt file containing graph objects
        
    Returns:
        List of torch_geometric.data.Data objects
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    
    logger.info(f"Loading graphs from {filepath}")
    graph_objects = torch.load(filepath)
    logger.info(f"Loaded {len(graph_objects)} graphs")
    
    return graph_objects


def inspect_dataset(graph_objects: List[Data], num_samples: int = 5):
    """
    Print detailed inspection of dataset properties.
    
    Args:
        graph_objects: List of Data objects
        num_samples: Number of sample objects to inspect
    """
    print("\n" + "=" * 70)
    print("DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f"Total graphs: {len(graph_objects)}")
    
    # Extract statistics
    num_nodes_list = [data.x.shape[0] for data in graph_objects]
    num_edges_list = [data.edge_index.shape[1] for data in graph_objects]
    target_values = [data.y.item() for data in graph_objects]
    node_feature_dims = [data.x.shape[1] for data in graph_objects]
    edge_feature_dims = [data.edge_attr.shape[1] for data in graph_objects]
    
    # Print statistics
    print(f"\nNode count per molecule:")
    print(f"  Mean: {np.mean(num_nodes_list):.2f}")
    print(f"  Std: {np.std(num_nodes_list):.2f}")
    print(f"  Min: {np.min(num_nodes_list)}")
    print(f"  Max: {np.max(num_nodes_list)}")
    
    print(f"\nEdge count per molecule:")
    print(f"  Mean: {np.mean(num_edges_list):.2f}")
    print(f"  Std: {np.std(num_edges_list):.2f}")
    print(f"  Min: {np.min(num_edges_list)}")
    print(f"  Max: {np.max(num_edges_list)}")
    
    print(f"\nTarget values (pChEMBL_Value):")
    print(f"  Mean: {np.mean(target_values):.3f}")
    print(f"  Std: {np.std(target_values):.3f}")
    print(f"  Min: {np.min(target_values):.3f}")
    print(f"  Max: {np.max(target_values):.3f}")
    
    print(f"\nFeature dimensions:")
    print(f"  Node features: {node_feature_dims[0]} (consistent: {len(set(node_feature_dims)) == 1})")
    print(f"  Edge features: {edge_feature_dims[0]} (consistent: {len(set(edge_feature_dims)) == 1})")
    
    # Sample inspection
    print(f"\n" + "=" * 70)
    print(f"SAMPLE GRAPHS (first {min(num_samples, len(graph_objects))} molecules)")
    print("=" * 70)
    
    for i in range(min(num_samples, len(graph_objects))):
        data = graph_objects[i]
        print(f"\nSample {i + 1}:")
        print(f"  SMILES: {data.smiles}")
        print(f"  pChEMBL Value: {data.y.item():.4f}")
        print(f"  Atoms: {data.x.shape[0]}")
        print(f"  Bonds: {data.edge_index.shape[1] // 2}")  # Divide by 2 for undirected edges
        print(f"  Node features shape: {data.x.shape}")
        print(f"  Edge index shape: {data.edge_index.shape}")
        print(f"  Edge features shape: {data.edge_attr.shape}")


def split_dataset(
    graph_objects: List[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Split dataset into train/val/test sets with stratification based on target values.
    
    Args:
        graph_objects: List of Data objects
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "Train, val, test ratios must sum to 1.0"
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n = len(graph_objects)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_graphs = [graph_objects[i] for i in train_idx]
    val_graphs = [graph_objects[i] for i in val_idx]
    test_graphs = [graph_objects[i] for i in test_idx]
    
    logger.info(f"Dataset split - Train: {len(train_graphs)}, "
                f"Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    return train_graphs, val_graphs, test_graphs


def save_split_datasets(
    train_graphs: List[Data],
    val_graphs: List[Data],
    test_graphs: List[Data],
    output_dir: str = "./splits"
):
    """
    Save train/val/test splits to separate files.
    
    Args:
        train_graphs: Training set graphs
        val_graphs: Validation set graphs
        test_graphs: Test set graphs
        output_dir: Directory to save split files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    train_file = output_path / "train_graphs.pt"
    val_file = output_path / "val_graphs.pt"
    test_file = output_path / "test_graphs.pt"
    
    torch.save(train_graphs, train_file)
    torch.save(val_graphs, val_file)
    torch.save(test_graphs, test_file)
    
    logger.info(f"Saved splits to {output_dir}")
    logger.info(f"  Train: {train_file} ({len(train_graphs)} graphs)")
    logger.info(f"  Val: {val_file} ({len(val_graphs)} graphs)")
    logger.info(f"  Test: {test_file} ({len(test_graphs)} graphs)")


def visualize_target_distribution(
    graph_objects: List[Data],
    bins: int = 30,
    save_path: str = None
):
    """
    Visualize distribution of target values (pChEMBL_Value).
    
    Args:
        graph_objects: List of Data objects
        bins: Number of histogram bins
        save_path: Optional path to save figure
    """
    target_values = np.array([data.y.item() for data in graph_objects])
    
    plt.figure(figsize=(10, 6))
    plt.hist(target_values, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('pChEMBL Value (pIC50)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Target Values (n={len(graph_objects)})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Mean: {np.mean(target_values):.2f}\n" \
                f"Std: {np.std(target_values):.2f}\n" \
                f"Min: {np.min(target_values):.2f}\n" \
                f"Max: {np.max(target_values):.2f}"
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {save_path}")
    
    plt.show()


def get_dataset_statistics(graph_objects: List[Data]) -> dict:
    """
    Get comprehensive dataset statistics.
    
    Args:
        graph_objects: List of Data objects
        
    Returns:
        Dictionary with dataset statistics
    """
    num_nodes = [data.x.shape[0] for data in graph_objects]
    num_edges = [data.edge_index.shape[1] for data in graph_objects]
    target_values = [data.y.item() for data in graph_objects]
    
    stats = {
        'num_graphs': len(graph_objects),
        'node_counts': {
            'mean': float(np.mean(num_nodes)),
            'std': float(np.std(num_nodes)),
            'min': int(np.min(num_nodes)),
            'max': int(np.max(num_nodes)),
        },
        'edge_counts': {
            'mean': float(np.mean(num_edges)),
            'std': float(np.std(num_edges)),
            'min': int(np.min(num_edges)),
            'max': int(np.max(num_edges)),
        },
        'target_values': {
            'mean': float(np.mean(target_values)),
            'std': float(np.std(target_values)),
            'min': float(np.min(target_values)),
            'max': float(np.max(target_values)),
        },
        'node_feature_dim': graph_objects[0].x.shape[1] if graph_objects else 0,
        'edge_feature_dim': graph_objects[0].edge_attr.shape[1] if graph_objects else 0,
    }
    
    return stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load processed graphs
    graphs = load_processed_graphs("processed_egfr_graphs.pt")
    
    # Inspect dataset
    inspect_dataset(graphs, num_samples=5)
    
    # Get statistics
    stats = get_dataset_statistics(graphs)
    print("\n" + "=" * 70)
    print("DATASET STATISTICS (JSON format)")
    print("=" * 70)
    import json
    print(json.dumps(stats, indent=2))
    
    # Split dataset
    train, val, test = split_dataset(graphs, train_ratio=0.7, val_ratio=0.15)
    save_split_datasets(train, val, test, output_dir="./dataset_splits")
    
    # Visualize target distribution
    visualize_target_distribution(graphs, bins=30, save_path="target_distribution.png")
