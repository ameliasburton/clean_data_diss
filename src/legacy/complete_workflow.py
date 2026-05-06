"""
EGFR Drug Discovery Pipeline - Complete Example Workflow
=========================================================

This script demonstrates a complete end-to-end workflow:
1. Preprocess raw bioactivity data
2. Load and inspect the processed graphs
3. Split into train/val/test sets
4. Visualize the dataset
5. Prepare for GNN training
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete preprocessing and dataset preparation pipeline."""
    
    logger.info("=" * 70)
    logger.info("EGFR DRUG DISCOVERY PIPELINE - COMPLETE WORKFLOW")
    logger.info("=" * 70)
    
    # Step 1: Import required modules
    logger.info("\n[Step 1] Loading modules...")
    try:
        from preprocess_egfr_graphs import process_dataset, print_statistics
        from graph_dataset_utils import (
            load_processed_graphs, 
            inspect_dataset,
            split_dataset,
            save_split_datasets,
            get_dataset_statistics,
            visualize_target_distribution
        )
        import json
        logger.info("✓ All modules imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import modules: {e}")
        logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 2: Define configuration
    logger.info("\n[Step 2] Setting configuration...")
    config = {
        'input_csv': 'raw_chembl_egfr.csv',  # TODO: Update to your CSV filename
        'output_graphs': 'processed_egfr_graphs.pt',
        'output_dir': './dataset_splits',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'seed': 42,
        'plot_save_path': 'target_distribution.png',
    }
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Step 3: Check if input file exists
    logger.info("\n[Step 3] Checking input file...")
    input_path = Path(config['input_csv'])
    if not input_path.exists():
        logger.error(f"✗ Input file not found: {config['input_csv']}")
        logger.info("Please ensure your CSV file is in the current directory")
        logger.info("Expected columns: 'Smiles', 'pChEMBL_Value'")
        sys.exit(1)
    logger.info(f"✓ Input file found: {input_path}")
    
    # Step 4: Preprocess and convert to PyG graphs
    logger.info("\n[Step 4] Preprocessing bioactivity data...")
    logger.info("This may take several minutes depending on dataset size...")
    try:
        graphs, stats = process_dataset(
            csv_path=config['input_csv'],
            output_path=config['output_graphs'],
            max_molecules=None  # Set to integer (e.g., 1000) for testing
        )
        logger.info("✓ Preprocessing complete")
        print_statistics(graphs, stats)
    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        sys.exit(1)
    
    # Step 5: Save processing statistics
    logger.info("\n[Step 5] Saving processing statistics...")
    stats_file = "processing_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Statistics saved to {stats_file}")
    
    # Step 6: Load and inspect graphs
    logger.info("\n[Step 6] Loading and inspecting processed graphs...")
    try:
        graphs = load_processed_graphs(config['output_graphs'])
        logger.info(f"✓ Loaded {len(graphs)} graphs")
        inspect_dataset(graphs, num_samples=min(5, len(graphs)))
    except Exception as e:
        logger.error(f"✗ Failed to load graphs: {e}")
        sys.exit(1)
    
    # Step 7: Get comprehensive statistics
    logger.info("\n[Step 7] Computing comprehensive dataset statistics...")
    dataset_stats = get_dataset_statistics(graphs)
    logger.info("Dataset statistics:")
    for key, value in dataset_stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Save detailed statistics
    stats_detailed_file = "dataset_statistics.json"
    with open(stats_detailed_file, 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    logger.info(f"✓ Detailed statistics saved to {stats_detailed_file}")
    
    # Step 8: Split dataset
    logger.info("\n[Step 8] Splitting dataset into train/val/test...")
    try:
        train_graphs, val_graphs, test_graphs = split_dataset(
            graphs,
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio'],
            seed=config['seed']
        )
        logger.info(f"✓ Split complete:")
        logger.info(f"  Train: {len(train_graphs)} graphs")
        logger.info(f"  Val: {len(val_graphs)} graphs")
        logger.info(f"  Test: {len(test_graphs)} graphs")
    except Exception as e:
        logger.error(f"✗ Dataset splitting failed: {e}")
        sys.exit(1)
    
    # Step 9: Save split datasets
    logger.info("\n[Step 9] Saving dataset splits...")
    try:
        save_split_datasets(
            train_graphs,
            val_graphs,
            test_graphs,
            output_dir=config['output_dir']
        )
        logger.info("✓ Dataset splits saved")
    except Exception as e:
        logger.error(f"✗ Failed to save splits: {e}")
        sys.exit(1)
    
    # Step 10: Visualize target distribution
    logger.info("\n[Step 10] Visualizing target value distribution...")
    try:
        visualize_target_distribution(
            graphs,
            bins=30,
            save_path=config['plot_save_path']
        )
        logger.info(f"✓ Distribution plot saved to {config['plot_save_path']}")
    except Exception as e:
        logger.warning(f"⚠ Visualization failed (non-critical): {e}")
    
    # Step 11: Create a sample data object info file
    logger.info("\n[Step 11] Creating sample code for loading data...")
    sample_code = '''
# Loading the processed dataset for model training

from graph_dataset_utils import load_processed_graphs
import torch
from torch_geometric.data import DataLoader

# Load full dataset
graphs = load_processed_graphs("processed_egfr_graphs.pt")

# Or load individual splits
train_graphs = torch.load("./dataset_splits/train_graphs.pt")
val_graphs = torch.load("./dataset_splits/val_graphs.pt")
test_graphs = torch.load("./dataset_splits/test_graphs.pt")

# Create data loaders for training
batch_size = 32
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

# Example: Access a single graph
sample_graph = graphs[0]
print(f"Sample molecule SMILES: {sample_graph.smiles}")
print(f"Number of atoms: {sample_graph.x.shape[0]}")
print(f"Number of bonds: {sample_graph.edge_index.shape[1] // 2}")
print(f"Node feature dimension: {sample_graph.x.shape[1]}")
print(f"Edge feature dimension: {sample_graph.edge_attr.shape[1]}")
print(f"Target value (pChEMBL): {sample_graph.y.item():.4f}")

# Example: Train a simple GNN
# See PyTorch Geometric documentation for GNN architecture examples
'''
    
    sample_file = "sample_loading_code.py"
    with open(sample_file, 'w') as f:
        f.write(sample_code)
    logger.info(f"✓ Sample loading code saved to {sample_file}")
    
    # Step 12: Summary
    logger.info("\n" + "=" * 70)
    logger.info("WORKFLOW COMPLETE!")
    logger.info("=" * 70)
    logger.info("\nGenerated files:")
    logger.info(f"  • {config['output_graphs']} - All processed graphs")
    logger.info(f"  • {config['output_dir']}/train_graphs.pt - Training set")
    logger.info(f"  • {config['output_dir']}/val_graphs.pt - Validation set")
    logger.info(f"  • {config['output_dir']}/test_graphs.pt - Test set")
    logger.info(f"  • {stats_file} - Processing statistics")
    logger.info(f"  • {stats_detailed_file} - Dataset statistics")
    logger.info(f"  • {config['plot_save_path']} - Target distribution plot")
    logger.info(f"  • {sample_file} - Sample code for model training")
    
    logger.info("\nNext steps:")
    logger.info("  1. Review the generated statistics and plots")
    logger.info("  2. Use sample_loading_code.py as a template for your GNN model")
    logger.info("  3. Consider data normalization for target values")
    logger.info("  4. Implement your GNN architecture using PyTorch Geometric")
    
    logger.info("\nDataset info for model training:")
    logger.info(f"  • Number of training graphs: {len(train_graphs)}")
    logger.info(f"  • Number of validation graphs: {len(val_graphs)}")
    logger.info(f"  • Number of test graphs: {len(test_graphs)}")
    logger.info(f"  • Node feature dimension: {dataset_stats['node_feature_dim']}")
    logger.info(f"  • Edge feature dimension: {dataset_stats['edge_feature_dim']}")
    logger.info(f"  • Target variable range: [{dataset_stats['target_values']['min']:.2f}, "
                f"{dataset_stats['target_values']['max']:.2f}]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
