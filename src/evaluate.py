"""
Evaluation / inference utilities for EGFR GNN model.
Tests the trained model on example SMILES and batch evaluation.
"""
import argparse
import logging
import torch
from torch_geometric.data import DataLoader, Batch
from model_arch import EGFR_GNN_Regressor
from data_prep import smiles_to_pyg_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict_batch(model, graphs, device='cpu', batch_size=64):
    """Batch prediction on a list of PyG Data objects."""
    model.to(device)
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                edge_attr=batch.edge_attr,
                global_features=batch.global_features,
            )
            preds.extend(out.cpu().tolist())
    return preds


def predict_single(model, data, device='cpu'):
    """Single prediction on a PyG Data object (adds batch dimension)."""
    model.to(device)
    model.eval()
    data = data.to(device)
    # Add batch index for single sample
    batch_idx = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(
            x=data.x.unsqueeze(0) if data.x.dim() == 1 else data.x,
            edge_index=data.edge_index,
            batch=batch_idx,
            edge_attr=data.edge_attr,
            global_features=data.global_features.unsqueeze(0),
        )
    return float(out.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EGFR GNN on test SMILES")
    parser.add_argument('--model-checkpoint', default='egfr-gnn-project/models/best_gnn_regressor.pth',
                        help='Path to saved model checkpoint')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--test-file', default=None,
                        help='Optional: path to CSV file with test data (requires Smiles + pChEMBL_Value columns)')
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model_checkpoint}")
    checkpoint = torch.load(args.model_checkpoint, map_location=args.device, weights_only=True)
    model = EGFR_GNN_Regressor()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test on specific SMILES strings
    logger.info("\n" + "=" * 80)
    logger.info("SANITY CHECK: Testing on Known Molecules")
    logger.info("=" * 80)

    test_smiles = [
        ("Osimertinib (Active)", "C=CC(=O)Nc1cc(Nc2nccc(N(C)c3ccc(N(C)C)cc3OC)n2)c(OC)cc1N(C)C"),
        ("Ibuprofen (Decoy)", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ]

    predictions = []
    for label, smiles in test_smiles:
        logger.info(f"\n{label}")
        logger.info(f"SMILES: {smiles}")
        
        data = smiles_to_pyg_data(smiles, label=0.0)
        if data is None:
            logger.error("  ❌ Failed to convert SMILES")
            continue
        
        pred = predict_single(model, data, device=args.device)
        predictions.append((label, pred))
        logger.info(f"  Predicted pIC50: {pred:.4f}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    if len(predictions) == 2:
        active_label, active_pred = predictions[0]
        inactive_label, inactive_pred = predictions[1]
        logger.info(f"{active_label:30s}: {active_pred:.4f}")
        logger.info(f"{inactive_label:30s}: {inactive_pred:.4f}")
        logger.info(f"Difference (Active - Inactive): {active_pred - inactive_pred:+.4f}")
        logger.info(f"✓ Model correctly prioritizes active compounds" if active_pred > inactive_pred else "⚠ Model may need tuning")
    
    # Optional: Batch evaluation on external test set
    if args.test_file:
        logger.info(f"\n" + "=" * 80)
        logger.info(f"Evaluating on test file: {args.test_file}")
        logger.info("=" * 80)
        import pandas as pd
        df = pd.read_csv(args.test_file)
        if 'Smiles' not in df.columns:
            logger.error("Test file must contain 'Smiles' column")
        else:
            test_graphs = []
            for idx, row in df.iterrows():
                data = smiles_to_pyg_data(row['Smiles'], label=0.0)
                if data is not None:
                    test_graphs.append(data)
            
            if test_graphs:
                preds = predict_batch(model, test_graphs, device=args.device)
                for i, p in enumerate(preds[:10]):
                    logger.info(f"Sample {i}: pIC50 = {p:.4f}")
            else:
                logger.warning("No valid graphs generated from test file")
