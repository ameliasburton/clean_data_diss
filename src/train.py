"""
Training script for the EGFR graph regressor.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model_arch import EGFR_GNN_Regressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_graphs(graph_path: str) -> List[Data]:
    load_kwargs = {"map_location": "cpu"}
    try:
        load_kwargs["weights_only"] = False
        return torch.load(graph_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        return torch.load(graph_path, **load_kwargs)


def split_graphs(graphs: Sequence[Data], val_split: float = 0.2, seed: int = 42) -> Tuple[List[Data], List[Data]]:
    if not graphs:
        raise ValueError("No graphs were loaded from the processed dataset")
    if len(graphs) == 1:
        return list(graphs), []

    rng = random.Random(seed)
    indices = list(range(len(graphs)))
    rng.shuffle(indices)

    val_count = max(1, int(round(len(graphs) * val_split)))
    val_count = min(val_count, len(graphs) - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_graphs = [graphs[idx] for idx in train_indices]
    val_graphs = [graphs[idx] for idx in val_indices]
    return train_graphs, val_graphs


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                edge_attr=batch.edge_attr,
                global_features=batch.global_features,
            )
            targets = batch.y.view(-1).to(predictions.dtype)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


def train_model(
    train_graphs: Sequence[Data],
    val_graphs: Sequence[Data],
    hidden_dim: int = 128,
    epochs: int = 75,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    out_path: str = "models/best_gnn_regressor.pth",
    log_dir: str = "logs",
) -> EGFR_GNN_Regressor:
    device_obj = torch.device(device)
    model = EGFR_GNN_Regressor(hidden_dim=hidden_dim).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(list(train_graphs), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(val_graphs), batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_graphs = 0

            for batch in train_loader:
                batch = batch.to(device_obj)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    edge_attr=batch.edge_attr,
                    global_features=batch.global_features,
                )
                targets = batch.y.view(-1).to(predictions.dtype)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch.num_graphs
                total_graphs += batch.num_graphs

            train_loss = total_loss / max(total_graphs, 1)
            val_loss = evaluate_loss(model, val_loader, criterion, device_obj) if len(val_graphs) > 0 else float("nan")

            writer.add_scalar("loss/train", train_loss, epoch)
            if len(val_graphs) > 0:
                writer.add_scalar("loss/val", val_loss, epoch)

            logger.info("Epoch %d/%d - train_loss=%.4f - val_loss=%.4f", epoch, epochs, train_loss, val_loss)

            if len(val_graphs) > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "hidden_dim": hidden_dim,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    out_file,
                )

        if len(val_graphs) == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "epoch": epochs,
                    "val_loss": None,
                },
                out_file,
            )
    finally:
        writer.flush()
        writer.close()

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/processed/graph_data.pt")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="models/best_gnn_regressor.pth")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    graphs = load_graphs(args.train_file)
    train_graphs, val_graphs = split_graphs(graphs, val_split=args.val_split, seed=args.seed)

    logger.info("Loaded %d graphs (%d train / %d val)", len(graphs), len(train_graphs), len(val_graphs))
    train_model(
        train_graphs,
        val_graphs,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        out_path=args.out,
        log_dir=args.log_dir,
    )
    logger.info("Best model saved to %s", args.out)


if __name__ == "__main__":
    main()
