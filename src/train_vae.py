#!/usr/bin/env python3
"""Two-phase training loop for GraphVAE.

Phase 1 pre-trains on a curated subset of ZINC 250k for broad chemical coverage.
Phase 2 fine-tunes on EGFR graphs with T790M pocket conditioning.

The script uses KL annealing in each phase, logs to TensorBoard, and saves the
best fine-tuned checkpoint to models/best_graphvae.pth.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import requests
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from data_prep import smiles_to_pyg_data
from generative_model import GraphVAE
from pocket_extraction import get_pocket_embedding

logger = logging.getLogger(__name__)


ATOM_CLASS_MAP = {
    6: 0,   # C
    7: 1,   # N
    8: 2,   # O
    16: 3,  # S
    9: 4,   # F
    17: 5,  # Cl
}
IGNORE_INDEX = -100


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_graphs_from_pt(graph_path: str, max_samples: Optional[int] = None) -> List[Data]:
    logger.info("Loading graphs from %s", graph_path)
    graphs = torch.load(graph_path, map_location="cpu")
    if max_samples is not None:
        graphs = graphs[:max_samples]
    logger.info("Loaded %d graphs", len(graphs))
    return graphs


def load_zinc_smiles(csv_path: str, max_samples: int) -> List[str]:
    logger.info("Loading ZINC SMILES from %s", csv_path)
    df = pd.read_csv(csv_path)
    smiles_column = None
    for candidate in ("smiles", "Smiles", "SMILES"):
        if candidate in df.columns:
            smiles_column = candidate
            break
    if smiles_column is None:
        raise ValueError("ZINC CSV must contain a smiles column")

    smiles = df[smiles_column].dropna().astype(str).tolist()
    if max_samples > 0:
        smiles = smiles[:max_samples]
    logger.info("Loaded %d SMILES for ZINC pre-training", len(smiles))
    return smiles


def build_zinc_graphs(smiles_list: Sequence[str], max_nodes: int) -> List[Data]:
    graphs: List[Data] = []
    skipped = 0

    for smiles in tqdm(smiles_list, desc="ZINC -> graphs", leave=False):
        data = smiles_to_pyg_data(smiles, label=0.0)
        if data is None:
            skipped += 1
            continue

        if data.x.size(0) > max_nodes:
            skipped += 1
            continue

        graphs.append(data)

    logger.info("Built %d ZINC graphs (%d skipped)", len(graphs), skipped)
    return graphs


def split_graphs(graphs: Sequence[Data], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Data], List[Data]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(graphs), generator=generator).tolist()
    split_point = int(len(indices) * train_ratio)
    train_graphs = [graphs[index] for index in indices[:split_point]]
    val_graphs = [graphs[index] for index in indices[split_point:]]
    logger.info("Split %d graphs: %d train / %d val", len(graphs), len(train_graphs), len(val_graphs))
    return train_graphs, val_graphs


def atom_to_class(atomic_num: int) -> int:
    return ATOM_CLASS_MAP.get(int(atomic_num), 0)


def build_reconstruction_targets(
    batch: Data,
    max_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_graphs = batch.num_graphs
    node_targets = torch.full((num_graphs, max_nodes), IGNORE_INDEX, dtype=torch.long, device=device)
    edge_adj_targets = torch.zeros((num_graphs, max_nodes, max_nodes), dtype=torch.float32, device=device)
    edge_type_targets = torch.full((num_graphs, max_nodes, max_nodes), IGNORE_INDEX, dtype=torch.long, device=device)

    ptr = batch.ptr
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr

    for graph_idx in range(num_graphs):
        start = int(ptr[graph_idx].item())
        end = int(ptr[graph_idx + 1].item())
        local_nodes = min(end - start, max_nodes)
        if local_nodes <= 0:
            continue

        node_slice = batch.x[start : start + local_nodes]
        atomic_numbers = node_slice[:, 0].round().to(torch.long)
        node_targets[graph_idx, :local_nodes] = torch.tensor(
            [atom_to_class(int(number.item())) for number in atomic_numbers],
            dtype=torch.long,
            device=device,
        )

        edge_mask = (
            (edge_index[0] >= start)
            & (edge_index[0] < end)
            & (edge_index[1] >= start)
            & (edge_index[1] < end)
        )
        if edge_mask.any():
            local_src = edge_index[0, edge_mask] - start
            local_dst = edge_index[1, edge_mask] - start
            valid_edges = (local_src < max_nodes) & (local_dst < max_nodes)
            local_src = local_src[valid_edges]
            local_dst = local_dst[valid_edges]
            local_edge_attr = edge_attr[edge_mask][valid_edges]

            edge_adj_targets[graph_idx, local_src, local_dst] = 1.0
            edge_types = local_edge_attr.argmax(dim=-1).to(torch.long)
            edge_type_targets[graph_idx, local_src, local_dst] = edge_types

    return node_targets, edge_adj_targets, edge_type_targets


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def compute_reconstruction_loss(
    node_logits: torch.Tensor,
    edge_adj_logits: torch.Tensor,
    edge_type_logits: torch.Tensor,
    batch: Data,
    max_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_targets, edge_adj_targets, edge_type_targets = build_reconstruction_targets(batch, max_nodes, device)

    node_loss = F.cross_entropy(
        node_logits.reshape(-1, node_logits.size(-1)),
        node_targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    edge_adj_loss = F.binary_cross_entropy_with_logits(edge_adj_logits, edge_adj_targets)
    edge_type_loss = F.cross_entropy(
        edge_type_logits.reshape(-1, edge_type_logits.size(-1)),
        edge_type_targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )

    recon_loss = node_loss + edge_adj_loss + edge_type_loss
    return recon_loss, node_loss, edge_adj_loss, edge_type_loss


def kl_beta(epoch: int, phase_epochs: int) -> float:
    halfway = max(1, phase_epochs // 2)
    if epoch <= 1:
        return 0.0
    if halfway == 1:
        return 1.0
    progress = min(epoch - 1, halfway - 1) / float(halfway - 1)
    return float(max(0.0, min(1.0, progress)))


def repeat_pocket_embedding(
    pocket_embedding: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if pocket_embedding is None:
        return None
    if pocket_embedding.dim() == 1:
        pocket_embedding = pocket_embedding.unsqueeze(0)
    if pocket_embedding.size(0) == 1 and batch_size > 1:
        pocket_embedding = pocket_embedding.repeat(batch_size, 1)
    return pocket_embedding.to(device)


def zero_pocket_embedding(batch_size: int, pocket_dim: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((batch_size, pocket_dim), device=device)


def safe_backward_step(loss: torch.Tensor, optimizer: Adam, model: GraphVAE) -> bool:
    try:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return True
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise

        logger.warning("OOM encountered; clearing cache and skipping batch")
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return False


def run_epoch(
    model: GraphVAE,
    loader: DataLoader,
    optimizer: Optional[Adam],
    beta: float,
    device: torch.device,
    max_nodes: int,
    pocket_embedding: Optional[torch.Tensor],
    pocket_dim: int,
    train: bool,
) -> Tuple[float, float, float]:
    model.train(train)

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    batch_count = 0

    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in iterator:
        batch = batch.to(device)

        global_features = batch.global_features
        if global_features.dim() == 1:
            global_features = global_features.view(batch.num_graphs, -1)

        if pocket_embedding is None:
            pocket_batch = zero_pocket_embedding(batch.num_graphs, pocket_dim, device)
        else:
            pocket_batch = repeat_pocket_embedding(pocket_embedding, batch.num_graphs, device)

        with torch.set_grad_enabled(train):
            node_logits, edge_adj_logits, edge_type_logits, mu, logvar = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                global_features=global_features,
                pocket_embedding=pocket_batch,
            )

            recon_loss, _, _, _ = compute_reconstruction_loss(
                node_logits=node_logits,
                edge_adj_logits=edge_adj_logits,
                edge_type_logits=edge_type_logits,
                batch=batch,
                max_nodes=max_nodes,
                device=device,
            )
            kl_loss = compute_kl_divergence(mu, logvar)
            total_loss = recon_loss + beta * kl_loss

        if train:
            step_ok = safe_backward_step(total_loss, optimizer, model) if optimizer is not None else False
            if not step_ok:
                continue

        batch_count += 1
        total_loss_sum += float(total_loss.item())
        recon_loss_sum += float(recon_loss.item())
        kl_loss_sum += float(kl_loss.item())
        iterator.set_postfix(
            total=f"{total_loss.item():.4f}",
            recon=f"{recon_loss.item():.4f}",
            kl=f"{kl_loss.item():.4f}",
        )

    if batch_count == 0:
        return float("nan"), float("nan"), float("nan")

    return (
        total_loss_sum / batch_count,
        recon_loss_sum / batch_count,
        kl_loss_sum / batch_count,
    )


def maybe_log(writer: SummaryWriter, phase: str, epoch: int, total: float, recon: float, kl: float, beta: float) -> None:
    writer.add_scalar(f"{phase}/Total_Loss", total, epoch)
    writer.add_scalar(f"{phase}/Recon_Loss", recon, epoch)
    writer.add_scalar(f"{phase}/KL_Loss", kl, epoch)
    writer.add_scalar(f"{phase}/Beta", beta, epoch)


def save_checkpoint(path: Path, model: GraphVAE, optimizer: Adam, epoch: int, val_loss: float, args: argparse.Namespace) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        },
        path,
    )


def ensure_t790m_pdb(pdb_path: Path) -> Path:
    """Download the T790M structure if it is not already available locally."""
    if pdb_path.exists():
        return pdb_path

    ensure_dir(pdb_path.parent)
    url = "https://files.rcsb.org/download/3W2S.pdb"
    logger.info("Downloading T790M pocket structure from %s", url)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(pdb_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                handle.write(chunk)

    logger.info("Saved T790M PDB to %s", pdb_path)
    return pdb_path


def load_t790m_pocket_embedding(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    candidates: List[Tuple[Path, str]] = []

    if args.pdb_file:
        candidates.append((Path(args.pdb_file), args.ligand_code))

    candidates.extend(
        [
            (Path("data/raw/pdb/3W2S.pdb"), "W2R"),
            (Path("data/raw/pdb/6LUD.pdb"), "IRE"),
            (Path("data/raw/pdb/3IKA.pdb"), "IRE"),
            (Path("data/raw/pdb/1M17.pdb"), "IRE"),
        ]
    )

    tried = set()
    for pdb_path, ligand_code in candidates:
        key = (str(pdb_path.resolve()) if pdb_path.exists() else str(pdb_path), ligand_code)
        if key in tried:
            continue
        tried.add(key)

        if pdb_path.name == "3W2S.pdb":
            try:
                pdb_path = ensure_t790m_pdb(pdb_path)
            except Exception as error:
                logger.warning("Could not download 3W2S.pdb: %s", error)

        if not pdb_path.exists():
            continue

        try:
            embedding = get_pocket_embedding(str(pdb_path), ligand_code=ligand_code, pocket_radius=7.0, embedding_dim=128)
            logger.info("Loaded T790M pocket embedding from %s (%s)", pdb_path, ligand_code)
            return embedding.to(device)
        except Exception as error:
            logger.warning("Pocket extraction failed for %s (%s): %s", pdb_path, ligand_code, error)

    logger.warning("Falling back to a zero pocket embedding for fine-tuning")
    return torch.zeros(128, device=device)


def train_phase(
    phase_name: str,
    model: GraphVAE,
    optimizer: Adam,
    train_loader: DataLoader,
    val_loader: DataLoader,
    writer: SummaryWriter,
    device: torch.device,
    phase_epochs: int,
    max_nodes: int,
    pocket_embedding: Optional[torch.Tensor],
    pocket_dim: int,
    checkpoint_path: Optional[Path],
    args: argparse.Namespace,
) -> float:
    best_val_loss = float("inf")

    for epoch in range(1, phase_epochs + 1):
        beta = kl_beta(epoch, phase_epochs)

        train_total, train_recon, train_kl = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            beta=beta,
            device=device,
            max_nodes=max_nodes,
            pocket_embedding=pocket_embedding,
            pocket_dim=pocket_dim,
            train=True,
        )

        with torch.no_grad():
            val_total, val_recon, val_kl = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                beta=beta,
                device=device,
                max_nodes=max_nodes,
                pocket_embedding=pocket_embedding,
                pocket_dim=pocket_dim,
                train=False,
            )

        maybe_log(writer, f"{phase_name}/Train", epoch, train_total, train_recon, train_kl, beta)
        maybe_log(writer, f"{phase_name}/Val", epoch, val_total, val_recon, val_kl, beta)

        logger.info(
            "%s Epoch %3d/%d - Train: %.4f (recon=%.4f, kl=%.4f, beta=%.3f) | Val: %.4f (recon=%.4f, kl=%.4f)",
            phase_name,
            epoch,
            phase_epochs,
            train_total,
            train_recon,
            train_kl,
            beta,
            val_total,
            val_recon,
            val_kl,
        )

        if checkpoint_path is not None and val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(checkpoint_path, model, optimizer, epoch, val_total, args)
            logger.info("%s checkpoint improved and saved to %s", phase_name, checkpoint_path)

    return best_val_loss


def build_loader(graphs: Sequence[Data], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(list(graphs), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main(args: argparse.Namespace) -> None:
    configure_logging()
    device = select_device()
    logger.info("Using device: %s", device)

    project_root = Path(args.project_root).resolve()
    output_path = (project_root / args.output).resolve()
    log_dir = (project_root / args.log_dir).resolve()
    ensure_dir(output_path.parent)
    ensure_dir(log_dir)

    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Phase 1: ZINC pre-training
    # ------------------------------------------------------------------
    logger.info("\n%s", "=" * 80)
    logger.info("PHASE 1: ZINC 250k pre-training")
    logger.info("%s", "=" * 80)

    zinc_smiles = load_zinc_smiles(args.zinc_file, args.zinc_samples)
    zinc_graphs = build_zinc_graphs(zinc_smiles, max_nodes=args.max_nodes)
    zinc_train_graphs, zinc_val_graphs = split_graphs(zinc_graphs, train_ratio=0.8, seed=args.seed)

    zinc_train_loader = build_loader(zinc_train_graphs, batch_size=args.batch_size, shuffle=True)
    zinc_val_loader = build_loader(zinc_val_graphs, batch_size=args.batch_size, shuffle=False)

    model = GraphVAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_nodes=args.max_nodes,
        dropout=args.dropout,
        use_pocket_conditioning=True,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    logger.info("Model parameters: %s", f"{sum(parameter.numel() for parameter in model.parameters()):,}")
    logger.info("ZINC train batches: %d", len(zinc_train_loader))
    logger.info("ZINC val batches: %d", len(zinc_val_loader))

    writer = SummaryWriter(log_dir=str(log_dir))
    zinc_zero_pocket = torch.zeros(128, device=device)

    _ = train_phase(
        phase_name="ZINC",
        model=model,
        optimizer=optimizer,
        train_loader=zinc_train_loader,
        val_loader=zinc_val_loader,
        writer=writer,
        device=device,
        phase_epochs=args.pretrain_epochs,
        max_nodes=args.max_nodes,
        pocket_embedding=zinc_zero_pocket,
        pocket_dim=128,
        checkpoint_path=project_root / "models" / "zinc_pretrained_graphvae.pth",
        args=args,
    )

    # ------------------------------------------------------------------
    # Phase 2: EGFR + T790M fine-tuning
    # ------------------------------------------------------------------
    logger.info("\n%s", "=" * 80)
    logger.info("PHASE 2: EGFR + T790M fine-tuning")
    logger.info("%s", "=" * 80)

    egfr_graphs = load_graphs_from_pt(args.egfr_graph_file, max_samples=args.egfr_samples)
    egfr_train_graphs, egfr_val_graphs = split_graphs(egfr_graphs, train_ratio=0.8, seed=args.seed)
    egfr_train_loader = build_loader(egfr_train_graphs, batch_size=args.batch_size, shuffle=True)
    egfr_val_loader = build_loader(egfr_val_graphs, batch_size=args.batch_size, shuffle=False)

    pocket_embedding = load_t790m_pocket_embedding(args, device)
    logger.info("Pocket embedding shape: %s", tuple(pocket_embedding.shape))

    best_val_loss = train_phase(
        phase_name="EGFR",
        model=model,
        optimizer=optimizer,
        train_loader=egfr_train_loader,
        val_loader=egfr_val_loader,
        writer=writer,
        device=device,
        phase_epochs=args.finetune_epochs,
        max_nodes=args.max_nodes,
        pocket_embedding=pocket_embedding,
        pocket_dim=128,
        checkpoint_path=output_path,
        args=args,
    )

    if not output_path.exists():
        save_checkpoint(output_path, model, optimizer, args.finetune_epochs, best_val_loss, args)

    writer.close()
    logger.info("\n%s", "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("%s", "=" * 80)
    logger.info("Final fine-tuned checkpoint: %s", output_path)
    logger.info("TensorBoard logdir: %s", log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GraphVAE in two phases: ZINC pretrain then EGFR fine-tune.")

    parser.add_argument("--project-root", default=".", help="Project root used to resolve relative paths")
    parser.add_argument("--zinc-file", default="data/raw/zinc250k.csv", help="Path to ZINC 250k CSV")
    parser.add_argument("--egfr-graph-file", default="data/processed/graph_data.pt", help="Path to EGFR PyG graphs")
    parser.add_argument("--zinc-samples", type=int, default=50000, help="Maximum ZINC SMILES to use")
    parser.add_argument("--egfr-samples", type=int, default=None, help="Optional EGFR graph limit")
    parser.add_argument("--pretrain-epochs", type=int, default=25, help="ZINC pre-training epochs")
    parser.add_argument("--finetune-epochs", type=int, default=50, help="EGFR fine-tuning epochs")

    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--max-nodes", type=int, default=50, help="Maximum graph nodes")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--pdb-file", default=None, help="Optional PDB file for pocket extraction")
    parser.add_argument("--ligand-code", default="W2R", help="Ligand code used for pocket extraction")

    parser.add_argument("--output", default="models/best_graphvae.pth", help="Final fine-tuned checkpoint path")
    parser.add_argument("--log-dir", default="logs_vae", help="TensorBoard log directory")

    main(parser.parse_args())
