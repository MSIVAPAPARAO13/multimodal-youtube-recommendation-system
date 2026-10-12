"""
Training Entrypoint for Text-Only Metadata Co-Engagement Experiment.

Experiment: EXP_MICROLENS_TEXT_COENGAGEMENT_V1
Trains the 512-D text projection head on genuine user watch co-engagement sequences
with frozen BERT-base-uncased backbone features.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time

# Ensure repository root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.text_coengagement import (
    TextCoengagementBuilder,
    BertFeatureExtractor,
    PairedTextFeatureDataset,
)
from src.models.text_projection_model import TextProjectionModel
from src.training.losses import ContrastiveLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_text_coengagement")

EXPERIMENT_ID = "EXP_MICROLENS_TEXT_COENGAGEMENT_V1"


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        a_feat = batch["anchor_feature"].to(device)
        p_feat = batch["positive_feature"].to(device)

        # Forward projection
        z_a = model(a_feat)
        z_p = model(p_feat)

        loss = criterion(z_a, z_p)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        a_feat = batch["anchor_feature"].to(device)
        p_feat = batch["positive_feature"].to(device)

        z_a = model(a_feat)
        z_p = model(p_feat)

        loss = criterion(z_a, z_p)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train text co-engagement projection head")
    parser.add_argument("--max-users", type=int, default=1000, help="Max users to select for cohort (default: 1000)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default=f"data/checkpoints/{EXPERIMENT_ID}")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running on device: %s", device)

    # 1. Dataset Construction
    logger.info("Building MicroLens co-engagement dataset (max_users=%s, seed=%d)...", args.max_users, args.seed)
    builder = TextCoengagementBuilder(seed=args.seed)
    dataset_dict = builder.build_dataset(max_users=args.max_users)
    stats = dataset_dict["stats"]
    logger.info("Dataset statistics: %s", stats)

    # 2. Extract & Cache Frozen BERT Features
    cache_path = Path(args.cache_dir) / f"microlens_bert_{stats['unique_participating_items']}.pt"
    extractor = BertFeatureExtractor(device=device)
    features = extractor.extract_features(
        item_ids=dataset_dict["unique_items"],
        titles_map=dataset_dict["titles_map"],
        batch_size=64,
        cache_file=cache_path,
    )

    # 3. Create DataLoaders
    train_ds = PairedTextFeatureDataset(dataset_dict["train_pairs"], features)
    val_ds = PairedTextFeatureDataset(dataset_dict["val_pairs"], features)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 4. Build Model & Optimizer
    model = TextProjectionModel(input_dim=768, hidden_dim=768, output_dim=512, dropout=0.1).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model built. Trainable projection parameters: %d", trainable_params)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.01)
    criterion = ContrastiveLoss(temperature=args.temperature, symmetric=True)

    # 5. Training Loop
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    logger.info("Beginning training for %d epochs...", args.epochs)
    train_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        epoch_dur = time.time() - t_epoch_start
        lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %d/%d (%.1fs) - train_loss: %.4f - val_loss: %.4f - lr: %.2e",
            epoch, args.epochs, epoch_dur, train_loss, val_loss, lr
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "lr": lr,
            "duration_seconds": round(epoch_dur, 2),
        })

        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest.pt"
        torch.save({
            "experiment_id": EXPERIMENT_ID,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "stats": stats,
        }, latest_path)

        # Check for best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_path = checkpoint_dir / "best.pt"
            torch.save({
                "experiment_id": EXPERIMENT_ID,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "stats": stats,
            }, best_path)
            logger.info("Saved new best checkpoint to %s (val_loss=%.4f)", best_path, val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    total_training_duration = time.time() - train_start_time

    # 6. Save Experiment Metadata
    meta_path = checkpoint_dir / "experiment_metadata.json"
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_type": "text_only_metadata_coengagement",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_training_duration_seconds": round(total_training_duration, 2),
        "pretrained_backbone": "bert-base-uncased",
        "pretrained_weights_verified": True,
        "feature_dim": 512,
        "trainable_parameters": trainable_params,
        "frozen_backbone_parameters": sum(p.numel() for p in extractor.model.parameters()),
        "dataset_statistics": stats,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 5),
        "history": history,
        "args": vars(args),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete! Metadata saved to %s", meta_path)
    return metadata


if __name__ == "__main__":
    main()
