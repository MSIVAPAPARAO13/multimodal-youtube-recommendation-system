"""
Training Entrypoint for Phase 8C Text Co-Engagement Optimization.

Experiment: EXP_MICROLENS_TEXT_COENGAGEMENT_V2
Addresses Phase 8B diagnostic findings:
1. Uses linear projection (Linear(768, 512)) preserving pretrained geometry without ReLU truncation.
2. Employs validation retrieval ranking score (MRR + NDCG@10) for model selection instead of raw contrastive loss.
3. Uses optimal learning rate lr=3e-4 selected via validation-only search.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.text_coengagement import (
    TextCoengagementBuilder,
    BertFeatureExtractor,
    PairedTextFeatureDataset,
)
from src.training.losses import ContrastiveLoss
from scripts.evaluate_text_coengagement import compute_metrics
from scripts.optimize_text_coengagement import LinearProjectionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_text_coengagement_v2")

EXPERIMENT_ID = "EXP_MICROLENS_TEXT_COENGAGEMENT_V2"


def main():
    parser = argparse.ArgumentParser(description="Train text co-engagement v2 model")
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=f"data/checkpoints/{EXPERIMENT_ID}")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # 1. Dataset Construction
    logger.info("Building MicroLens dataset with seed=%d...", args.seed)
    builder = TextCoengagementBuilder(seed=args.seed)
    ds = builder.build_dataset(max_users=args.max_users)
    stats = ds["stats"]
    unique_items = ds["unique_items"]
    titles_map = ds["titles_map"]

    logger.info("Dataset statistics: %s", stats)

    # 2. Load BERT Features
    cache_path = Path(args.cache_dir) / f"microlens_bert_{len(unique_items)}.pt"
    extractor = BertFeatureExtractor(device=device)
    features = extractor.extract_features(unique_items, titles_map, cache_file=cache_path)

    # 3. DataLoaders
    train_ds = PairedTextFeatureDataset(ds["train_pairs"], features)
    val_ds = PairedTextFeatureDataset(ds["val_pairs"], features)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 4. Validation query & candidate tensors for ranking evaluation
    val_ground_truth = {}
    for p in ds["val_pairs"]:
        val_ground_truth.setdefault(p.anchor_item, set()).add(p.positive_item)

    val_query_ids = sorted(list(val_ground_truth.keys()))
    candidate_ids = list(unique_items)

    val_query_tensor = torch.stack([features[qid] for qid in val_query_ids]).to(device)
    val_cand_tensor = torch.stack([features[cid] for cid in candidate_ids]).to(device)

    # 5. Build Model (LinearProjectionModel selected in validation optimization)
    model = LinearProjectionModel(input_dim=768, output_dim=512, normalize=True).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model built: LinearProjectionModel (%d trainable parameters)", trainable_params)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(len(train_loader) * args.epochs, 1), eta_min=args.lr * 0.01)
    criterion = ContrastiveLoss(temperature=args.temperature, symmetric=True)

    # 6. Training with Retrieval-Score Validation Selection
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_score = 0.0
    best_epoch = 0
    history = []

    logger.info("Starting training for %d epochs...", args.epochs)
    train_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            a_feat = batch["anchor_feature"].to(device)
            p_feat = batch["positive_feature"].to(device)

            z_a = model(a_feat)
            z_p = model(p_feat)

            loss = criterion(z_a, z_p)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / max(num_batches, 1)

        # Validation evaluation
        model.eval()
        v_total_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                a_feat = batch["anchor_feature"].to(device)
                p_feat = batch["positive_feature"].to(device)
                z_a = model(a_feat)
                z_p = model(p_feat)
                v_total_loss += criterion(z_a, z_p).item()
                v_batches += 1

            val_loss = v_total_loss / max(v_batches, 1)
            q_emb = model(val_query_tensor)
            c_emb = model(val_cand_tensor)
            val_metrics = compute_metrics(q_emb, val_query_ids, c_emb, candidate_ids, val_ground_truth)

        dur = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        val_score = val_metrics["MRR"] + val_metrics["NDCG@10"]

        logger.info(
            "Epoch %d/%d (%.1fs) - train_loss: %.4f - val_loss: %.4f - val_MRR: %.4f - val_NDCG@10: %.4f",
            epoch, args.epochs, dur, train_loss, val_loss, val_metrics["MRR"], val_metrics["NDCG@10"]
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_metrics": val_metrics,
            "val_score": round(val_score, 5),
            "duration_seconds": round(dur, 2),
            "lr": lr_now,
        })

        # Save latest
        torch.save({
            "experiment_id": EXPERIMENT_ID,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
            "val_loss": val_loss,
        }, checkpoint_dir / "latest.pt")

        # Select best by retrieval score
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_path = checkpoint_dir / "best.pt"
            torch.save({
                "experiment_id": EXPERIMENT_ID,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "val_loss": val_loss,
            }, best_path)
            logger.info("Saved new best checkpoint to %s (val_score=%.4f)", best_path, val_score)

    total_duration = time.time() - train_start_time

    # Fingerprint best checkpoint
    best_ckpt_bytes = (checkpoint_dir / "best.pt").read_bytes()
    best_sha256 = hashlib.sha256(best_ckpt_bytes).hexdigest()

    # Save metadata
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_training_duration_seconds": round(total_duration, 2),
        "architecture": "LinearProjectionModel",
        "input_dim": 768,
        "output_dim": 512,
        "trainable_parameters": trainable_params,
        "frozen_backbone": "bert-base-uncased",
        "checkpoint_sha256": best_sha256,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "history": history,
        "dataset_statistics": stats,
        "args": vars(args),
    }

    with open(checkpoint_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training V2 completed! Metadata written to %s", checkpoint_dir / "experiment_metadata.json")
    return metadata


if __name__ == "__main__":
    main()
