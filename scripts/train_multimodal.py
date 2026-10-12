"""
Multimodal Contrastive Training Entrypoint.

Usage:
    python scripts/train_multimodal.py [--data-path DATA_PATH] [--epochs 10] [--batch-size 32]
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.db.config import get_settings
from src.data.training_dataset import MultimodalTrainingDataset
from src.data.pair_generator import PairGenerator, PairedMultimodalDataset
from src.models.two_tower import TwoTowerModel
from src.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_multimodal")


def main():
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Multimodal Contrastive Training")
    parser.add_argument("--data-path", type=str, default=None, help="Path to validated training JSON dataset")
    parser.add_argument("--epochs", type=int, default=settings.TRAINING_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=settings.TRAINING_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=settings.TRAINING_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=settings.TRAINING_TEMPERATURE, help="Contrastive temperature")
    parser.add_argument("--checkpoint-dir", type=str, default=settings.TRAINING_CHECKPOINT_DIR, help="Checkpoint output directory")
    parser.add_argument("--device", type=str, default=settings.DEVICE, help="Execution device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--seed", type=int, default=settings.TRAINING_SEED, help="Random seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # 1. Verify Dataset Availability
    if not args.data_path or not Path(args.data_path).exists():
        logger.error(
            "DATA AVAILABILITY GATE: No valid multimodal training dataset provided or found at '%s'. "
            "Real aligned multimodal video/audio data is required for actual training. "
            "Status: BLOCKED — TRAINING NOT VERIFIED.",
            args.data_path
        )
        return 1

    logger.info("Loading training dataset from %s...", args.data_path)
    with open(args.data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    dataset = MultimodalTrainingDataset.from_dict_list(records, feature_dim=settings.FEATURE_DIM)
    integrity = dataset.validate_integrity()
    logger.info("Dataset loaded successfully: %s", integrity)

    # 2. Build Positive Pairs
    pair_gen = PairGenerator(dataset, seed=args.seed)
    pairs = pair_gen.generate_pairs()

    if not pairs:
        logger.error(
            "DATA AVAILABILITY GATE: Dataset contains 0 valid positive pairs. "
            "Cannot train contrastive representations without positive supervision. "
            "Status: BLOCKED — TRAINING NOT VERIFIED."
        )
        return 1

    train_ds = dataset.filter_split("train")
    val_ds = dataset.filter_split("val")

    train_pairs = PairGenerator(train_ds, seed=args.seed).generate_pairs()
    val_pairs = PairGenerator(val_ds, seed=args.seed).generate_pairs()

    train_loader = DataLoader(
        PairedMultimodalDataset(train_ds, train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PairedMultimodalDataset(val_ds, val_pairs),
        batch_size=args.batch_size,
        shuffle=False,
    ) if len(val_pairs) > 0 else None

    # 3. Build Model
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(
        feature_dim=settings.FEATURE_DIM,
        share_towers=True,
        freeze_encoders=settings.TRAINING_FREEZE_BACKBONES,
        fusion_type=settings.FUSION_TYPE,
        pretrained=settings.PRETRAINED_FOUNDATION,
    ).to(device)

    # 4. Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        temperature=args.temperature,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        freeze_backbones=settings.TRAINING_FREEZE_BACKBONES,
    )

    result = trainer.fit()
    logger.info("Training complete: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
