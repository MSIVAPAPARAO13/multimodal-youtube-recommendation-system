"""
Multimodal Contrastive Trainer.

Coordinates training and validation of TwoTowerModel / VideoTower components:
- Trains projection heads and GatedFusion parameters while keeping backbones frozen by default
- Feeds genuine positive pairs with false-negative suppression
- Handles missing modalities via explicit modality masks
- Enforces gradient clipping, early stopping, and secure checkpointing
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.two_tower import TwoTowerModel
from src.training.checkpoint import CheckpointManager, CheckpointMetadata
from src.training.losses import ContrastiveLoss, InfoNCELoss
from src.utils.device import get_device

logger = logging.getLogger(__name__)


class Trainer:
    """
    Contrastive Trainer for Multimodal YouTube Recommendation Model.
    """

    def __init__(
        self,
        model: TwoTowerModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        num_epochs: int = 10,
        early_stopping_patience: int = 3,
        checkpoint_dir: str = "data/checkpoints",
        device: Optional[Union[str, torch.device]] = None,
        max_grad_norm: float = 1.0,
        freeze_backbones: bool = True,
    ):
        if device is None:
            self.device = get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        # Configure backbone freezing
        if freeze_backbones:
            self._freeze_backbones()

        # Optimizer: collect only parameters that require gradients (projection heads + fusion)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found in model!")

        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = max(len(train_loader) * num_epochs, 1)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.01,
        )

        self.criterion = ContrastiveLoss(temperature=temperature, symmetric=True)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            feature_dim=self.model.feature_dim,
            fusion_type=self.model.fusion_type,
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.history: List[Dict[str, Any]] = []

    def _freeze_backbones(self) -> None:
        """Ensure heavy pre-trained backbones are frozen to train only projection and fusion."""
        towers = [self.model.video_tower]
        if hasattr(self.model, "query_tower") and self.model.query_tower is not self.model.video_tower:
            towers.append(self.model.query_tower)

        for tower in towers:
            if hasattr(tower, "video_encoder") and hasattr(tower.video_encoder, "backbone"):
                for p in tower.video_encoder.backbone.parameters():
                    p.requires_grad = False
            if hasattr(tower, "audio_encoder") and hasattr(tower.audio_encoder, "backbone"):
                for p in tower.audio_encoder.backbone.parameters():
                    p.requires_grad = False
            if hasattr(tower, "text_encoder") and hasattr(tower.text_encoder, "backbone"):
                for p in tower.text_encoder.backbone.parameters():
                    p.requires_grad = False

    def train_epoch(self, epoch: int) -> float:
        """Run a single training epoch with genuine positive pairs and gradient clipping."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            loss = self._process_batch(batch, is_training=True)

            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError(f"Encountered non-finite loss: {loss.item()} at step {self.global_step}")

            self.optimizer.zero_grad()
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=self.max_grad_norm,
                )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        """Compute validation loss across validation dataset."""
        if self.val_loader is None or len(self.val_loader) == 0:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            loss = self._process_batch(batch, is_training=False)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _process_batch(self, batch: Dict[str, Any], is_training: bool) -> torch.Tensor:
        """
        Process a batch:
        Supports either:
        1. Paired input with 'anchor_*' and 'positive_*'
        2. Direct batch embeddings
        """
        if "anchor_visual" in batch and "positive_visual" in batch:
            # Paired multimodal batch
            a_v = batch["anchor_visual"].to(self.device)
            a_a = batch["anchor_audio"].to(self.device)
            a_t = batch["anchor_text"].to(self.device)

            p_v = batch["positive_visual"].to(self.device)
            p_a = batch["positive_audio"].to(self.device)
            p_t = batch["positive_text"].to(self.device)

            # Pass through towers or fusion
            # When inputs are already 512-d embeddings, pass directly to fusion
            a_emb = self.model.video_tower.fusion(a_v, a_a, a_t)
            p_emb = self.model.query_tower.fusion(p_v, p_a, p_t)

            a_emb = nn.functional.normalize(a_emb, p=2, dim=-1)
            p_emb = nn.functional.normalize(p_emb, p=2, dim=-1)

            loss = self.criterion(a_emb, p_emb)
            return loss

        elif "visual" in batch:
            # Single-stream self-contrast (loss against identical vector) is mathematically degenerate
            # and does NOT represent genuine recommendation relevance. We explicitly reject ungrounded
            # single-stream training to prevent presenting synthetic/trivial loss as trained relevance.
            raise ValueError(
                "Degenerate single-stream batch detected. Contrastive recommendation training strictly requires "
                "genuine positive pairs ('anchor_*' and 'positive_*') derived from user co-engagement or interaction data. "
                "Synthetic self-contrast and noise-augmented pairs are disabled to preserve model integrity."
            )

        else:
            raise KeyError("Unsupported batch keys for contrastive training. Expected paired 'anchor_*' and 'positive_*' inputs.")

    def fit(self) -> Dict[str, Any]:
        """
        Execute full training loop with early stopping.
        """
        start_time = time.time()
        logger.info("Starting training loop for %d epochs...", self.num_epochs)

        best_epoch = 0
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            current_lr = self.scheduler.get_last_lr()[0]
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "lr": current_lr,
            }
            self.history.append(record)

            logger.info("Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, lr=%.2e", epoch, self.num_epochs, train_loss, val_loss, current_lr)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                best_epoch = epoch
                self.patience_counter = 0
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    global_step=self.global_step,
                    val_loss=val_loss,
                    is_best=True,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, self.early_stopping_patience)
                    break

        elapsed = time.time() - start_time
        return {
            "total_epochs": len(self.history),
            "best_epoch": best_epoch,
            "best_val_loss": self.best_val_loss,
            "duration_seconds": elapsed,
            "history": self.history,
        }

    def train(self) -> float:
        """Backward-compatible train method returning best_val_loss."""
        res = self.fit()
        return res["best_val_loss"]


def train_model(
    model: TwoTowerModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "data/checkpoints",
    **kwargs,
) -> float:
    """
    Convenience function for backward compatibility.
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )
    return trainer.train()

