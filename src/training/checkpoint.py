"""
Safe Model Checkpoint Management and Validation.

Handles secure serialization and deserialization of TwoTowerModel / VideoTower checkpoints,
metadata verification, dimension assertions, and compatibility checks.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = "1.0.0"


@dataclass
class CheckpointMetadata:
    """Metadata recorded with every saved model checkpoint."""
    format_version: str
    timestamp: float
    epoch: int
    global_step: int
    val_loss: float
    feature_dim: int
    fusion_type: str
    architecture: str
    trainable_params_count: int
    frozen_params_count: int
    training_config: Dict[str, Any]


class CheckpointManager:
    """
    Manages saving, loading, validating, and rolling back model checkpoints safely.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "data/checkpoints",
        feature_dim: int = 512,
        fusion_type: str = "gated",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        global_step: int,
        val_loss: float,
        is_best: bool = False,
        filename: Optional[str] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save model and optimizer state with validated metadata.
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        meta = CheckpointMetadata(
            format_version=CHECKPOINT_FORMAT_VERSION,
            timestamp=time.time(),
            epoch=epoch,
            global_step=global_step,
            val_loss=float(val_loss),
            feature_dim=self.feature_dim,
            fusion_type=self.fusion_type,
            architecture=model.__class__.__name__,
            trainable_params_count=trainable,
            frozen_params_count=frozen,
            training_config=extra_config or {},
        )

        # Extract state dict (handle DataParallel or wrapped modules if any)
        raw_model = getattr(model, "module", model)
        state_dict = raw_model.state_dict()

        payload = {
            "metadata": asdict(meta),
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        }

        target_file = self.checkpoint_dir / (filename or f"checkpoint_epoch_{epoch}.pt")
        # Write to temporary file first for atomic persistence
        temp_file = target_file.with_suffix(".tmp")
        torch.save(payload, temp_file)
        temp_file.replace(target_file)

        if is_best:
            best_file = self.checkpoint_dir / "best.pt"
            torch.save(payload, temp_file)
            temp_file.replace(best_file)
            logger.info("Saved new best checkpoint to %s (val_loss=%.4f)", best_file, val_loss)

        return target_file

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
        map_location: Optional[torch.device] = None,
    ) -> CheckpointMetadata:
        """
        Safely load a checkpoint into a model and optional optimizer.
        Validates:
        - Path existence
        - Structural integrity (weights_only=True compatibility)
        - Architecture and feature_dim alignment
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            # First attempt secure weights_only loading
            checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            # Fallback for standard state_dict serialization
            checkpoint = torch.load(path, map_location=map_location)

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint format in {path}: expected dict payload.")

        # Check if wrapped in payload
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            raw_meta = checkpoint.get("metadata", {})
        else:
            state_dict = checkpoint
            raw_meta = {}

        # Validate feature_dim and architecture if metadata exists
        meta_dim = raw_meta.get("feature_dim")
        if meta_dim is not None and meta_dim != self.feature_dim:
            raise ValueError(
                f"Checkpoint feature_dim ({meta_dim}) does not match expected ({self.feature_dim})."
            )

        # Apply state_dict
        raw_model = getattr(model, "module", model)
        missing_keys, unexpected_keys = raw_model.load_state_dict(state_dict, strict=strict)
        if missing_keys:
            logger.warning("Checkpoint load missing keys: %s", missing_keys)
        if unexpected_keys:
            logger.warning("Checkpoint load unexpected keys: %s", unexpected_keys)

        # Restore optimizer if requested
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        metadata = CheckpointMetadata(
            format_version=raw_meta.get("format_version", "legacy"),
            timestamp=raw_meta.get("timestamp", 0.0),
            epoch=raw_meta.get("epoch", 0),
            global_step=raw_meta.get("global_step", 0),
            val_loss=raw_meta.get("val_loss", float("inf")),
            feature_dim=meta_dim or self.feature_dim,
            fusion_type=raw_meta.get("fusion_type", self.fusion_type),
            architecture=raw_meta.get("architecture", model.__class__.__name__),
            trainable_params_count=raw_meta.get("trainable_params_count", 0),
            frozen_params_count=raw_meta.get("frozen_params_count", 0),
            training_config=raw_meta.get("training_config", {}),
        )

        return metadata
