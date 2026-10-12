"""
Tests for Multimodal Training Loop and Architecture Integration.
Verifies:
- Gradient propagation to projection heads and fusion
- Frozen backbone behavior
- Configuration parameter validation
- Inference API compatibility with checkpoints
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unittest.mock import MagicMock, patch

from src.db.config import Settings
from src.models.fusion import GatedFusion
from src.models.two_tower import TwoTowerModel
from src.training.trainer import Trainer
from src.data.training_dataset import MultimodalTrainingDataset, MultimodalTrainingSample
from src.data.pair_generator import PairGenerator, PairedMultimodalDataset
from src.services.multimodal_inference import MultimodalInferenceService


def test_training_configuration_validation():
    # Valid configuration
    s = Settings(
        TRAINING_BATCH_SIZE=16,
        TRAINING_EPOCHS=5,
        TRAINING_LEARNING_RATE=1e-4,
        TRAINING_TEMPERATURE=0.07,
        TRAINING_VALIDATION_SPLIT=0.2,
    )
    assert s.TRAINING_BATCH_SIZE == 16

    # Invalid batch size
    with pytest.raises(ValueError, match="strictly positive"):
        Settings(TRAINING_BATCH_SIZE=0)

    # Invalid epochs
    with pytest.raises(ValueError, match="strictly positive"):
        Settings(TRAINING_EPOCHS=-1)

    # Invalid learning rate
    with pytest.raises(ValueError, match="strictly positive"):
        Settings(TRAINING_LEARNING_RATE=-0.01)

    # Invalid validation split
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\)"):
        Settings(TRAINING_VALIDATION_SPLIT=1.5)


def test_gradient_propagation_and_frozen_backbone(tmp_path):
    """
    Ensure that when training TwoTowerModel:
    1. Backbones remain frozen (requires_grad == False).
    2. Projection heads and fusion gate receive gradients.
    """
    with patch("transformers.BertTokenizer.from_pretrained") as mock_tok:
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(2, 16, dtype=torch.long),
            "attention_mask": torch.ones(2, 16, dtype=torch.long)
        }
        mock_tok.return_value = mock_tokenizer

        model = TwoTowerModel(
            feature_dim=512,
            share_towers=True,
            freeze_encoders=True,
            pretrained=False,
        )

    # Verify backbone parameters have requires_grad == False
    for param in model.video_tower.video_encoder.backbone.parameters():
        assert not param.requires_grad

    # Verify fusion parameters have requires_grad == True
    for param in model.video_tower.fusion.gate.parameters():
        assert param.requires_grad

    # Build tiny synthetic dataset
    ds = MultimodalTrainingDataset(feature_dim=512)
    s1 = MultimodalTrainingSample(
        video_id="v1",
        visual_embedding=torch.randn(512),
        audio_embedding=torch.randn(512),
        text_embedding=torch.randn(512),
        positive_video_ids=["v2"],
        split="train",
    )
    s2 = MultimodalTrainingSample(
        video_id="v2",
        visual_embedding=torch.randn(512),
        audio_embedding=torch.randn(512),
        text_embedding=torch.randn(512),
        positive_video_ids=["v1"],
        split="train",
    )
    ds.add_sample(s1)
    ds.add_sample(s2)

    pair_gen = PairGenerator(ds, seed=42)
    pairs = pair_gen.generate_pairs()
    loader = DataLoader(PairedMultimodalDataset(ds, pairs), batch_size=2)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        num_epochs=1,
        checkpoint_dir=str(tmp_path),
        device="cpu",
    )

    # Run one epoch
    loss = trainer.train_epoch(1)
    assert loss > 0.0

    # Confirm gate parameters received gradients
    has_grad = any(
        p.grad is not None and torch.norm(p.grad) > 0.0
        for p in model.video_tower.fusion.gate.parameters()
    )
    assert has_grad, "Fusion gate parameters did not receive gradients during training!"


def test_inference_service_compatibility_with_checkpoint(tmp_path):
    """
    Ensure MultimodalInferenceService can safely initialize and utilize a trained checkpoint.
    """
    with patch("transformers.BertTokenizer.from_pretrained") as mock_tok:
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(2, 16, dtype=torch.long),
            "attention_mask": torch.ones(2, 16, dtype=torch.long)
        }
        mock_tok.return_value = mock_tokenizer

        model = TwoTowerModel(
            feature_dim=512,
            share_towers=True,
            pretrained=False,
        )
        ckpt_path = tmp_path / "test_model.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        settings = Settings(
            MODEL_PATH=str(ckpt_path),
            ALLOW_UNINITIALIZED_MODEL=True,
            PRETRAINED_FOUNDATION=False,
        )

        svc = MultimodalInferenceService(settings=settings, device=torch.device("cpu"))
        assert svc.is_checkpoint_loaded is True

        # Check forward fusion works
        v = torch.randn(1, 512)
        a = torch.randn(1, 512)
        t = torch.randn(1, 512)
        fused = svc.fuse(v, a, t)
        assert fused.shape == (1, 512)
        assert torch.isclose(torch.norm(fused, p=2), torch.tensor(1.0), atol=1e-4)

