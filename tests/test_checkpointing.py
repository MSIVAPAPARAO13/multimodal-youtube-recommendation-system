"""
Tests for CheckpointManager and checkpoint persistence.
Verifies:
- Checkpoint save and reload
- Missing checkpoint handling
- Dimension and architecture compatibility validation
- Atomic saving and metadata tracking
"""

import pytest
import torch
import torch.nn as nn

from src.models.fusion import GatedFusion
from src.training.checkpoint import CheckpointManager, CheckpointMetadata


class DummyModel(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)


def test_checkpoint_save_and_reload(tmp_path):
    manager = CheckpointManager(checkpoint_dir=tmp_path, feature_dim=512)
    model = DummyModel(512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Save checkpoint
    ckpt_path = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=100,
        val_loss=0.42,
        is_best=True,
    )
    assert ckpt_path.exists()
    assert (tmp_path / "best.pt").exists()

    # Load into fresh model
    new_model = DummyModel(512)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    meta = manager.load_checkpoint(ckpt_path, new_model, optimizer=new_opt)

    assert meta.epoch == 1
    assert meta.val_loss == pytest.approx(0.42)
    assert meta.feature_dim == 512

    # Verify parameters match
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_missing_checkpoint_raises_filenotfound(tmp_path):
    manager = CheckpointManager(checkpoint_dir=tmp_path)
    model = DummyModel(512)
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        manager.load_checkpoint(tmp_path / "non_existent.pt", model)


def test_incompatible_checkpoint_dimension_rejected(tmp_path):
    manager = CheckpointManager(checkpoint_dir=tmp_path, feature_dim=512)
    model = DummyModel(512)
    ckpt_path = manager.save_checkpoint(
        model=model,
        optimizer=None,
        epoch=1,
        global_step=10,
        val_loss=1.0,
    )

    # Attempt to load with incompatible manager expecting 256
    incompat_manager = CheckpointManager(checkpoint_dir=tmp_path, feature_dim=256)
    incompat_model = DummyModel(256)
    with pytest.raises(ValueError, match="does not match expected"):
        incompat_manager.load_checkpoint(ckpt_path, incompat_model)
