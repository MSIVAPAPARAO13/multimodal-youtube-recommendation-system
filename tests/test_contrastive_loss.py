"""
Tests for Contrastive Loss Functions.
Verifies:
- InfoNCE and Symmetric Contrastive Loss
- Temperature validation
- Numerical stability (finite loss)
- Missing-modality zero vector handling
- Gradient propagation to parameters
"""

import pytest
import torch
import torch.nn as nn

from src.training.losses import InfoNCELoss, ContrastiveLoss, SupervisedContrastiveLoss, TripletLoss


def test_invalid_temperature_raises_error():
    with pytest.raises(ValueError, match="Temperature must be strictly positive"):
        InfoNCELoss(temperature=0.0)

    with pytest.raises(ValueError, match="Temperature must be strictly positive"):
        InfoNCELoss(temperature=-0.5)


def test_infonce_finite_loss_and_gradient():
    criterion = InfoNCELoss(temperature=0.07)
    anchor = torch.randn(8, 512, requires_grad=True)
    positive = torch.randn(8, 512, requires_grad=True)

    loss = criterion(anchor, positive)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert anchor.grad is not None
    assert torch.isfinite(anchor.grad).all()
    assert positive.grad is not None
    assert torch.isfinite(positive.grad).all()


def test_infonce_with_identical_embeddings():
    criterion = InfoNCELoss(temperature=0.07)
    base = torch.randn(8, 512)
    # Identical positive pairs should yield low loss compared to random noise
    loss_identical = criterion(base, base)
    random_other = torch.randn(8, 512)
    loss_random = criterion(base, random_other)

    assert loss_identical.item() < loss_random.item()


def test_contrastive_loss_symmetric():
    criterion = ContrastiveLoss(temperature=0.07, symmetric=True)
    a = torch.randn(4, 512, requires_grad=True)
    p = torch.randn(4, 512, requires_grad=True)

    loss = criterion(a, p)
    assert torch.isfinite(loss)
    loss.backward()
    assert a.grad is not None
    assert p.grad is not None


def test_zero_vector_missing_modality_handling():
    criterion = InfoNCELoss(temperature=0.07)
    anchor = torch.randn(4, 512)
    positive = torch.randn(4, 512)

    # Make row 2 an all-zero vector (missing representation)
    anchor[2] = 0.0
    positive[2] = 0.0

    # Loss should compute without crashing or producing NaN
    loss = criterion(anchor, positive)
    assert torch.isfinite(loss)


def test_supervised_contrastive_loss():
    supcon = SupervisedContrastiveLoss(temperature=0.07)
    features = torch.randn(6, 512, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])  # 3 classes, 2 views each

    loss = supcon(features, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
