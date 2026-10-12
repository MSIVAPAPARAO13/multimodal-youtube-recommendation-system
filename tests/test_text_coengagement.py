"""
Unit and Integration Tests for Text-Only Metadata Co-Engagement Experiment.

Verifies:
- Interaction parsing and 24h time delta filtering
- Deterministic user-disjoint splitting (zero user leakage)
- Feature extraction & projection head forward pass (768-D -> 512-D)
- Gradient flow to projection parameters
- Checkpoint persistence and loading into TextEncoder.projection
- Retrieval evaluation metric calculation & self-match exclusion
"""

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from src.data.text_coengagement import (
    CoengagementPair,
    TextCoengagementBuilder,
    PairedTextFeatureDataset,
)
from src.models.text_projection_model import TextProjectionModel
from src.training.losses import ContrastiveLoss
from scripts.evaluate_text_coengagement import compute_metrics


def test_coengagement_pair_dataclass():
    pair = CoengagementPair(
        user_id=101,
        anchor_item=1,
        positive_item=2,
        delta_seconds=3600.0,
        anchor_timestamp=1000000,
        positive_timestamp=1003600,
        split="train",
    )
    assert pair.user_id == 101
    assert pair.anchor_item == 1
    assert pair.positive_item == 2
    assert pair.delta_seconds == 3600.0
    assert pair.split == "train"


def test_user_disjoint_splitting_and_zero_leakage():
    builder = TextCoengagementBuilder(seed=42)
    # Test with cohort of 50 users
    res = builder.build_dataset(max_users=50)

    train_users = {p.user_id for p in res["train_pairs"]}
    val_users = {p.user_id for p in res["val_pairs"]}
    test_users = {p.user_id for p in res["test_pairs"]}

    # Verify complete disjointness (zero leakage)
    assert len(train_users.intersection(val_users)) == 0
    assert len(train_users.intersection(test_users)) == 0
    assert len(val_users.intersection(test_users)) == 0

    assert len(res["train_pairs"]) > 0
    assert len(res["unique_items"]) > 0


def test_time_gap_filtering():
    # Builder with strict 1-hour window
    builder_1h = TextCoengagementBuilder(max_delta_seconds=3600.0, seed=42)
    res_1h = builder_1h.build_dataset(max_users=100)

    for p in res_1h["train_pairs"] + res_1h["val_pairs"] + res_1h["test_pairs"]:
        assert p.delta_seconds <= 3600.0
        assert p.delta_seconds >= 0.0


def test_text_projection_head_forward_and_norm():
    model = TextProjectionModel(input_dim=768, hidden_dim=768, output_dim=512)
    dummy_input = torch.randn(8, 768)

    output = model(dummy_input)

    assert output.shape == (8, 512)
    # Check L2 normalization
    norms = torch.norm(output, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-4)


def test_projection_gradient_propagation_and_finite_loss():
    model = TextProjectionModel(input_dim=768, hidden_dim=768, output_dim=512)
    criterion = ContrastiveLoss(temperature=0.07, symmetric=True)

    anchor_feats = torch.randn(4, 768)
    pos_feats = torch.randn(4, 768)

    z_a = model(anchor_feats)
    z_p = model(pos_feats)

    loss = criterion(z_a, z_p)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()

    # Verify all projection layers received gradients
    for name, param in model.projection.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.norm(param.grad) > 0.0


def test_compute_metrics_self_match_exclusion():
    # 2 queries and 3 candidates: c1, c2, c3
    # q1 (id=1) matches c2 (id=2). If candidate gallery includes id=1, it must be excluded.
    query_emb = torch.tensor([[1.0, 0.0]])
    query_ids = [1]

    # candidate 1 has id=1 (self match!), candidate 2 has id=2 (true match)
    cand_emb = torch.tensor([
        [1.0, 0.0],  # id=1 (self)
        [0.9, 0.1],  # id=2 (true positive)
        [0.0, 1.0],  # id=3
    ])
    cand_ids = [1, 2, 3]
    ground_truth = {1: {2}}

    metrics = compute_metrics(
        query_embeddings=query_emb,
        query_item_ids=query_ids,
        candidate_embeddings=cand_emb,
        candidate_item_ids=cand_ids,
        ground_truth_map=ground_truth,
        top_k_list=(1, 2),
    )

    # Since self-match (id=1) is excluded, id=2 is at rank 1!
    assert metrics["Recall@1"] == 1.0
    assert metrics["MRR"] == 1.0


def test_checkpoint_reloading_into_production_text_encoder(tmp_path):
    """
    Verify that TextProjectionModel state_dict loads cleanly into
    the production TextEncoder.projection layer without shape or key mismatches.
    """
    model = TextProjectionModel(input_dim=768, hidden_dim=768, output_dim=512)
    ckpt_file = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_file)

    # Load into mock TextEncoder
    from src.features.text_encoder import TextEncoder

    with patch("transformers.BertTokenizer.from_pretrained") as mock_tok, \
         patch("transformers.BertModel.from_pretrained") as mock_bert:

        encoder = TextEncoder(feature_dim=512, pretrained=False)
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)

        # Load projection weights
        encoder.projection.load_state_dict(ckpt["model_state_dict"]["projection"] if "projection" in ckpt["model_state_dict"] else {
            k.replace("projection.", ""): v for k, v in ckpt["model_state_dict"].items()
        })

        # Test forward pass through projection
        dummy_bert_out = torch.randn(2, 768)
        proj_out = encoder.projection(dummy_bert_out)
        assert proj_out.shape == (2, 512)
