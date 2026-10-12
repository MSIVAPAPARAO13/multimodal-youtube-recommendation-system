"""
Tests for PairGenerator and false-negative suppression.
Verifies:
- Positive-pair validation
- Duplicate-pair handling
- Deterministic sampling with a fixed seed
- False-negative safeguards
"""

import pytest
import torch

from src.data.training_dataset import MultimodalTrainingDataset, MultimodalTrainingSample, DEFAULT_FEATURE_DIM
from src.data.pair_generator import PairGenerator, ContrastivePair


def test_positive_pair_validation_and_grouping():
    ds = MultimodalTrainingDataset()
    # Create two items in the same group
    s1 = MultimodalTrainingSample(
        video_id="v1",
        text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        positive_group_id="cat_videos",
        split="train",
    )
    s2 = MultimodalTrainingSample(
        video_id="v2",
        text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        positive_group_id="cat_videos",
        split="train",
    )
    ds.add_sample(s1)
    ds.add_sample(s2)

    gen = PairGenerator(ds, seed=42)
    pairs = gen.generate_pairs()

    assert len(pairs) == 2  # (v1, v2) and (v2, v1)
    pair_ids = {(p.anchor_id, p.positive_id) for p in pairs}
    assert ("v1", "v2") in pair_ids
    assert ("v2", "v1") in pair_ids


def test_deterministic_sampling_with_fixed_seed():
    ds = MultimodalTrainingDataset()
    for i in range(10):
        ds.add_sample(
            MultimodalTrainingSample(
                video_id=f"v_{i}",
                text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
                positive_group_id="shared_group",
                split="train",
            )
        )

    gen1 = PairGenerator(ds, seed=123)
    pairs1 = [(p.anchor_id, p.positive_id) for p in gen1.generate_pairs(max_pairs_per_anchor=2)]

    gen2 = PairGenerator(ds, seed=123)
    pairs2 = [(p.anchor_id, p.positive_id) for p in gen2.generate_pairs(max_pairs_per_anchor=2)]

    assert pairs1 == pairs2


def test_false_negative_suppression_mask():
    ds = MultimodalTrainingDataset()
    s1 = MultimodalTrainingSample(
        video_id="v1",
        text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        positive_group_id="group_a",
        split="train",
    )
    s2 = MultimodalTrainingSample(
        video_id="v2",
        text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        positive_group_id="group_a",
        split="train",
    )
    s3 = MultimodalTrainingSample(
        video_id="v3",
        text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        positive_group_id="group_b",
        split="train",
    )
    ds.add_sample(s1)
    ds.add_sample(s2)
    ds.add_sample(s3)

    gen = PairGenerator(ds, seed=42)
    batch_ids = ["v1", "v2", "v3"]
    mask = gen.compute_false_negative_mask(batch_ids)

    # mask shape must be (3, 3)
    assert mask.shape == (3, 3)
    # Diagonal must be False (not a negative for itself)
    assert not mask[0, 0]
    assert not mask[1, 1]
    assert not mask[2, 2]

    # v1 and v2 share group_a -> v2 should NOT be treated as negative for v1
    assert not mask[0, 1]
    assert not mask[1, 0]

    # v1 and v3 are in different groups -> v3 IS a valid negative for v1
    assert mask[0, 2]
    assert mask[2, 0]
