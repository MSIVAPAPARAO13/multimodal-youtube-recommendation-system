"""
Tests for MultimodalTrainingDataset and schema validation.
Verifies:
- Schema validation
- Dimension checks
- NaN and Inf rejection
- Split leakage prevention
- Missing modality masks
"""

import pytest
import torch

from src.data.training_dataset import (
    MultimodalTrainingDataset,
    MultimodalTrainingSample,
    DEFAULT_FEATURE_DIM,
)


def test_valid_training_sample_creation():
    sample = MultimodalTrainingSample(
        video_id="vid_001",
        visual_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        audio_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        title="Test Video",
        positive_group_id="group_1",
        positive_video_ids=["vid_002"],
        label_source="curated",
        split="train",
    )
    assert sample.video_id == "vid_001"
    assert sample.modality_availability == [True, True, True]
    v, a, t, mask = sample.get_fused_input()
    assert v.shape == (DEFAULT_FEATURE_DIM,)
    assert mask.tolist() == [True, True, True]


def test_invalid_embedding_dimensions_rejected():
    with pytest.raises(ValueError, match="visual_embedding must have shape"):
        MultimodalTrainingSample(
            video_id="vid_bad_dim",
            visual_embedding=torch.randn(256),  # Incorrect dimension (expected 512)
            split="train",
        )


def test_nan_and_inf_rejected():
    nan_tensor = torch.randn(DEFAULT_FEATURE_DIM)
    nan_tensor[10] = float("nan")

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        MultimodalTrainingSample(
            video_id="vid_nan",
            text_embedding=nan_tensor,
            split="train",
        )

    inf_tensor = torch.randn(DEFAULT_FEATURE_DIM)
    inf_tensor[5] = float("inf")

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        MultimodalTrainingSample(
            video_id="vid_inf",
            audio_embedding=inf_tensor,
            split="train",
        )


def test_empty_video_id_rejected():
    with pytest.raises(ValueError, match="video_id must be a non-empty string"):
        MultimodalTrainingSample(
            video_id="",
            text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
            split="train",
        )


def test_invalid_split_rejected():
    with pytest.raises(ValueError, match="split 'invalid_split' must be one of"):
        MultimodalTrainingSample(
            video_id="vid_001",
            text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
            split="invalid_split",
        )


def test_self_positive_reference_rejected():
    with pytest.raises(ValueError, match="cannot have itself in positive_video_ids"):
        MultimodalTrainingSample(
            video_id="vid_001",
            text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
            positive_video_ids=["vid_001"],
            split="train",
        )


def test_dataset_duplicate_video_id_rejected():
    ds = MultimodalTrainingDataset()
    sample1 = MultimodalTrainingSample(
        video_id="vid_001",
        text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        split="train",
    )
    sample2 = MultimodalTrainingSample(
        video_id="vid_001",
        text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        split="train",
    )
    ds.add_sample(sample1)
    with pytest.raises(ValueError, match="Duplicate video_id 'vid_001' detected"):
        ds.add_sample(sample2)


def test_split_leakage_detection():
    ds = MultimodalTrainingDataset()
    sample_train = MultimodalTrainingSample(
        video_id="vid_train",
        text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        split="train",
        positive_video_ids=["vid_test"],  # Leaking across split!
    )
    sample_test = MultimodalTrainingSample(
        video_id="vid_test",
        text_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        split="test",
    )
    ds.add_sample(sample_train)
    ds.add_sample(sample_test)

    with pytest.raises(RuntimeError, match="positive pairs span different splits"):
        ds.validate_integrity()


def test_serialization_and_deserialization():
    ds = MultimodalTrainingDataset()
    s = MultimodalTrainingSample(
        video_id="vid_serial",
        visual_embedding=torch.randn(DEFAULT_FEATURE_DIM),
        title="Serialized Title",
        split="train",
    )
    ds.add_sample(s)
    dict_list = ds.to_dict_list()
    assert len(dict_list) == 1

    restored = MultimodalTrainingDataset.from_dict_list(dict_list)
    assert len(restored) == 1
    r_sample = restored.get_by_id("vid_serial")
    assert r_sample.title == "Serialized Title"
    assert r_sample.visual_embedding.shape == (DEFAULT_FEATURE_DIM,)
