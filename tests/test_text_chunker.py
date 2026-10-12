"""
Tests for TextChunker (TASK 12: Text Chunking and Hierarchical Pooling Tests).
"""

import pytest
import torch
from unittest.mock import MagicMock
from src.features.text_chunker import (
    TextChunker,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_TOKENS,
)


def test_chunker_initialization_validation():
    """Verify validation on chunk sizes, overlaps, and max tokens."""
    # Invalid chunk_size <= 0
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0)

    # Invalid overlap >= chunk_size
    with pytest.raises(ValueError, match="chunk_overlap .* strictly less than chunk_size"):
        TextChunker(chunk_size=128, chunk_overlap=128)

    # Invalid max_tokens <= 0
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        TextChunker(max_tokens=-5)


def test_chunk_token_ids_short_sequence():
    """Verify text shorter than chunk_size produces exactly 1 chunk."""
    chunker = TextChunker(chunk_size=128, chunk_overlap=32, max_tokens=1024)
    tokens = list(range(50))
    chunks = chunker.chunk_token_ids(tokens)

    assert len(chunks) == 1
    assert chunks[0] == tokens


def test_chunk_token_ids_long_sequence():
    """Verify long token sequence is chunked with proper step stride."""
    chunk_size = 100
    overlap = 20
    stride = chunk_size - overlap  # 80
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap, max_tokens=1000)

    tokens = list(range(250))
    chunks = chunker.chunk_token_ids(tokens)

    # 0..100 (len 100)
    # 80..180 (len 100)
    # 160..250 (len 90)
    assert len(chunks) == 3
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert len(chunks[2]) == 90


def test_chunk_overlap_exact_match():
    """Verify consecutive chunks share precisely chunk_overlap tokens."""
    chunk_size = 64
    overlap = 16
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap, max_tokens=500)

    tokens = list(range(200))
    chunks = chunker.chunk_token_ids(tokens)

    # Verify overlap between chunk 0 and chunk 1
    tail_chunk_0 = chunks[0][-overlap:]
    head_chunk_1 = chunks[1][:overlap]
    assert tail_chunk_0 == head_chunk_1


def test_deterministic_chunking():
    """Verify chunking behavior is 100% deterministic across repeated calls."""
    chunker = TextChunker(chunk_size=128, chunk_overlap=32, max_tokens=1024)
    tokens = [i * 3 for i in range(350)]

    run1 = chunker.chunk_token_ids(tokens)
    run2 = chunker.chunk_token_ids(tokens)

    assert run1 == run2
    for c1, c2 in zip(run1, run2):
        assert c1 == c2


def test_token_limits_bound():
    """Verify token sequence is strictly bounded by max_tokens."""
    max_tokens = 300
    chunker = TextChunker(chunk_size=100, chunk_overlap=20, max_tokens=max_tokens)

    tokens = list(range(1000))
    chunks = chunker.chunk_token_ids(tokens)

    # Max tokens processed should never exceed 300
    total_unique_tokens = max(chunks[-1]) + 1
    assert total_unique_tokens == max_tokens


def test_chunk_tensors_padding_and_shape():
    """Verify tensor chunking produces correct batch tensor dimensions and masks."""
    chunk_size = 64
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=16, max_tokens=256)

    # 150 token IDs
    input_ids = torch.arange(1, 151, dtype=torch.long)
    attention_mask = torch.ones(150, dtype=torch.long)

    chunked_ids, chunked_mask = chunker.chunk_tensors(input_ids, attention_mask, pad_token_id=0)

    assert chunked_ids.dim() == 2
    assert chunked_ids.size(1) == chunk_size
    assert chunked_mask.shape == chunked_ids.shape

    # First chunk should have full attention mask
    assert chunked_mask[0].sum().item() == chunk_size


def test_pool_chunk_embeddings_mean():
    """Verify mean pooling averages across chunks into (1, 512)."""
    chunker = TextChunker()
    # 3 chunks of 512-d embeddings
    chunk_embs = torch.ones(3, 512)
    chunk_embs[0] = 1.0
    chunk_embs[1] = 2.0
    chunk_embs[2] = 3.0

    pooled = chunker.pool_chunk_embeddings(chunk_embs, method="mean")
    assert pooled.shape == (1, 512)
    assert torch.allclose(pooled, torch.full((1, 512), 2.0))


def test_pool_chunk_embeddings_decay_and_weighted():
    """Verify decay and custom weighted pooling."""
    chunker = TextChunker()
    chunk_embs = torch.randn(4, 512)

    # Decay pooling
    decay_pooled = chunker.pool_chunk_embeddings(chunk_embs, method="decay")
    assert decay_pooled.shape == (1, 512)
    assert not torch.isnan(decay_pooled).any()

    # Weighted pooling
    weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
    weighted_pooled = chunker.pool_chunk_embeddings(chunk_embs, method="weighted", weights=weights)
    assert weighted_pooled.shape == (1, 512)
    assert not torch.isnan(weighted_pooled).any()


def test_nan_inf_protection_in_pooling():
    """Verify pooling replaces NaNs and Infs with zeros."""
    chunker = TextChunker()
    bad_embs = torch.tensor([[float("nan"), 1.0, float("inf")], [2.0, float("-inf"), 3.0]])

    pooled = chunker.pool_chunk_embeddings(bad_embs, method="mean")
    assert not torch.isnan(pooled).any()
    assert not torch.isinf(pooled).any()
