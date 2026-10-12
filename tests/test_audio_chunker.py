"""
Tests for AudioChunker: Windowing, attention masks, bounded processing, and deterministic chunk aggregation.
"""

import pytest
import torch

from src.features.audio_chunker import AudioChunker, AudioChunkResult


def test_deterministic_chunk_boundaries():
    """Verify exact chunk boundaries and time spans for 16 kHz audio."""
    chunker = AudioChunker(chunk_duration_seconds=5.0, overlap_seconds=1.0, sample_rate=16000)
    # 12 seconds audio = 192,000 samples
    waveform = torch.arange(192000, dtype=torch.float32)

    res = chunker.chunk_waveform(waveform)
    assert res.num_chunks > 1
    assert res.chunk_duration_seconds == 5.0
    assert res.overlap_seconds == 1.0

    # First chunk spans 0s to 5s
    assert res.chunk_times[0] == (0.0, 5.0)
    # Second chunk starts at 4.0s (5.0 - 1.0) and spans to 9.0s
    assert res.chunk_times[1] == (4.0, 9.0)


def test_final_partial_chunk_and_attention_mask():
    """Verify that a final partial chunk is zero-padded and masked with zeros."""
    chunker = AudioChunker(chunk_duration_seconds=5.0, overlap_seconds=1.0, sample_rate=16000)
    chunk_samples = 5 * 16000  # 80,000

    # 6.5 seconds = 104,000 samples
    # Chunk 1: [0 : 80000] (full, mask all 1s)
    # Chunk 2: [64000 : 104000] has 40,000 samples; padded by 40,000 zeros
    waveform = torch.ones(104000, dtype=torch.float32)

    res = chunker.chunk_waveform(waveform)
    assert res.num_chunks == 2
    assert res.chunks.shape == (2, chunk_samples)
    assert res.attention_mask.shape == (2, chunk_samples)

    # Chunk 1 attention mask is all 1s
    assert (res.attention_mask[0] == 1.0).all()

    # Chunk 2 attention mask: first 40,000 are 1s, next 40,000 are 0s
    assert (res.attention_mask[1, :40000] == 1.0).all()
    assert (res.attention_mask[1, 40000:] == 0.0).all()


def test_chunk_count_limit_and_truncation():
    """Verify that audio exceeding max_chunks is safely capped and recorded."""
    chunker = AudioChunker(chunk_duration_seconds=2.0, overlap_seconds=0.0, max_chunks=3, sample_rate=16000)
    # 10 seconds of audio would normally require 5 chunks
    waveform = torch.randn(10 * 16000)

    res = chunker.chunk_waveform(waveform)
    assert res.num_chunks == 3
    assert res.is_truncated is True
    assert res.chunks.shape[0] == 3


def test_aggregation_mean():
    """Verify deterministic mean aggregation of chunk representations."""
    chunker = AudioChunker()
    # 3 chunks, feature_dim 4
    embeddings = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [3.0, 6.0, 9.0, 12.0],
    ])
    aggregated = chunker.aggregate_chunk_embeddings(embeddings, mode="mean")
    assert aggregated.shape == (1, 4)
    expected = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    assert torch.allclose(aggregated, expected)


def test_aggregation_weighted_and_max():
    """Verify weighted/recency and max pooling aggregations."""
    chunker = AudioChunker()
    embeddings = torch.tensor([
        [1.0, 10.0],
        [5.0, 2.0],
    ])

    agg_max = chunker.aggregate_chunk_embeddings(embeddings, mode="max")
    assert agg_max.shape == (1, 2)
    assert torch.allclose(agg_max, torch.tensor([[5.0, 10.0]]))

    agg_weighted = chunker.aggregate_chunk_embeddings(embeddings, mode="weighted")
    assert agg_weighted.shape == (1, 2)
    assert not torch.isnan(agg_weighted).any()


def test_empty_waveform_chunking():
    """Verify chunking an empty waveform returns a zero-chunk result."""
    chunker = AudioChunker(chunk_duration_seconds=5.0, sample_rate=16000)
    res = chunker.chunk_waveform(torch.tensor([]))
    assert res.num_chunks == 0
    assert res.chunks.shape == (0, 5 * 16000)
    assert res.attention_mask.shape == (0, 5 * 16000)
