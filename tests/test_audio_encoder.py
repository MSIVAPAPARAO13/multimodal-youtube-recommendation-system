"""
Tests for AudioEncoder: Architecture, Wav2Vec2 backbone configuration, pooling, and projection.
"""

import pytest
import torch
import torch.nn.functional as F

from src.features.audio_encoder import AudioEncoder


@pytest.fixture
def local_audio_encoder():
    """Create local initialized AudioEncoder without network downloads."""
    encoder = AudioEncoder(
        model_name="facebook/wav2vec2-base-960h",
        feature_dim=512,
        aggregation="mean",
        freeze_backbone=True,
        pretrained=False,
    )
    encoder.eval()
    return encoder


def test_audio_encoder_shapes(local_audio_encoder):
    """Test 1: Output shape is strictly (B, 512) for single and batched waveforms."""
    # Single 1D waveform: 16,000 samples (1 second)
    waveform_1d = torch.randn(16000)
    with torch.no_grad():
        emb_1d = local_audio_encoder(waveform_1d)
    assert emb_1d.shape == (1, 512)

    # Batched 2D waveforms: (2, 32000)
    waveform_2d = torch.randn(2, 32000)
    with torch.no_grad():
        emb_2d = local_audio_encoder(waveform_2d)
    assert emb_2d.shape == (2, 512)


def test_audio_encoder_attention_mask(local_audio_encoder):
    """Test 2: Mask-aware pooling ignores padding region."""
    waveform = torch.randn(1, 32000)
    # Mask half the waveform as padding
    mask = torch.ones(1, 32000)
    mask[:, 16000:] = 0.0

    with torch.no_grad():
        emb_masked = local_audio_encoder(waveform, attention_mask=mask)
        emb_unmasked = local_audio_encoder(waveform)

    assert emb_masked.shape == (1, 512)
    assert not torch.allclose(emb_masked, emb_unmasked)


def test_audio_encoder_encode_chunks(local_audio_encoder):
    """Test 3: Chunk-level encoding and aggregation."""
    # 3 chunks of 16,000 samples
    chunks = torch.randn(3, 16000)
    mask = torch.ones(3, 16000)

    with torch.no_grad():
        emb = local_audio_encoder.encode_chunks(chunks, attention_mask=mask, aggregation="mean")

    assert emb.shape == (1, 512)
    assert not torch.isnan(emb).any()


def test_audio_encoder_deterministic_output(local_audio_encoder):
    """Test 4: Inference is strictly deterministic on identical input."""
    waveform = torch.randn(1, 16000)

    with torch.no_grad():
        emb1 = local_audio_encoder(waveform)
        emb2 = local_audio_encoder(waveform)

    assert torch.allclose(emb1, emb2, atol=1e-6)


def test_audio_encoder_nan_inf_safety(local_audio_encoder):
    """Test 5: Encoder outputs finite numbers even when non-finite values are in input."""
    waveform = torch.tensor([[0.5, float("nan"), -0.5, float("inf")] + [0.0] * 15996])

    with torch.no_grad():
        emb = local_audio_encoder(waveform)

    assert not torch.isnan(emb).any()
    assert not torch.isinf(emb).any()
