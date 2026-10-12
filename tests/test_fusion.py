import torch
from models.fusion import GatedFusion, ConcatFusion, AttentionFusion


def test_concat_fusion_shape():
    """Test ConcatFusion produces expected output shape."""
    fusion = ConcatFusion(input_dim=512, output_dim=512, num_modalities=3)
    batch_size = 4
    video = torch.randn(batch_size, 512)
    audio = torch.randn(batch_size, 512)
    text = torch.randn(batch_size, 512)

    output = fusion(video, audio, text)
    assert output.shape == (batch_size, 512)
    assert not torch.isnan(output).any()


def test_gated_fusion_shape():
    """Test GatedFusion produces expected output shape and weights."""
    fusion = GatedFusion(input_dim=512, output_dim=512)
    batch_size = 4
    video = torch.randn(batch_size, 512)
    audio = torch.randn(batch_size, 512)
    text = torch.randn(batch_size, 512)

    output = fusion(video, audio, text)
    assert output.shape == (batch_size, 512)
    assert not torch.isnan(output).any()

    weights = fusion.get_gate_weights(video, audio, text)
    assert weights.shape == (batch_size, 3)
    # Weights must sum to 1.0 across the 3 modalities
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)


def test_gated_fusion_missing_modality():
    """Test GatedFusion safely handles missing (all-zero) modalities."""
    fusion = GatedFusion(input_dim=512, output_dim=512)
    batch_size = 2
    video = torch.randn(batch_size, 512)
    audio = torch.zeros(batch_size, 512)  # Missing audio track
    text = torch.randn(batch_size, 512)

    output = fusion(video, audio, text)
    assert output.shape == (batch_size, 512)
    assert not torch.isnan(output).any()

    weights = fusion.get_gate_weights(video, audio, text)
    # Weight for missing audio (index 1) should be 0.0 or near zero
    assert (weights[:, 1] < 1e-4).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)


def test_attention_fusion_shape():
    """Test AttentionFusion produces expected output shape."""
    fusion = AttentionFusion(input_dim=512, output_dim=512, num_heads=8)
    batch_size = 3
    video = torch.randn(batch_size, 512)
    audio = torch.randn(batch_size, 512)
    text = torch.randn(batch_size, 512)

    output = fusion(video, audio, text)
    assert output.shape == (batch_size, 512)
    assert not torch.isnan(output).any()
