"""
Tests for VisualEncoder (Phase 6):
- ResNet-50 2048-D backbone feature extraction
- 2048-D -> 512-D projection layer
- Deterministic inference
- Frozen backbone parameters
- Frame aggregation methods (mean, decay/weighted, max)
- Single image / thumbnail encoding
- CLIP experimental encoder factory registration without model download
"""

import pytest
import torch

from src.features.visual_encoder import (
    BaseVisualEncoder,
    ResNet50VisualEncoder,
    CLIPExperimentalEncoder,
    get_visual_encoder,
)


@pytest.fixture
def resnet_encoder():
    """ResNet-50 visual encoder without requiring download (pretrained=False for fast offline test)."""
    return ResNet50VisualEncoder(
        feature_dim=512,
        pretrained=False,
        freeze_backbone=True,
        aggregation="mean",
    )


def test_resnet_backbone_and_projection_shapes(resnet_encoder):
    """Verify backbone extracts 2048-D and projection produces 512-D."""
    # (B, T, 3, H, W) = (2, 8, 3, 224, 224)
    dummy_input = torch.randn(2, 8, 3, 224, 224)

    # 1. Test backbone features
    flat_frames = dummy_input.view(16, 3, 224, 224)
    backbone_feat = resnet_encoder.extract_backbone_features(flat_frames)
    assert backbone_feat.shape == (16, 2048)

    # 2. Test projection
    proj_feat = resnet_encoder.projection(backbone_feat)
    assert proj_feat.shape == (16, 512)

    # 3. Test forward end-to-end
    output = resnet_encoder(dummy_input)
    assert output.shape == (2, 512)


def test_frozen_backbone_parameters(resnet_encoder):
    """Verify all parameters in backbone are frozen (requires_grad is False)."""
    for param in resnet_encoder.backbone.parameters():
        assert param.requires_grad is False

    # Projection layer should have parameters (can be trained or frozen as needed)
    assert any(p.numel() > 0 for p in resnet_encoder.projection.parameters())


def test_deterministic_inference(resnet_encoder):
    """Inference on identical input tensors produces identical output tensors."""
    resnet_encoder.eval()
    tensor = torch.randn(1, 4, 3, 224, 224)

    with torch.no_grad():
        out1 = resnet_encoder(tensor)
        out2 = resnet_encoder(tensor)

    torch.testing.assert_close(out1, out2)


def test_frame_aggregation_methods(resnet_encoder):
    """Verify mean, decay/weighted, and max aggregation methods."""
    tensor = torch.randn(1, 6, 3, 224, 224)

    with torch.no_grad():
        out_mean = resnet_encoder(tensor, aggregation_method="mean")
        out_decay = resnet_encoder(tensor, aggregation_method="decay")
        out_max = resnet_encoder(tensor, aggregation_method="max")

    assert out_mean.shape == (1, 512)
    assert out_decay.shape == (1, 512)
    assert out_max.shape == (1, 512)

    # Different pooling strategies produce distinct embeddings
    assert not torch.allclose(out_mean, out_decay)
    assert not torch.allclose(out_mean, out_max)


def test_encode_single_image(resnet_encoder):
    """Thumbnail / single image input shape (1, 3, 224, 224) produces (1, 512)."""
    single_img = torch.randn(1, 3, 224, 224)
    out = resnet_encoder.encode_single_image(single_img)

    assert out.shape == (1, 512)
    assert not torch.isnan(out).any()


def test_clip_experimental_encoder_without_network_download():
    """Verify CLIPExperimentalEncoder can be initialized with pretrained=False without internet."""
    clip_enc = CLIPExperimentalEncoder(feature_dim=512, pretrained=False, freeze_backbone=True)
    assert clip_enc.feature_dim == 512

    dummy_input = torch.randn(1, 4, 3, 224, 224)
    with torch.no_grad():
        out = clip_enc(dummy_input)
    assert out.shape == (1, 512)


def test_visual_encoder_factory():
    """Verify factory returns appropriate encoders and defaults to resnet50."""
    resnet_factory = get_visual_encoder("resnet50", feature_dim=512, pretrained=False)
    assert isinstance(resnet_factory, ResNet50VisualEncoder)

    clip_factory = get_visual_encoder("clip", feature_dim=512, pretrained=False)
    assert isinstance(clip_factory, CLIPExperimentalEncoder)

    with pytest.raises(ValueError):
        get_visual_encoder("unsupported_model_architecture")
