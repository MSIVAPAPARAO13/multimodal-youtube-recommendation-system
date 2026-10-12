"""
Tests for End-to-End Visual Pipeline (Phase 6):
- Video-only visual embedding
- Thumbnail-only visual embedding
- Video + thumbnail visual fusion (70% video + 30% thumbnail default)
- Configurable weights
- Auto-discovery of thumbnail alongside video file
- L2 normalization on the unit sphere
- Robustness against NaN/Inf values
- Missing modality handling
- Integration with MultimodalInferenceService
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from src.db.config import Settings
from src.features.visual_encoder import ResNet50VisualEncoder
from src.services.multimodal_inference import (
    VideoEmbeddingService,
    VisualEmbeddingService,
    VisualEmbeddingResult,
    MultimodalInferenceService,
)
from tests.test_frame_sampler import create_synthetic_video


@pytest.fixture
def mock_visual_encoder():
    """Deterministic visual encoder with pretrained=False for fast offline test."""
    return ResNet50VisualEncoder(feature_dim=512, pretrained=False, freeze_backbone=True)


@pytest.fixture
def visual_service(mock_visual_encoder):
    """VisualEmbeddingService with mock ResNet50 encoder."""
    return VisualEmbeddingService(
        video_encoder=mock_visual_encoder,
        device=torch.device("cpu"),
        feature_dim=512,
        video_weight=0.7,
        thumbnail_weight=0.3,
    )


@pytest.fixture
def media_bundle(tmp_path):
    """Provides a synthetic video and matching thumbnail in a directory structure."""
    video_dir = tmp_path / "video_123"
    video_dir.mkdir()
    video_path = video_dir / "video.mp4"
    thumb_path = video_dir / "thumbnail.jpg"

    create_synthetic_video(video_path, num_frames=10)
    img = Image.new("RGB", (320, 180), color=(50, 100, 150))
    img.save(thumb_path)

    return video_path, thumb_path


def test_video_only_embedding(visual_service, tmp_path):
    """Passing video only produces normalized embedding with modality_status='video_only'."""
    standalone_video = tmp_path / "standalone.mp4"
    create_synthetic_video(standalone_video, num_frames=10)
    res = visual_service.embed_video(video_path=standalone_video, thumbnail_path=None)

    assert res.is_available is True
    assert res.modality_status == "video_only"
    assert res.video_frames_used > 0
    assert res.thumbnail_used is False
    assert res.embedding.shape == (1, 512)

    # L2 normalized
    norm = torch.norm(res.embedding, p=2).item()
    assert pytest.approx(norm, abs=1e-4) == 1.0


def test_thumbnail_only_embedding(visual_service, media_bundle):
    """Passing thumbnail only produces normalized embedding with modality_status='thumbnail_only'."""
    _, thumb_path = media_bundle
    res = visual_service.embed_video(video_path=None, thumbnail_path=thumb_path)

    assert res.is_available is True
    assert res.modality_status == "thumbnail_only"
    assert res.video_frames_used == 0
    assert res.thumbnail_used is True
    assert res.embedding.shape == (1, 512)

    norm = torch.norm(res.embedding, p=2).item()
    assert pytest.approx(norm, abs=1e-4) == 1.0


def test_video_and_thumbnail_visual_fusion(visual_service, media_bundle):
    """Passing both video and thumbnail executes weighted visual fusion and L2 normalization."""
    video_path, thumb_path = media_bundle
    res = visual_service.embed_video(video_path=video_path, thumbnail_path=thumb_path)

    assert res.is_available is True
    assert res.modality_status == "video+thumbnail"
    assert res.video_frames_used > 0
    assert res.thumbnail_used is True
    assert res.embedding.shape == (1, 512)

    norm = torch.norm(res.embedding, p=2).item()
    assert pytest.approx(norm, abs=1e-4) == 1.0

    # Verify mathematical fusion formula: normalize(0.7 * v + 0.3 * t)
    v_emb = res.video_embedding
    t_emb = res.thumbnail_embedding
    expected_fused = F.normalize(0.7 * v_emb + 0.3 * t_emb, p=2, dim=-1)
    torch.testing.assert_close(res.embedding, expected_fused)


def test_auto_discovery_of_sibling_thumbnail(visual_service, media_bundle):
    """When thumbnail_path is not specified, sibling thumbnail.jpg is auto-discovered."""
    video_path, thumb_path = media_bundle
    assert thumb_path.exists()

    res = visual_service.embed_video(video_path=video_path, thumbnail_path=None)
    assert res.modality_status == "video+thumbnail"
    assert res.thumbnail_used is True


def test_missing_both_modalities(visual_service, tmp_path):
    """When both video and thumbnail are missing, returns controlled zeros without crashing."""
    res = visual_service.embed_video(
        video_path=tmp_path / "non_existent.mp4",
        thumbnail_path=tmp_path / "non_existent.jpg",
    )

    assert res.is_available is False
    assert res.modality_status == "missing"
    assert res.embedding.shape == (1, 512)
    assert (res.embedding == 0).all()


def test_no_nan_or_inf_in_embeddings(visual_service):
    """Ensure no NaN or Inf values are produced even with extreme inputs."""
    extreme_tensor = torch.full((1, 8, 3, 224, 224), 1e6)
    res = visual_service.embed_video(frames_tensor=extreme_tensor)

    assert not torch.isnan(res.embedding).any()
    assert not torch.isinf(res.embedding).any()


def test_custom_fusion_weights(mock_visual_encoder, media_bundle):
    """Verify customizable visual video and thumbnail weights."""
    video_path, thumb_path = media_bundle
    service = VisualEmbeddingService(
        video_encoder=mock_visual_encoder,
        video_weight=0.9,
        thumbnail_weight=0.1,
    )
    res = service.embed_video(video_path=video_path, thumbnail_path=thumb_path)
    assert res.diagnostics["video_weight"] == 0.9
    assert res.diagnostics["thumbnail_weight"] == 0.1


def test_backward_compatible_unpacking(visual_service, media_bundle):
    """Calling encode(...) supports `emb, ok = svc.encode(...)` unpacking."""
    video_path, _ = media_bundle
    emb, ok = visual_service.encode(video_path)

    assert ok is True
    assert emb.shape == (1, 512)
