"""
Tests for ThumbnailEncoder (Phase 6):
- Local thumbnail embedding extraction
- Missing thumbnail handling
- Empty and corrupt thumbnail handling
- Direct tensor input support
- Backward-compatible tuple unpacking
- Structured diagnostics
"""

from pathlib import Path
from PIL import Image
import pytest
import torch

from src.features.visual_encoder import ResNet50VisualEncoder
from src.features.thumbnail_encoder import ThumbnailEncoder, ThumbnailEncodingResult


@pytest.fixture
def thumbnail_encoder():
    encoder = ResNet50VisualEncoder(feature_dim=512, pretrained=False, freeze_backbone=True)
    return ThumbnailEncoder(encoder=encoder, feature_dim=512)


@pytest.fixture
def valid_thumbnail(tmp_path):
    path = tmp_path / "valid_thumb.jpg"
    img = Image.new("RGB", (640, 360), color=(120, 200, 80))
    img.save(path)
    return path


def test_valid_thumbnail_embedding(thumbnail_encoder, valid_thumbnail):
    """Valid local thumbnail extracts a 512-D embedding with positive availability flags."""
    res = thumbnail_encoder.encode_thumbnail(valid_thumbnail)

    assert isinstance(res, ThumbnailEncodingResult)
    assert res.is_available is True
    assert res.thumbnail_available is True
    assert res.thumbnail_path == str(valid_thumbnail)
    assert res.embedding.shape == (1, 512)
    assert not torch.isnan(res.embedding).any()
    assert not torch.isinf(res.embedding).any()


def test_missing_thumbnail_handling(thumbnail_encoder, tmp_path):
    """Missing thumbnail path returns zeros without raising exception."""
    missing_path = tmp_path / "does_not_exist.jpg"

    res = thumbnail_encoder.encode_thumbnail(missing_path)

    assert res.is_available is False
    assert res.thumbnail_available is False
    assert res.embedding.shape == (1, 512)
    assert (res.embedding == 0).all()
    assert "not found" in (res.error_message or "").lower()


def test_none_thumbnail_handling(thumbnail_encoder):
    """None input returns zeros gracefully."""
    res = thumbnail_encoder.encode_thumbnail(None)

    assert res.is_available is False
    assert res.thumbnail_available is False
    assert res.embedding.shape == (1, 512)
    assert (res.embedding == 0).all()


def test_empty_thumbnail_handling(thumbnail_encoder, tmp_path):
    """Empty 0-byte file returns zeros gracefully."""
    empty_path = tmp_path / "empty_thumb.jpg"
    empty_path.touch()

    res = thumbnail_encoder.encode_thumbnail(empty_path)

    assert res.is_available is False
    assert (res.embedding == 0).all()
    assert "empty" in (res.error_message or "").lower()


def test_corrupt_thumbnail_handling(thumbnail_encoder, tmp_path):
    """Corrupted thumbnail image returns zeros with informative error message."""
    corrupt_path = tmp_path / "corrupt_thumb.jpg"
    corrupt_path.write_bytes(b"Not a valid JPEG/PNG file header.")

    res = thumbnail_encoder.encode_thumbnail(corrupt_path)

    assert res.is_available is False
    assert (res.embedding == 0).all()
    assert res.error_message is not None


def test_tensor_input_direct_encoding(thumbnail_encoder):
    """Passing a preprocessed (1, 3, 224, 224) tensor directly encodes successfully."""
    tensor = torch.randn(1, 3, 224, 224)
    res = thumbnail_encoder.encode_thumbnail(tensor)

    assert res.is_available is True
    assert res.embedding.shape == (1, 512)


def test_thumbnail_unpacking_and_to_dict(thumbnail_encoder, valid_thumbnail):
    """Verify backwards-compatible unpacking (emb, ok = ...) and dictionary serialization."""
    emb, ok = thumbnail_encoder.encode(valid_thumbnail)

    assert ok is True
    assert emb.shape == (1, 512)

    res = thumbnail_encoder.encode(valid_thumbnail)
    diag = res.to_dict()

    assert diag["is_available"] is True
    assert diag["thumbnail_available"] is True
    assert len(diag["embedding"]) == 512
