"""
Tests for Phase 2: Multimodal Inference Pipeline, Services, and Normalization.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.db.config import Settings
from src.models.fusion import GatedFusion
from src.services.multimodal_inference import (
    VideoEmbeddingService,
    AudioEmbeddingService,
    TextEmbeddingService,
    MultimodalInferenceService,
    MultimodalInferenceResult,
)


class DummyVideoEncoder(nn.Module):
    """Lightweight deterministic video encoder for testing without ImageNet download."""
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(3 * 224 * 224, feature_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, C, H, W)
        B, T = frames.shape[:2]
        flat = frames.reshape(B, T, -1).mean(dim=1)
        return self.proj(flat)


class DummyAudioEncoder(nn.Module):
    """Lightweight deterministic audio encoder for testing without Wav2Vec2 download."""
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(100, feature_dim)

    def forward(self, waveform: torch.Tensor, attention_mask=None) -> torch.Tensor:
        # waveform: (B, N)
        B = waveform.shape[0]
        # sample 100 points
        sampled = waveform[:, :100] if waveform.shape[1] >= 100 else F.pad(waveform, (0, 100 - waveform.shape[1]))
        return self.proj(sampled)


class DummyTextEncoder(nn.Module):
    """Lightweight deterministic text encoder for testing without BERT download."""
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(1, feature_dim)

    def forward(self, input_ids=None, attention_mask=None, texts=None) -> torch.Tensor:
        B = len(texts) if texts is not None else input_ids.shape[0]
        # create deterministic vector based on text lengths
        if texts is not None:
            lengths = torch.tensor([[float(len(t))] for t in texts], dtype=torch.float32)
        else:
            lengths = torch.ones(B, 1, dtype=torch.float32)
        return self.proj(lengths)


@pytest.fixture
def dummy_inference_service():
    """Create MultimodalInferenceService with dummy encoders and real GatedFusion."""
    feature_dim = 512
    device = torch.device("cpu")

    video_enc = DummyVideoEncoder(feature_dim)
    audio_enc = DummyAudioEncoder(feature_dim)
    text_enc = DummyTextEncoder(feature_dim)
    fusion = GatedFusion(feature_dim, feature_dim)
    fusion.eval()

    video_svc = VideoEmbeddingService(video_encoder=video_enc, device=device, feature_dim=feature_dim)
    audio_svc = AudioEmbeddingService(audio_encoder=audio_enc, device=device, feature_dim=feature_dim)
    text_svc = TextEmbeddingService(text_encoder=text_enc, device=device, feature_dim=feature_dim)

    service = MultimodalInferenceService.__new__(MultimodalInferenceService)
    service.settings = Settings(APP_ENV="development", ALLOW_UNINITIALIZED_MODEL=True)
    service.device = device
    service.feature_dim = feature_dim
    service.is_checkpoint_loaded = False
    service.video_service = video_svc
    service.audio_service = audio_svc
    service.text_service = text_svc
    service.fusion = fusion

    return service


def test_video_embedding_dimensions(dummy_inference_service):
    """Verify VideoEmbeddingService produces expected (1, 512) and (B, 512) shapes."""
    svc = dummy_inference_service.video_service

    # Single video frame tensor (8 frames, 3 channels, 224, 224)
    single_frames = torch.randn(8, 3, 224, 224)
    emb, ok = svc.encode(single_frames)
    assert ok is True
    assert emb.shape == (1, 512)

    # Batch of 2 videos
    batch_frames = torch.randn(2, 8, 3, 224, 224)
    emb_batch, ok = svc.encode(batch_frames)
    assert ok is True
    assert emb_batch.shape == (2, 512)


def test_audio_embedding_dimensions(dummy_inference_service):
    """Verify AudioEmbeddingService produces expected (1, 512) and (B, 512) shapes."""
    svc = dummy_inference_service.audio_service

    # Single waveform (80,000 samples)
    single_audio = torch.randn(80000)
    emb, ok = svc.encode(single_audio)
    assert ok is True
    assert emb.shape == (1, 512)

    # Batch of 2 waveforms
    batch_audio = torch.randn(2, 80000)
    emb_batch, ok = svc.encode(batch_audio)
    assert ok is True
    assert emb_batch.shape == (2, 512)


def test_text_embedding_dimensions_and_formats(dummy_inference_service):
    """Verify TextEmbeddingService supports strings, dicts, and multi-field inputs."""
    svc = dummy_inference_service.text_service

    # Plain string title
    emb_str, ok1 = svc.encode("Multimodal Deep Learning")
    assert ok1 is True
    assert emb_str.shape == (1, 512)

    # Structured dictionary
    metadata = {
        "title": "Multimodal Deep Learning",
        "description": "Video explaining ResNet, Wav2Vec2, and BERT fusion",
        "tags": ["AI", "PyTorch"],
        "transcript": "In this video we demonstrate multimodal architectures"
    }
    emb_dict, ok2 = svc.encode(metadata)
    assert ok2 is True
    assert emb_dict.shape == (1, 512)

    # List batch input
    emb_list, ok3 = svc.encode(["Video 1", metadata])
    assert ok3 is True
    assert emb_list.shape == (2, 512)


def test_fused_embedding_dimensions_and_l2_normalization(dummy_inference_service):
    """Verify GatedFusion outputs (1, 512) with unit L2 norm suitable for cosine similarity."""
    v = torch.randn(1, 512)
    a = torch.randn(1, 512)
    t = torch.randn(1, 512)

    fused = dummy_inference_service.fuse(v, a, t)
    assert fused.shape == (1, 512)

    # L2 norm must be approximately 1.0
    norm = torch.norm(fused, p=2, dim=-1).item()
    assert abs(norm - 1.0) < 1e-4


def test_deterministic_inference(dummy_inference_service):
    """Verify identical inputs produce numerically identical embeddings in eval mode."""
    frames = torch.ones(8, 3, 224, 224)
    audio = torch.ones(80000)
    text = {"title": "Deterministic Test Video"}

    res1 = dummy_inference_service.encode_multimodal(frames=frames, audio=audio, text=text)
    res2 = dummy_inference_service.encode_multimodal(frames=frames, audio=audio, text=text)

    assert torch.allclose(res1.embedding, res2.embedding, atol=1e-5)
    for v1, v2 in zip(res1.embedding_list, res2.embedding_list):
        assert abs(v1 - v2) < 1e-5


def test_missing_audio_handling(dummy_inference_service):
    """Verify system gracefully degrades when audio is missing or silent without crashing."""
    frames = torch.randn(8, 3, 224, 224)
    text = "Audio-less silent video"

    # Silent audio (all zeros)
    res = dummy_inference_service.encode_multimodal(frames=frames, audio=torch.zeros(80000), text=text)
    assert "video" in res.modalities_present
    assert "text" in res.modalities_present
    assert "audio" not in res.modalities_present
    assert res.embedding.shape == (1, 512)
    assert abs(torch.norm(res.embedding, p=2).item() - 1.0) < 1e-4


def test_missing_text_handling(dummy_inference_service):
    """Verify system handles empty text without crashing."""
    frames = torch.randn(8, 3, 224, 224)
    audio = torch.randn(80000)

    res = dummy_inference_service.encode_multimodal(frames=frames, audio=audio, text="")
    assert "video" in res.modalities_present
    assert "audio" in res.modalities_present
    assert "text" not in res.modalities_present
    assert res.embedding.shape == (1, 512)
    assert abs(torch.norm(res.embedding, p=2).item() - 1.0) < 1e-4


def test_missing_video_handling(dummy_inference_service):
    """Verify system handles absent video file without crashing."""
    audio = torch.randn(80000)
    text = "Audio podcast episode"

    res = dummy_inference_service.encode_multimodal(video_path="non_existent_file.mp4", audio=audio, text=text)
    assert "audio" in res.modalities_present
    assert "text" in res.modalities_present
    assert "video" not in res.modalities_present
    assert res.embedding.shape == (1, 512)
    assert abs(torch.norm(res.embedding, p=2).item() - 1.0) < 1e-4


def test_nan_inf_protection(dummy_inference_service):
    """Verify NaN and Inf inputs are cleaned and do not pollute the output embedding."""
    frames = torch.full((8, 3, 224, 224), float("nan"))
    audio = torch.full((80000,), float("inf"))
    text = "NaN resistance test"

    res = dummy_inference_service.encode_multimodal(frames=frames, audio=audio, text=text)
    assert not torch.isnan(res.embedding).any()
    assert not torch.isinf(res.embedding).any()


def test_checkpoint_policy_production_rejection():
    """Verify production mode strictly rejects missing project checkpoints."""
    prod_settings = Settings(
        APP_ENV="production",
        ALLOW_UNINITIALIZED_MODEL=False,
        MODEL_PATH="non_existent_checkpoint_file.pt"
    )

    with pytest.raises(RuntimeError, match="Production environment requires a valid checkpoint"):
        MultimodalInferenceService(settings=prod_settings)


def test_end_to_end_multimodal_inference_result(dummy_inference_service):
    """Verify complete MultimodalInferenceResult structure and properties."""
    frames = torch.randn(8, 3, 224, 224)
    audio = torch.randn(80000)
    text = {
        "title": "Cooking Tutorial",
        "description": "How to make homemade sourdough",
        "tags": ["Baking", "Food"]
    }

    result = dummy_inference_service.encode_multimodal(frames=frames, audio=audio, text=text)

    assert isinstance(result, MultimodalInferenceResult)
    assert result.embedding.shape == (1, 512)
    assert len(result.embedding_list) == 512
    assert all(isinstance(x, float) for x in result.embedding_list)
    assert result.modalities_present == ["video", "audio", "text"]
    assert result.is_checkpoint_loaded is False
    assert result.device == "cpu"
