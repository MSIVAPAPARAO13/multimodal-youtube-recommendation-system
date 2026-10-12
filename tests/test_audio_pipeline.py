"""
Tests for Audio Intelligence Pipeline: End-to-end integration, tuple unpacking, modality isolation, and speech recognition.
"""

from pathlib import Path
import numpy as np
import pytest
import scipy.io.wavfile as wavfile
import torch

from src.features.audio_encoder import AudioEncoder
from src.features.speech_recognizer import OptionalSpeechRecognizer, SpeechRecognitionResult
from src.services.multimodal_inference import (
    AudioEmbeddingService,
    AudioEmbeddingResult,
    MultimodalInferenceService,
    MultimodalInferenceResult,
)


@pytest.fixture
def mock_audio_service():
    """Create AudioEmbeddingService with local initialized encoder (no downloads)."""
    enc = AudioEncoder(feature_dim=512, pretrained=False)
    enc.eval()
    return AudioEmbeddingService(audio_encoder=enc, feature_dim=512)


def _create_wav(path: Path, sample_rate: int = 16000, duration_sec: float = 1.0) -> Path:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, sig)
    return path


def test_audio_pipeline_tuple_unpacking(mock_audio_service, tmp_path):
    """Test 1: Backward-compatible tuple unpacking: emb, ok = service.encode(...)."""
    wav_path = _create_wav(tmp_path / "valid.wav", duration_sec=1.0)
    result = mock_audio_service.encode(wav_path)

    # Tuple unpacking
    emb, ok = result
    assert ok is True
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (1, 512)
    assert pytest.approx(torch.norm(emb, p=2, dim=-1).item(), rel=1e-3) == 1.0


def test_audio_pipeline_result_structure(mock_audio_service, tmp_path):
    """Test 2: Structured AudioEmbeddingResult properties and to_dict()."""
    wav_path = _create_wav(tmp_path / "structure.wav", duration_sec=1.5)
    result = mock_audio_service.encode(wav_path)

    assert isinstance(result, AudioEmbeddingResult)
    assert result.is_available is True
    assert result.embedding_dimension == 512
    assert result.sample_rate == 16000
    assert result.duration_seconds > 1.0
    assert result.processing_time_ms >= 0.0
    assert "rms_energy" in result.acoustic_diagnostics

    d = result.to_dict()
    assert d["is_available"] is True
    assert d["embedding_dimension"] == 512
    assert "acoustic_diagnostics" in d
    assert isinstance(d["embedding"], list)
    assert len(d["embedding"]) == 512


def test_audio_pipeline_missing_file_handling(mock_audio_service):
    """Test 3: Missing audio file returns zero vector and is_available=False."""
    result = mock_audio_service.encode("non_existent_audio_path_xyz.wav")
    emb, ok = result

    assert ok is False
    assert (emb == 0.0).all()
    assert result.failure_reason is not None


def test_audio_pipeline_silence_handling(mock_audio_service):
    """Test 4: Purely silent audio returns zero vector with is_available=False."""
    silent = torch.zeros(16000)
    result = mock_audio_service.encode(silent)
    emb, ok = result

    assert ok is False
    assert (emb == 0.0).all()


def test_audio_pipeline_long_recording_chunking(mock_audio_service, tmp_path):
    """Test 5: Multi-chunk handling for long audio recordings."""
    # 25 seconds of audio (> 20s default chunk)
    wav_path = _create_wav(tmp_path / "long.wav", duration_sec=25.0)
    result = mock_audio_service.encode(wav_path)

    assert result.is_available is True
    assert result.num_chunks > 1
    assert result.duration_seconds >= 24.5
    assert result.embedding.shape == (1, 512)


def test_speech_recognizer_disabled_by_default():
    """Test 6: Speech recognition is disabled by default and does not crash."""
    recognizer = OptionalSpeechRecognizer(enabled=False)
    res = recognizer.transcribe(torch.randn(16000))

    assert isinstance(res, SpeechRecognitionResult)
    assert res.is_available is False
    assert res.text is None
    assert "disabled" in res.error_message.lower()


def test_speech_recognizer_preserve_existing_transcript():
    """Test 7: Speech recognizer policy preserves existing valid transcript."""
    assert OptionalSpeechRecognizer.should_transcribe("Existing high quality transcript") is False
    assert OptionalSpeechRecognizer.should_transcribe("") is True
    assert OptionalSpeechRecognizer.should_transcribe(None) is True
    assert OptionalSpeechRecognizer.should_transcribe("Existing", force=True) is True


@pytest.fixture
def dummy_inference_service():
    """Create lightweight MultimodalInferenceService for audio pipeline isolation tests."""
    from src.db.config import Settings
    from src.models.fusion import GatedFusion
    from src.services.multimodal_inference import VideoEmbeddingService, TextEmbeddingService

    class DummyVideoEnc(torch.nn.Module):
        def forward(self, frames):
            B = frames.shape[0] if frames.dim() == 5 else 1
            return torch.randn(B, 512)

    class DummyTextEnc(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, texts=None):
            B = len(texts) if texts is not None else 1
            return torch.randn(B, 512)

    class DummyAudioEnc(torch.nn.Module):
        def forward(self, waveform, attention_mask=None):
            B = waveform.shape[0] if waveform.dim() == 2 else 1
            return torch.randn(B, 512)

    device = torch.device("cpu")
    service = MultimodalInferenceService.__new__(MultimodalInferenceService)
    service.settings = Settings(APP_ENV="development", ALLOW_UNINITIALIZED_MODEL=True)
    service.device = device
    service.feature_dim = 512
    service.is_checkpoint_loaded = False
    service.video_service = VideoEmbeddingService(video_encoder=DummyVideoEnc(), device=device, feature_dim=512)
    service.audio_service = AudioEmbeddingService(audio_encoder=DummyAudioEnc(), device=device, feature_dim=512)
    service.text_service = TextEmbeddingService(text_encoder=DummyTextEnc(), device=device, feature_dim=512)
    service.fusion = GatedFusion(512, 512)
    service.fusion.eval()
    return service


def test_missing_audio_does_not_break_multimodal_inference(dummy_inference_service):
    """Test 8: Missing audio preserves text and video modalities."""
    frames = torch.randn(8, 3, 224, 224)
    text = "Machine Learning Overview"

    # Audio missing (None)
    result = dummy_inference_service.encode_multimodal(frames=frames, audio=None, text=text)

    assert isinstance(result, MultimodalInferenceResult)
    assert "video" in result.modalities_present
    assert "text" in result.modalities_present
    assert "audio" not in result.modalities_present
    assert (result.audio_embedding == 0.0).all()
    assert result.embedding.shape == (1, 512)
