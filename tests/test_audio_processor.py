"""
Tests for AudioProcessor: Input validation, mono conversion, resampling, and acoustic features.
"""

import io
from pathlib import Path
import wave
import numpy as np
import pytest
import scipy.io.wavfile as wavfile
import torch

from src.features.audio_processor import AudioProcessor, AudioProcessingResult


def _create_wav_file(path: Path, sample_rate: int, duration_sec: float, channels: int = 1, freq: float = 440.0) -> Path:
    """Helper to synthesize a clean sine wave WAV file for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * freq * t)
    
    if channels == 1:
        data = (sig * 32767).astype(np.int16)
    else:
        # Stereo: L and R channels
        stereo = np.column_stack([sig, sig * 0.5])
        data = (stereo * 32767).astype(np.int16)
        
    wavfile.write(str(path), sample_rate, data)
    return path


def test_valid_wav_processing(tmp_path):
    """Test 1: Valid WAV loading, decoding, normalization, and properties."""
    wav_path = _create_wav_file(tmp_path / "test_16k.wav", sample_rate=16000, duration_sec=1.5, channels=1)
    processor = AudioProcessor(target_sample_rate=16000)
    
    res = processor.process_file(wav_path)
    assert res.is_valid is True
    assert res.is_silent is False
    assert res.sample_rate == 16000
    assert pytest.approx(res.duration_seconds, rel=1e-2) == 1.5
    assert res.waveform.ndim == 1
    assert len(res.waveform) == 24000
    assert res.waveform.abs().max() <= 1.0
    assert res.rms_energy > 0.0


def test_stereo_to_mono_conversion(tmp_path):
    """Test 2: Deterministic stereo-to-mono downmixing: (L + R) / 2."""
    wav_path = _create_wav_file(tmp_path / "stereo_44k.wav", sample_rate=16000, duration_sec=1.0, channels=2)
    processor = AudioProcessor(target_sample_rate=16000)
    
    res = processor.process_file(wav_path)
    assert res.is_valid is True
    assert res.original_channels == 2
    assert res.waveform.ndim == 1
    assert len(res.waveform) == 16000


def test_resampling_to_16khz(tmp_path):
    """Test 3: Resampling from 44.1 kHz, 48 kHz, and 8 kHz to target 16 kHz."""
    processor = AudioProcessor(target_sample_rate=16000)
    
    for sr in [8000, 44100, 48000]:
        wav_path = _create_wav_file(tmp_path / f"test_{sr}.wav", sample_rate=sr, duration_sec=1.0)
        res = processor.process_file(wav_path)
        assert res.is_valid is True
        assert res.sample_rate == 16000
        assert res.original_sample_rate == sr
        # Tolerating resampling boundary effects within 50 samples
        assert abs(len(res.waveform) - 16000) <= 50


def test_empty_audio_handling(tmp_path):
    """Test 4: Empty (0-byte) file and empty tensor handling."""
    empty_file = tmp_path / "empty.wav"
    empty_file.touch()
    
    processor = AudioProcessor(target_sample_rate=16000)
    res = processor.process_file(empty_file)
    assert res.is_valid is False
    assert res.is_silent is True
    assert "empty" in res.error_message.lower()
    
    # Empty tensor
    res_tensor = processor.process_waveform(torch.tensor([]))
    assert res_tensor.is_valid is False


def test_corrupted_audio_handling(tmp_path):
    """Test 5: Corrupted or non-audio file handling."""
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(b"NOT_A_VALID_WAV_HEADER_DATA_1234567890")
    
    processor = AudioProcessor(target_sample_rate=16000)
    res = processor.process_file(corrupt_file)
    assert res.is_valid is False
    assert res.error_message is not None


def test_missing_file_handling():
    """Test 6: Non-existent audio file handling."""
    processor = AudioProcessor(target_sample_rate=16000)
    res = processor.process_file(Path("non_existent_folder/missing_audio.wav"))
    assert res.is_valid is False
    assert "not found" in res.error_message.lower()


def test_silent_waveform_handling():
    """Test 7: Purely silent audio detection without crashing."""
    processor = AudioProcessor(target_sample_rate=16000)
    silence = torch.zeros(16000, dtype=torch.float32)
    
    res = processor.process_waveform(silence)
    assert res.is_valid is False
    assert res.is_silent is True
    assert res.rms_energy == 0.0
    assert res.silence_ratio == 1.0


def test_non_finite_waveform_handling():
    """Test 8: Rejection or cleansing of NaN and Infinite values."""
    processor = AudioProcessor(target_sample_rate=16000)
    bad_waveform = torch.tensor([0.1, float("nan"), 0.2, float("inf"), -float("inf"), 0.3], dtype=torch.float32)
    
    res = processor.process_waveform(bad_waveform)
    assert res.is_valid is True
    assert not torch.isnan(res.waveform).any()
    assert not torch.isinf(res.waveform).any()


def test_duration_limit_enforcement():
    """Test 9: Enforcing maximum duration limit."""
    processor = AudioProcessor(target_sample_rate=16000, max_duration_seconds=2.0)
    # 5 seconds waveform
    long_waveform = torch.randn(16000 * 5)
    
    res = processor.process_waveform(long_waveform)
    assert res.is_truncated is True
    assert res.duration_seconds <= 2.05
    assert len(res.waveform) <= 32000


def test_clipping_detection():
    """Test 10: Detection of clipped waveforms exceeding clipping threshold."""
    processor = AudioProcessor(target_sample_rate=16000, clipping_threshold=0.95)
    clipped = torch.full((16000,), 1.0, dtype=torch.float32)
    
    res = processor.process_waveform(clipped)
    assert res.is_clipped is True
