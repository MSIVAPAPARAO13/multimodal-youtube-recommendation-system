"""
Tests for Acoustic Feature Intelligence: Deterministic signal statistics, RMS energy, and silence ratio.
"""

import numpy as np
import pytest
import torch

from src.features.audio_processor import AudioProcessor


def test_rms_energy_calculation():
    """Verify RMS energy calculation on synthetic waveforms."""
    processor = AudioProcessor(target_sample_rate=16000)

    # 1. Zero amplitude waveform has 0.0 RMS
    zeros = torch.zeros(16000)
    res_zeros = processor.process_waveform(zeros)
    assert res_zeros.rms_energy == 0.0

    # 2. Constant amplitude 0.5 has RMS = 0.5
    constant = torch.full((16000,), 0.5)
    res_const = processor.process_waveform(constant)
    assert pytest.approx(res_const.rms_energy, rel=1e-3) == 0.5

    # 3. Sine wave with peak amplitude A has RMS = A / sqrt(2) approx 0.707 * A
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = 0.8 * np.sin(2 * np.pi * 440.0 * t)
    res_sine = processor.process_waveform(sine)
    expected_rms = 0.8 / np.sqrt(2)
    assert pytest.approx(res_sine.rms_energy, rel=1e-2) == expected_rms


def test_silence_ratio_calculation():
    """Verify silence ratio computation across quiet and active signals."""
    processor = AudioProcessor(target_sample_rate=16000, silence_threshold=1e-3)

    # Half silence (8000 zeros), half active sine wave
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    half_active = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    half_silent = np.zeros(8000)
    combined = np.concatenate([half_silent, half_active])

    res = processor.process_waveform(combined)
    # Approximately 50% silence
    assert pytest.approx(res.silence_ratio, abs=0.05) == 0.5


def test_acoustic_diagnostics_dict_sanitization(tmp_path):
    """Verify diagnostics dictionary contains clean serializable metrics."""
    processor = AudioProcessor(target_sample_rate=16000)
    waveform = torch.randn(16000) * 0.1

    res = processor.process_waveform(waveform)
    diag = res.to_dict()

    assert "sample_rate" in diag
    assert "duration_seconds" in diag
    assert "rms_energy" in diag
    assert "silence_ratio" in diag
    assert "is_clipped" in diag
    assert isinstance(diag["rms_energy"], float)
    assert isinstance(diag["silence_ratio"], float)
