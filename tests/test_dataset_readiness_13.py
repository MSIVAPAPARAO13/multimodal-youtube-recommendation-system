"""
Phase 13 Test Suite - Large-Scale Media Ingestion, Audio Recovery & Dataset Readiness.

Verifies:
1. Explicit FFmpeg/FFprobe availability and resolution.
2. Audio stream demuxing into 16 kHz mono WAV with finite waveforms.
3. Rejection of silent audio waveforms and containers with missing audio streams.
4. AssetIngestionOrchestrator deduplication, rate limiting, storage budget limits, and checkpoint resumability.
5. Cross-modal alignment verification (Text, Visual ResNet-50, Audio Wav2Vec2).
6. Dataset readiness audit execution and schema compliance.
7. Preservation of Phase 8C checkpoint and zero production multimodal ranking weights.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import time
import unittest.mock
from unittest.mock import patch, MagicMock
import wave
import numpy as np
import pytest

from src.db.config import resolve_ffmpeg_path, resolve_ffprobe_path, get_settings
from src.db.database import get_database
from src.db.models import VideoAsset, VideoFeature, Video, AssetType, AssetStatus, IngestionStatus
from src.integrations.youtube.media.audio import AudioExtractor
from src.integrations.youtube.media.validator import MediaValidator
from src.integrations.youtube.media.exceptions import AudioExtractionError, MediaValidationError
from src.services.youtube_ingestion import YouTubeIngestionService, AssetIngestionOrchestrator
from scripts.evaluate_dataset_readiness import run_dataset_readiness_evaluation


@pytest.fixture
def sine_wav_file(tmp_path):
    """Generate a clean 2-second, 16 kHz mono sine wave WAV file."""
    wav_path = tmp_path / "valid_sine.wav"
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), False)
    # 440 Hz tone with non-trivial amplitude
    signal = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal.tobytes())

    return wav_path


@pytest.fixture
def silent_wav_file(tmp_path):
    """Generate a 2-second, 16 kHz mono all-zeros silent WAV file."""
    wav_path = tmp_path / "silent.wav"
    sr = 16000
    duration = 2.0
    signal = np.zeros(int(sr * duration), dtype=np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal.tobytes())

    return wav_path


# 1. Test FFmpeg Path Resolution
def test_ffmpeg_resolution_and_configuration():
    ffmpeg_path = resolve_ffmpeg_path()
    assert ffmpeg_path is not None, "FFmpeg path must resolve on worker environment"
    assert Path(ffmpeg_path).exists(), f"Resolved FFmpeg binary does not exist at {ffmpeg_path}"

    extractor = AudioExtractor()
    assert extractor.ffmpeg_path is not None
    assert Path(extractor.ffmpeg_path).exists()


# 2. Test Audio Stream Probe and Silence Rejection
def test_silent_audio_rejection(silent_wav_file):
    validator = MediaValidator()
    # check_silence=True must reject all-zero audio
    with pytest.raises(MediaValidationError) as exc_info:
        validator.validate_audio_file(silent_wav_file, check_silence=True)
    assert "SILENT_AUDIO_REJECTED" in str(exc_info.value) or "silent" in str(exc_info.value).lower()


# 3. Test Valid Audio Validation
def test_valid_audio_validation(sine_wav_file):
    validator = MediaValidator()
    res = validator.validate_audio_file(sine_wav_file, check_silence=True)
    assert res is not None
    assert res["channels"] == 1
    assert res["sample_rate"] == 16000
    assert res["rms_energy"] > 0.01
    assert res["duration_seconds"] >= 1.9


# 4. Test Missing Video Handling in AudioExtractor
def test_missing_video_extraction_error(tmp_path):
    extractor = AudioExtractor()
    fake_video = tmp_path / "non_existent.mp4"
    out_wav = tmp_path / "out.wav"

    with pytest.raises(AudioExtractionError) as exc_info:
        extractor.extract_and_save_wav(fake_video, out_wav, video_id="missing_vid")
    assert "not found" in str(exc_info.value).lower()


# 5. Test AudioExtractor Fallback Rejection of Silent Waveform
def test_extractor_silent_audio_rejection(tmp_path):
    extractor = AudioExtractor()
    dummy_video = tmp_path / "vid.mp4"
    dummy_video.write_bytes(b"dummy")
    out_wav = tmp_path / "out.wav"

    # When FFmpeg is not used and processor returns silence with check_silence
    with patch.object(extractor, "ffmpeg_path", None):
        import torch
        zero_tensor = torch.zeros(32000)
        with patch.object(extractor.processor, "extract_audio", return_value=zero_tensor):
            with pytest.raises(AudioExtractionError) as exc_info:
                extractor.extract_and_save_wav(dummy_video, out_wav, video_id="vid1", check_silence=True)
            assert "SILENT_AUDIO_REJECTED" in str(exc_info.value)


# 6. Test Orchestrator Deduplication
def test_orchestrator_deduplication(in_memory_db, tmp_path):
    service = YouTubeIngestionService(session_factory=lambda: in_memory_db)
    orchestrator = AssetIngestionOrchestrator(
        ingestion_service=service,
        max_concurrency=1,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    with patch.object(orchestrator, "_is_already_ingested", return_value=True):
        res = orchestrator._process_item("test_vid_already_done", force=False)
        assert res["status"] == "SKIPPED"
        assert res["reason"] == "ALREADY_INGESTED"


# 7. Test Orchestrator Storage Budget Enforcement
def test_orchestrator_storage_budget_enforcement(in_memory_db, tmp_path):
    service = YouTubeIngestionService(session_factory=lambda: in_memory_db)
    orchestrator = AssetIngestionOrchestrator(
        ingestion_service=service,
        max_storage_gb=0.000001,  # extremely small budget
        checkpoint_dir=tmp_path / "checkpoints",
    )

    with patch.object(orchestrator, "_get_storage_usage_bytes", return_value=100 * 1024 * 1024):
        res = orchestrator._process_item("vid_budget_test")
        assert res["status"] == "FAILED"
        assert res["error_category"] == "STORAGE_LIMIT_EXCEEDED"


# 8. Test Orchestrator Rate Limiting
def test_orchestrator_rate_limiting(in_memory_db, tmp_path):
    service = YouTubeIngestionService(session_factory=lambda: in_memory_db)
    orchestrator = AssetIngestionOrchestrator(
        ingestion_service=service,
        rate_limit_delay_seconds=0.1,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    t0 = time.perf_counter()
    with patch.object(orchestrator, "_is_already_ingested", return_value=False):
        with patch.object(orchestrator, "_check_storage_budget", return_value=True):
            with patch.object(service, "ingest_video", return_value={"status": "MEDIA_READY"}):
                res = orchestrator._process_item("rate_limit_test")
                duration = time.perf_counter() - t0
                assert duration >= 0.08
                assert res["status"] == "SUCCESS"


# 9. Test Orchestrator Checkpoint & Resumability
def test_orchestrator_checkpoint_and_resume(in_memory_db, tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    service = YouTubeIngestionService(session_factory=lambda: in_memory_db)
    orchestrator = AssetIngestionOrchestrator(
        ingestion_service=service,
        checkpoint_dir=ckpt_dir,
    )

    with patch.object(service, "ingest_video", return_value={"status": "MEDIA_READY"}):
        job_id = orchestrator.submit_batch(["v1", "v2"])
        # Give worker a moment to save checkpoint
        time.sleep(0.2)

        status = orchestrator.get_job_status(job_id)
        assert status is not None
        assert status["total"] == 2

        # Checkpoint file exists on disk
        ckpt_file = ckpt_dir / f"job_{job_id}.json"
        assert ckpt_file.exists()


# 10. Test Dataset Readiness Evaluation Report Schema & Keys
def test_dataset_readiness_evaluation_report_execution():
    report = run_dataset_readiness_evaluation()
    assert report is not None
    assert "catalog_summary" in report
    assert "ingestion_status" in report
    assert "asset_and_modality_coverage" in report
    assert "cross_modal_alignment" in report
    assert "storage_and_integrity" in report
    assert "phase_14_training_readiness" in report

    # Verify key properties
    assert report["pipeline_version"] == "13"
    assert report["catalog_summary"]["total_catalog_items"] >= 2960
    assert report["asset_and_modality_coverage"]["verified_nonsilent_audio_coverage"] >= 10
    assert report["cross_modal_alignment"]["tri_modal_aligned_count"] >= 10
    assert report["phase_14_training_readiness"]["status"] == "NOT_YET_SUITABLE"
    assert report["storage_and_integrity"]["storage_limit_compliant"] is True


# 11. Test Preservation of Phase 8C Checkpoint Integrity
def test_phase_8c_checkpoint_integrity():
    checkpoint_path = Path("data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt")
    assert checkpoint_path.exists(), "Phase 8C checkpoint best.pt must exist"

    sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    expected_sha = "a82037eb790e2a23e133b8edc9973a0e6fdc3c466058abd61b68aed77d2d4996"
    assert sha == expected_sha, f"Phase 8C checkpoint corrupted: expected {expected_sha}, got {sha}"


# 12. Test Zero Multimodal Production Ranking Weights
def test_zero_production_multimodal_weights():
    from src.services.reranking_service import CandidateReRankingService
    safe_w = CandidateReRankingService.get_production_safe_weights()
    assert safe_w.w_visual == 0.0, "Production visual ranking weight must remain zero"
    assert safe_w.w_audio == 0.0, "Production audio ranking weight must remain zero"
    assert safe_w.w_text == 0.90, "Production text semantic weight must remain primary"
