"""
Phase 12A Tests - Multimodal Asset Ingestion, Alignment, and Validation.
Tests cover:
- VideoAsset database model & repository operations
- Audio format (16kHz mono WAV), channels, sample rate, and integrity validation
- FrameSampler deterministic frame extraction, timestamps, and checksums
- Local media ingestion and asset alignment
- Idempotent ingestion and duplicate prevention
- Bounded batch job orchestrator with failure isolation
- FastAPI endpoints for asset listing, batch submission, and asset details
"""

import io
from pathlib import Path
import wave
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.db.models import Video, VideoIngestion, VideoAsset, AssetType, AssetStatus, IngestionStatus
from src.db.repository import VideoAssetRepository, YouTubeVideoRepository
from src.integrations.youtube.media.validator import MediaValidator
from src.integrations.youtube.media.exceptions import MediaValidationError, MediaCorruptError
from src.features.frame_sampler import FrameSampler
from src.services.youtube_ingestion import YouTubeIngestionService, AssetIngestionOrchestrator


@pytest.fixture
def sample_wav_file(tmp_path):
    """Generate a valid 1-second 16kHz mono WAV file."""
    wav_path = tmp_path / "test_audio.wav"
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    # 440 Hz sine tone
    t = np.linspace(0, duration, num_samples, False)
    signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())

    return wav_path


@pytest.fixture
def stereo_wav_file(tmp_path):
    """Generate a 44.1kHz stereo WAV file."""
    wav_path = tmp_path / "stereo_audio.wav"
    sample_rate = 44100
    duration = 0.5
    num_samples = int(sample_rate * duration)
    signal = np.zeros(num_samples * 2, dtype=np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())

    return wav_path


# 1. Test VideoAsset Model and Repository
def test_video_asset_repository_operations(in_memory_db):
    repo = VideoAssetRepository(in_memory_db)

    asset = repo.upsert_asset(
        video_id="video_test_1",
        asset_type=AssetType.VIDEO,
        storage_path="test/video.mp4",
        youtube_video_id="abc12345678",
        mime_type="video/mp4",
        file_size_bytes=102400,
        checksum_sha256="abcdef123456",
        status=AssetStatus.READY,
        pipeline_version="12A",
    )

    assert asset is not None
    assert asset.video_id == "video_test_1"
    assert asset.asset_type == "VIDEO"
    assert asset.checksum_sha256 == "abcdef123456"

    # Query
    assets = repo.get_assets_for_video("video_test_1")
    assert len(assets) == 1
    assert assets[0].storage_path == "test/video.mp4"

    # Query by type
    fetched = repo.get_asset_by_type("video_test_1", AssetType.VIDEO)
    assert fetched is not None
    assert fetched.id == asset.id

    # Query by checksum
    by_hash = repo.get_by_checksum("abcdef123456")
    assert by_hash is not None
    assert by_hash.video_id == "video_test_1"

    # Serialization
    asset_dict = asset.to_dict()
    assert asset_dict["video_id"] == "video_test_1"
    assert asset_dict["asset_type"] == "VIDEO"
    assert asset_dict["pipeline_version"] == "12A"


# 2. Test Audio Validation on Valid 16kHz Mono WAV
def test_audio_validation_valid_wav(sample_wav_file):
    validator = MediaValidator()
    result = validator.validate_audio_file(
        sample_wav_file,
        video_id="test_video",
        expected_sample_rate=16000,
        expected_channels=1,
    )
    assert result["channels"] == 1
    assert result["sample_rate"] == 16000
    assert result["duration_seconds"] == 1.0
    assert result["frames"] == 16000
    assert result["size_bytes"] > 0


# 3. Test Audio Validation Rejects Channel Mismatch
def test_audio_validation_channel_mismatch(stereo_wav_file):
    validator = MediaValidator()
    with pytest.raises(MediaCorruptError) as exc_info:
        validator.validate_audio_file(
            stereo_wav_file,
            video_id="test_video",
            expected_sample_rate=None,
            expected_channels=1,
        )
    assert "channels mismatch" in str(exc_info.value).lower()


# 4. Test Audio Validation Rejects Sample Rate Mismatch
def test_audio_validation_sample_rate_mismatch(stereo_wav_file):
    validator = MediaValidator()
    with pytest.raises(MediaCorruptError) as exc_info:
        validator.validate_audio_file(
            stereo_wav_file,
            video_id="test_video",
            expected_sample_rate=16000,
            expected_channels=None,
        )
    assert "sample rate mismatch" in str(exc_info.value).lower()


# 5. Test Audio Validation Rejects Empty or Missing File
def test_audio_validation_missing_and_empty(tmp_path):
    validator = MediaValidator()
    with pytest.raises(MediaValidationError):
        validator.validate_audio_file(tmp_path / "nonexistent.wav")

    empty_wav = tmp_path / "empty.wav"
    empty_wav.write_bytes(b"")
    with pytest.raises(MediaCorruptError):
        validator.validate_audio_file(empty_wav)


# 6. Test SHA-256 Checksum Computation
def test_compute_sha256(tmp_path):
    test_file = tmp_path / "data.bin"
    test_file.write_bytes(b"hello world multimodal")
    checksum = MediaValidator.compute_sha256(test_file)
    assert len(checksum) == 64
    assert isinstance(checksum, str)
    # Verify deterministic
    assert MediaValidator.compute_sha256(test_file) == checksum


# 7. Test Frame Extraction and Checksumming
def test_frame_sampler_extract_and_save(tmp_path):
    sampler = FrameSampler(default_num_frames=4)
    # Mock sample_frames to return synthetic numpy frames
    mock_frame = np.ones((100, 100, 3), dtype=np.uint8) * 120
    from src.features.frame_sampler import FrameSamplingResult

    mock_result = FrameSamplingResult(
        frames=[mock_frame, mock_frame],
        requested_frames=2,
        decoded_frames=2,
        frame_indices=[0, 30],
        total_frames=60,
        fps=30.0,
        duration_seconds=2.0,
        video_path="mock.mp4",
        success=True,
    )

    with patch.object(sampler, "sample_frames", return_value=mock_result):
        out_dir = tmp_path / "frames"
        frames_meta = sampler.extract_and_save_frames("dummy.mp4", out_dir, num_frames=2)
        assert len(frames_meta) == 2
        assert frames_meta[0]["frame_index"] == 0
        assert frames_meta[0]["timestamp_seconds"] == 0.0
        assert frames_meta[1]["timestamp_seconds"] == 1.0
        assert (out_dir / "frame_000.jpg").exists()
        assert len(frames_meta[0]["checksum_sha256"]) == 64


# 8. Test Local Media Ingestion and Alignment
def test_local_video_ingestion_workflow(in_memory_db, tmp_path):
    service = YouTubeIngestionService(session_factory=lambda: in_memory_db)

    # Use existing synthetic fixture video
    fixture_video = Path("data/fixtures/synthetic_visual_eval/programming_0/video.mp4")
    if not fixture_video.exists():
        pytest.skip("Fixture video not present")

    res = service.ingest_local_video(
        video_id="prog_test_0",
        local_video_path=fixture_video,
        title="Programming Test Video",
    )

    assert res["status"] == "COMPLETED"
    assert res["video_id"] == "prog_test_0"
    assert res["frames_count"] > 0

    # Verify registered assets in repository
    asset_repo = VideoAssetRepository(in_memory_db)
    assets = asset_repo.get_assets_for_video("prog_test_0")
    assert len(assets) > 0
    asset_types = {a.asset_type for a in assets}
    assert "VIDEO" in asset_types
    assert "FRAME" in asset_types
    assert "METADATA" in asset_types


# 9. Test Ingestion Idempotency (Duplicate Calls Don't Duplicate Records)
def test_local_video_ingestion_idempotency(in_memory_db):
    service = YouTubeIngestionService(session_factory=lambda: in_memory_db)
    fixture_video = Path("data/fixtures/synthetic_visual_eval/programming_0/video.mp4")
    if not fixture_video.exists():
        pytest.skip("Fixture video not present")

    # Ingest once
    service.ingest_local_video(video_id="prog_idem_1", local_video_path=fixture_video)
    asset_repo = VideoAssetRepository(in_memory_db)
    count_first = len(asset_repo.get_assets_for_video("prog_idem_1"))

    # Ingest second time
    service.ingest_local_video(video_id="prog_idem_1", local_video_path=fixture_video)
    count_second = len(asset_repo.get_assets_for_video("prog_idem_1"))

    assert count_first == count_second


# 10. Test Ingestion Fails on Missing Local File
def test_local_video_ingestion_missing_file(in_memory_db):
    service = YouTubeIngestionService(session_factory=lambda: in_memory_db)
    with pytest.raises(MediaValidationError):
        service.ingest_local_video("missing_vid", "data/nonexistent/missing.mp4")


# 11. Test Batch Orchestrator Concurrency and Isolation
def test_batch_orchestrator_execution(in_memory_db):
    service = YouTubeIngestionService(session_factory=lambda: in_memory_db)
    orchestrator = AssetIngestionOrchestrator(
        ingestion_service=service,
        max_concurrency=2,
        batch_size=5,
    )

    with patch.object(service, "ingest_video", side_effect=[
        {"status": "MEDIA_READY", "video_id": "v1"},
        ValueError("Invalid format"),
        {"status": "MEDIA_READY", "video_id": "v3"},
    ]):
        job_id = orchestrator.submit_batch(["v1", "v2", "v3"])
        assert isinstance(job_id, str)
        status = orchestrator.get_job_status(job_id)
        assert status is not None
        assert status["total"] == 3


# 12. Test API GET /api/youtube/ingest (Listing)
def test_api_list_ingestions(client, in_memory_db):
    yt_repo = YouTubeVideoRepository(in_memory_db)
    yt_repo.get_or_create_ingestion(youtube_video_id="yt_test_01", video_id="v01")
    yt_repo.get_or_create_ingestion(youtube_video_id="yt_test_02", video_id="v02")

    response = client.get("/api/youtube/ingest?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 2
    assert len(data["items"]) >= 2


# 13. Test API GET /api/youtube/assets/{video_id}
def test_api_get_video_assets(client, in_memory_db):
    asset_repo = VideoAssetRepository(in_memory_db)
    asset_repo.upsert_asset(
        video_id="vid_api_test",
        asset_type=AssetType.VIDEO,
        storage_path="vid_api_test/video.mp4",
        file_size_bytes=5000,
        checksum_sha256="abc123sha",
    )

    response = client.get("/api/youtube/assets/vid_api_test")
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "vid_api_test"
    assert data["total_assets"] == 1
    assert data["assets"][0]["asset_type"] == "VIDEO"
    assert data["assets"][0]["checksum_sha256"] == "abc123sha"


# 14. Test API POST /api/youtube/ingest/batch
def test_api_batch_ingest(client):
    payload = {
        "video_ids": ["dQw4w9WgXcQ", "abc123def45"],
        "max_concurrency": 2
    }
    with patch("src.services.youtube_ingestion.AssetIngestionOrchestrator.submit_batch", return_value="job12345"):
        response = client.post("/api/youtube/ingest/batch", json=payload)
        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == "job12345"
        assert data["total_videos"] == 2
        assert data["status"] == "RUNNING"


# 15. Test API POST /api/youtube/ingest/batch Bounds Validation
def test_api_batch_ingest_bounds(client):
    # Payload exceeding 50 items should be rejected by Pydantic validation
    payload = {
        "video_ids": [f"vid_{i}" for i in range(55)],
    }
    response = client.post("/api/youtube/ingest/batch", json=payload)
    assert response.status_code == 422


# 16. Test Coverage Audit Execution
def test_coverage_audit_script_execution():
    from scripts.audit_multimodal_asset_coverage import run_asset_coverage_audit
    report = run_asset_coverage_audit()
    assert report is not None
    assert "catalog_summary" in report
    assert "asset_coverage_summary" in report
    assert "storage_and_integrity" in report
    assert report["pipeline_version"] == "12A"
