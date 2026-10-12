"""
Unit and integration tests for Phase 4 YouTube Ingestion & Media Pipeline.
Mocks all external network requests, yt-dlp, and ffmpeg calls.
"""

from datetime import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

from src.db.config import Settings
from src.db.models import IngestionStatus, Video, VideoIngestion
from src.db.repository import YouTubeVideoRepository
from src.storage.local import LocalStorage, StorageSecurityError
from src.integrations.youtube.schemas import (
    YouTubeVideoMetadata,
    YouTubeThumbnails,
    YouTubeStatistics,
)
from src.integrations.youtube.media.exceptions import (
    MediaCorruptError,
    MediaDownloadError,
    MediaDurationExceededError,
    MediaSizeExceededError,
    MediaValidationError,
    ThumbnailDownloadError,
)
from src.integrations.youtube.media.validator import MediaValidator
from src.integrations.youtube.media.thumbnails import ThumbnailDownloader
from src.integrations.youtube.media.audio import AudioExtractor
from src.integrations.youtube.transcript import (
    YouTubeTranscriptProvider,
    TranscriptResult,
)
from src.services.youtube_ingestion import YouTubeIngestionService
from src.integrations.youtube.exceptions import (
    YouTubeInvalidVideoIdError,
    YouTubeNotFoundError,
    YouTubePrivateVideoError,
)


@pytest.fixture
def mock_metadata():
    """Sample normalized YouTube video metadata."""
    return YouTubeVideoMetadata(
        video_id="dQw4w9WgXcQ",
        title="Never Gonna Give You Up",
        description="Official Rick Astley Music Video",
        channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
        channel_title="Rick Astley",
        published_at="2009-10-25T06:57:33Z",
        category_id="10",
        duration="PT3M33S",
        duration_seconds=213,
        privacy_status="public",
        thumbnail_default="https://i.ytimg.com/vi/dQw4w9WgXcQ/default.jpg",
        thumbnail_medium="https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg",
        thumbnail_high="https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
        view_count=1500000000,
        like_count=17000000,
        comment_count=2500000,
        tags=["Rick Astley", "Never Gonna Give You Up", "Pop"],
    )


# 1. Test Metadata Persistence
def test_metadata_persistence(in_memory_db, mock_metadata):
    repo = YouTubeVideoRepository(in_memory_db)
    video = repo.upsert_video_from_metadata(
        youtube_video_id=mock_metadata.video_id,
        title=mock_metadata.title,
        description=mock_metadata.description,
        channel_id=mock_metadata.channel_id,
        channel_title=mock_metadata.channel_title,
        published_at=datetime.utcnow(),
        category_id=mock_metadata.category_id,
        thumbnail_url=mock_metadata.thumbnail_high,
        duration=mock_metadata.duration_seconds,
        view_count=mock_metadata.view_count,
        like_count=mock_metadata.like_count,
        comment_count=mock_metadata.comment_count,
        privacy_status=mock_metadata.privacy_status,
        tags=mock_metadata.tags,
        ingestion_status=IngestionStatus.METADATA_READY.value,
    )

    assert video is not None
    assert video.youtube_video_id == "dQw4w9WgXcQ"
    assert video.title == "Never Gonna Give You Up"
    assert video.channel_title == "Rick Astley"
    assert video.duration == 213
    assert video.view_count == 1500000000
    assert video.ingestion_status == IngestionStatus.METADATA_READY.value


# 2. Test Duplicate Video Ingestion (No duplicates created)
def test_duplicate_video_ingestion(in_memory_db, mock_metadata):
    repo = YouTubeVideoRepository(in_memory_db)
    v1 = repo.upsert_video_from_metadata(
        youtube_video_id=mock_metadata.video_id,
        title=mock_metadata.title,
        ingestion_status=IngestionStatus.METADATA_READY.value,
    )
    v2 = repo.upsert_video_from_metadata(
        youtube_video_id=mock_metadata.video_id,
        title="Updated Title",
        ingestion_status=IngestionStatus.MEDIA_READY.value,
    )

    assert v1.id == v2.id
    all_videos = in_memory_db.query(Video).filter(Video.youtube_video_id == mock_metadata.video_id).all()
    assert len(all_videos) == 1
    assert all_videos[0].title == "Updated Title"
    assert all_videos[0].ingestion_status == IngestionStatus.MEDIA_READY.value


# 3. Test Ingestion State Transitions
def test_ingestion_state_transitions(in_memory_db):
    repo = YouTubeVideoRepository(in_memory_db)
    ingestion = repo.get_or_create_ingestion("dQw4w9WgXcQ")
    assert ingestion.status == IngestionStatus.PENDING.value

    repo.update_ingestion_state("dQw4w9WgXcQ", IngestionStatus.METADATA_READY)
    assert repo.get_ingestion("dQw4w9WgXcQ").status == IngestionStatus.METADATA_READY.value

    repo.update_ingestion_state("dQw4w9WgXcQ", IngestionStatus.MEDIA_DOWNLOADING)
    assert repo.get_ingestion("dQw4w9WgXcQ").status == IngestionStatus.MEDIA_DOWNLOADING.value

    repo.update_ingestion_state("dQw4w9WgXcQ", IngestionStatus.MEDIA_READY, video_path="/path/video.mp4")
    updated = repo.get_ingestion("dQw4w9WgXcQ")
    assert updated.status == IngestionStatus.MEDIA_READY.value
    assert updated.video_path == "/path/video.mp4"


# 4. Test Failed Ingestion
def test_failed_ingestion(in_memory_db):
    repo = YouTubeVideoRepository(in_memory_db)
    repo.get_or_create_ingestion("dQw4w9WgXcQ")
    repo.update_ingestion_state(
        "dQw4w9WgXcQ",
        status=IngestionStatus.FAILED,
        error_message="Network connection timed out",
    )
    record = repo.get_ingestion("dQw4w9WgXcQ")
    assert record.status == IngestionStatus.FAILED.value
    assert "timed out" in record.error_message


# 5. Test Retry Behavior
def test_retry_behavior(in_memory_db):
    repo = YouTubeVideoRepository(in_memory_db)
    ingestion = repo.get_or_create_ingestion("dQw4w9WgXcQ", max_attempts=3)
    assert ingestion.attempt_count == 0

    repo.update_ingestion_state("dQw4w9WgXcQ", status=IngestionStatus.PENDING, increment_attempt=True)
    assert repo.get_ingestion("dQw4w9WgXcQ").attempt_count == 1

    repo.update_ingestion_state("dQw4w9WgXcQ", status=IngestionStatus.FAILED, error_message="Fail 1")
    repo.update_ingestion_state("dQw4w9WgXcQ", status=IngestionStatus.PENDING, increment_attempt=True)
    assert repo.get_ingestion("dQw4w9WgXcQ").attempt_count == 2

    repo.update_ingestion_state("dQw4w9WgXcQ", status=IngestionStatus.FAILED, error_message="Fail 2")
    repo.update_ingestion_state("dQw4w9WgXcQ", status=IngestionStatus.PENDING, increment_attempt=True)
    assert repo.get_ingestion("dQw4w9WgXcQ").attempt_count == 3


# 6. Test Media Path Validation (Directory Traversal Prevention)
def test_media_path_validation(tmp_path):
    storage = LocalStorage(root_dir=tmp_path / "storage")

    # Valid video dir
    video_dir = storage.get_video_dir("dQw4w9WgXcQ")
    assert video_dir.exists()
    assert str(storage.root_path) in str(video_dir)

    # Path traversal attempt with ../
    with pytest.raises(StorageSecurityError):
        storage.get_path("../../etc/passwd")

    with pytest.raises(StorageSecurityError):
        storage.get_video_dir("../outside_folder")


# 7. Test Media Size Validation
def test_media_size_validation(tmp_path):
    validator = MediaValidator(max_video_size_mb=1)
    large_file = tmp_path / "large_video.mp4"
    large_file.write_bytes(b"0" * (2 * 1024 * 1024))  # 2MB > 1MB limit

    with pytest.raises(MediaSizeExceededError):
        validator.validate_video_file(large_file, video_id="test123")


# 8. Test Invalid Media Rejection
def test_invalid_media_rejection(tmp_path):
    validator = MediaValidator()

    # Empty 0-byte file
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")
    with pytest.raises(MediaCorruptError):
        validator.validate_video_file(empty_file, video_id="test123")

    # Non-existent file
    with pytest.raises(MediaValidationError):
        validator.validate_video_file(tmp_path / "missing.mp4", video_id="test123")


# 9. Test Audio Extraction
def test_audio_extraction(tmp_path):
    extractor = AudioExtractor(sample_rate=16000, duration=2.0)
    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"dummy video content")

    out_wav = tmp_path / "audio.wav"
    dummy_waveform = torch.zeros(32000)  # 2s @ 16kHz

    with patch.object(extractor.processor, "extract_audio", return_value=dummy_waveform):
        result_path = extractor.extract_and_save_wav(fake_video, out_wav, video_id="test123")
        assert result_path.exists()
        assert result_path.stat().st_size > 0


# 10. Test Missing Audio Handling
def test_missing_audio_handling(tmp_path):
    extractor = AudioExtractor(sample_rate=16000, duration=1.0)
    missing_video = tmp_path / "not_found.mp4"
    out_wav = tmp_path / "audio.wav"

    # Should raise AudioExtractionError if input video does not exist
    from src.integrations.youtube.media.exceptions import AudioExtractionError
    with pytest.raises(AudioExtractionError):
        extractor.extract_and_save_wav(missing_video, out_wav, video_id="test_missing")


# 11. Test Thumbnail Validation
def test_thumbnail_validation(tmp_path):
    downloader = ThumbnailDownloader(max_size_mb=5)

    # Valid YouTube CDN hostnames
    assert downloader.is_safe_url("https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg") is True
    assert downloader.is_safe_url("https://img.youtube.com/vi/dQw4w9WgXcQ/default.jpg") is True

    # Untrusted domain rejected
    assert downloader.is_safe_url("https://evil.com/hack.jpg") is False
    assert downloader.is_safe_url("http://localhost:8000/test.png") is False

    # Download from untrusted domain raises ThumbnailDownloadError
    with pytest.raises(ThumbnailDownloadError):
        downloader.download_thumbnail("https://attacker.org/image.jpg", tmp_path / "thumb.jpg")


# 12. Test Transcript Unavailable Graceful Handling
def test_transcript_unavailable():
    provider = YouTubeTranscriptProvider()
    with patch("src.integrations.youtube.transcript.YouTubeTranscriptApi") as mock_api:
        from youtube_transcript_api import TranscriptsDisabled
        mock_api.get_transcript.side_effect = TranscriptsDisabled("dQw4w9WgXcQ")

        result = provider.get_transcript("dQw4w9WgXcQ")
        assert result.available is False
        assert result.text is None
        assert "disabled" in result.error.lower()


# 13. Test Ingestion Endpoint
def test_ingestion_endpoint(client, in_memory_db):
    response = client.post("/api/youtube/ingest", json={"video_id": "dQw4w9WgXcQ"})
    assert response.status_code == 202
    data = response.json()
    assert data["youtube_video_id"] == "dQw4w9WgXcQ"
    assert data["status"] == IngestionStatus.PENDING.value

    # Reject invalid video ID format
    bad_response = client.post("/api/youtube/ingest", json={"video_id": "invalid_id_!!!"})
    assert bad_response.status_code == 400


# 14. Test Ingestion Status Endpoint
def test_ingestion_status_endpoint(client, in_memory_db):
    repo = YouTubeVideoRepository(in_memory_db)
    repo.get_or_create_ingestion("dQw4w9WgXcQ")
    repo.update_ingestion_state(
        "dQw4w9WgXcQ",
        status=IngestionStatus.MEDIA_READY,
        video_path="C:/secret/internal/path/video.mp4",  # Internal path
    )

    response = client.get("/api/youtube/ingest/dQw4w9WgXcQ")
    assert response.status_code == 200
    data = response.json()
    assert data["youtube_video_id"] == "dQw4w9WgXcQ"
    assert data["status"] == IngestionStatus.MEDIA_READY.value
    # Ensure internal paths are NEVER leaked in response
    assert "video_path" not in data
    assert "audio_path" not in data
    assert "secret" not in json.dumps(data)


# 15. Test Ingestion Idempotency
def test_ingestion_idempotency(client, in_memory_db):
    resp1 = client.post("/api/youtube/ingest", json={"video_id": "dQw4w9WgXcQ"})
    resp2 = client.post("/api/youtube/ingest", json={"video_id": "dQw4w9WgXcQ"})
    assert resp1.status_code == 202
    assert resp2.status_code == 202

    records = in_memory_db.query(VideoIngestion).filter(VideoIngestion.youtube_video_id == "dQw4w9WgXcQ").all()
    assert len(records) == 1


# 16. Test Database Rollback on Failure
def test_database_rollback_on_failure(in_memory_db, tmp_path, mock_metadata):
    settings = Settings(MEDIA_STORAGE_DIR=str(tmp_path))
    storage = LocalStorage(root_dir=tmp_path)

    # Ingestion service with mock dependencies
    mock_meta_service = MagicMock()
    mock_meta_service.get_video_metadata.return_value = mock_metadata

    mock_downloader = MagicMock()
    mock_downloader.download_video.side_effect = MediaDownloadError("Upstream download failed", video_id="dQw4w9WgXcQ")

    service = YouTubeIngestionService(
        settings=settings,
        session_factory=lambda: in_memory_db,
        storage=storage,
        metadata_service=mock_meta_service,
        downloader=mock_downloader,
    )

    result = service.ingest_video("dQw4w9WgXcQ")
    assert result["status"] == IngestionStatus.FAILED.value
    assert "Upstream download failed" in result["error_message"]

    repo = YouTubeVideoRepository(in_memory_db)
    record = repo.get_ingestion("dQw4w9WgXcQ")
    assert record.status == IngestionStatus.FAILED.value
