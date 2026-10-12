"""
YouTube Ingestion Service.
Orchestrates metadata extraction, persistence, media acquisition,
validation, audio extraction, thumbnail handling, and optional transcript retrieval.
"""

import concurrent.futures
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Optional, Union
import uuid
from sqlalchemy.orm import Session

from src.db.config import Settings, get_settings
from src.db.database import get_database
from src.db.models import IngestionStatus, Video, VideoIngestion, VideoAsset, AssetType, AssetStatus
from src.db.repository import YouTubeVideoRepository, VideoAssetRepository
from src.features.frame_sampler import FrameSampler
from src.storage.base import BaseStorage
from src.storage.local import LocalStorage
from src.integrations.youtube.parser import extract_video_id
from src.integrations.youtube.service import YouTubeMetadataService
from src.integrations.youtube.media.downloader import MediaDownloader
from src.integrations.youtube.media.validator import MediaValidator
from src.integrations.youtube.media.audio import AudioExtractor
from src.integrations.youtube.media.thumbnails import ThumbnailDownloader
from src.integrations.youtube.media.exceptions import (
    MediaError,
    MediaDownloadError,
    MediaValidationError,
    MediaSizeExceededError,
    MediaDurationExceededError,
    MediaCorruptError,
    AudioExtractionError,
    ThumbnailDownloadError,
)
from src.integrations.youtube.transcript import (
    TranscriptProvider,
    YouTubeTranscriptProvider,
)
from src.integrations.youtube.exceptions import (
    YouTubeAPIError,
    YouTubeAuthenticationError,
    YouTubeConfigurationError,
    YouTubeInvalidVideoIdError,
    YouTubeNotFoundError,
    YouTubePrivateVideoError,
    YouTubeQuotaExceededError,
)

logger = logging.getLogger(__name__)

# Non-retryable terminal errors
TERMINAL_EXCEPTIONS = (
    YouTubeInvalidVideoIdError,
    YouTubeNotFoundError,
    YouTubePrivateVideoError,
    YouTubeAuthenticationError,
    YouTubeQuotaExceededError,
    MediaSizeExceededError,
    MediaDurationExceededError,
)


class YouTubeIngestionService:
    """
    Coordinates end-to-end ingestion lifecycle for YouTube videos.
    Designed to run asynchronously and be safe against crashes and restarts.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        session_factory: Optional[Callable[[], Session]] = None,
        storage: Optional[BaseStorage] = None,
        metadata_service: Optional[YouTubeMetadataService] = None,
        downloader: Optional[MediaDownloader] = None,
        validator: Optional[MediaValidator] = None,
        audio_extractor: Optional[AudioExtractor] = None,
        thumbnail_downloader: Optional[ThumbnailDownloader] = None,
        transcript_provider: Optional[TranscriptProvider] = None,
        frame_sampler: Optional[FrameSampler] = None,
    ):
        self.settings = settings or get_settings()
        self._session_factory = session_factory or (lambda: get_database().SessionLocal())
        self.storage = storage or LocalStorage(root_dir=self.settings.MEDIA_STORAGE_DIR)
        self.metadata_service = metadata_service or YouTubeMetadataService(settings=self.settings)
        self.downloader = downloader or MediaDownloader(
            timeout=self.settings.YT_DLP_TIMEOUT,
            max_size_mb=self.settings.MAX_VIDEO_SIZE_MB,
            max_duration_seconds=self.settings.MAX_VIDEO_DURATION_SECONDS,
        )
        self.validator = validator or MediaValidator(
            max_video_size_mb=self.settings.MAX_VIDEO_SIZE_MB,
            max_audio_size_mb=self.settings.MAX_AUDIO_SIZE_MB,
            max_duration_seconds=self.settings.MAX_VIDEO_DURATION_SECONDS,
        )
        self.audio_extractor = audio_extractor or AudioExtractor(
            sample_rate=16000,
            duration=5.0,
        )
        self.thumbnail_downloader = thumbnail_downloader or ThumbnailDownloader(
            max_size_mb=self.settings.THUMBNAIL_MAX_SIZE_MB,
        )
        self.transcript_provider = transcript_provider or YouTubeTranscriptProvider()
        self.frame_sampler = frame_sampler or FrameSampler(default_num_frames=8)

    def _get_db_session(self) -> Session:
        return self._session_factory()

    def register_ingestion_request(self, video_input: str) -> dict[str, Any]:
        """
        Synchronously validate video ID and create/fetch PENDING ingestion record.
        Returns immediately with status info so API response is fast.
        """
        canonical_id = extract_video_id(video_input)
        session = self._get_db_session()
        try:
            repo = YouTubeVideoRepository(session)
            ingestion = repo.get_or_create_ingestion(
                youtube_video_id=canonical_id,
                max_attempts=self.settings.INGESTION_MAX_RETRIES,
            )
            return {
                "video_id": ingestion.video_id,
                "youtube_video_id": ingestion.youtube_video_id,
                "status": ingestion.status,
                "attempt_count": ingestion.attempt_count,
            }
        finally:
            session.close()

    def get_ingestion_status(self, video_input: str) -> Optional[dict[str, Any]]:
        """
        Query current ingestion status without exposing internal filesystem paths.
        """
        try:
            identifier = extract_video_id(video_input)
        except Exception:
            identifier = video_input

        session = self._get_db_session()
        try:
            repo = YouTubeVideoRepository(session)
            ingestion = repo.get_ingestion(identifier)
            if not ingestion:
                return None
            return ingestion.to_dict()
        finally:
            session.close()

    def ingest_video(self, video_input: str) -> dict[str, Any]:
        """
        Execute full asynchronous media ingestion workflow.
        Restart-safe and idempotent.
        """
        # 1. Validate video input
        canonical_id = extract_video_id(video_input)

        session = self._get_db_session()
        try:
            repo = YouTubeVideoRepository(session)
            ingestion = repo.get_or_create_ingestion(
                youtube_video_id=canonical_id,
                max_attempts=self.settings.INGESTION_MAX_RETRIES,
            )

            # Check retry budget
            if ingestion.status == IngestionStatus.FAILED.value and ingestion.attempt_count >= ingestion.max_attempts:
                logger.warning("Video %s has reached max ingestion attempts (%d)", canonical_id, ingestion.max_attempts)
                return ingestion.to_dict()

            # Record attempt start
            repo.update_ingestion_state(
                youtube_video_id=canonical_id,
                status=IngestionStatus.PENDING,
                increment_attempt=True,
            )
        finally:
            session.close()

        video_dir = self.storage.get_video_dir(canonical_id)

        try:
            # 2. Fetch metadata from YouTube Data API
            metadata = self.metadata_service.get_video_metadata(canonical_id)

            # 3. Persist normalized metadata
            session = self._get_db_session()
            try:
                repo = YouTubeVideoRepository(session)
                published_dt = None
                if metadata.published_at:
                    try:
                        published_dt = datetime.fromisoformat(metadata.published_at.replace("Z", "+00:00"))
                    except Exception:
                        published_dt = None

                repo.upsert_video_from_metadata(
                    youtube_video_id=canonical_id,
                    title=metadata.title,
                    description=metadata.description,
                    channel_id=metadata.channel_id,
                    channel_title=metadata.channel_title,
                    published_at=published_dt,
                    category_id=metadata.category_id,
                    thumbnail_url=metadata.thumbnail_high or metadata.thumbnail_default,
                    duration=metadata.duration_seconds,
                    view_count=metadata.view_count,
                    like_count=metadata.like_count,
                    comment_count=metadata.comment_count,
                    privacy_status=metadata.privacy_status,
                    tags=metadata.tags,
                    ingestion_status=IngestionStatus.METADATA_READY.value,
                )
                repo.update_ingestion_state(
                    youtube_video_id=canonical_id,
                    status=IngestionStatus.METADATA_READY,
                )
            finally:
                session.close()

            # 4. Clean previous incomplete temporary files before starting download
            if hasattr(self.storage, "clean_incomplete_files"):
                self.storage.clean_incomplete_files(canonical_id)

            # 5. Acquire media
            session = self._get_db_session()
            try:
                repo = YouTubeVideoRepository(session)
                repo.update_ingestion_state(
                    youtube_video_id=canonical_id,
                    status=IngestionStatus.MEDIA_DOWNLOADING,
                )
            finally:
                session.close()

            video_path = self.downloader.download_video(canonical_id, video_dir)

            # 6. Validate video media
            self.validator.validate_video_file(video_path, video_id=canonical_id)

            # 7. Extract audio (16 kHz mono WAV)
            audio_path = video_dir / "audio.wav"
            self.audio_extractor.extract_and_save_wav(video_path, audio_path, video_id=canonical_id)
            self.validator.validate_audio_file(audio_path, video_id=canonical_id)

            # 8. Acquire thumbnail
            thumbnail_path = None
            thumb_url = metadata.thumbnail_high or metadata.thumbnail_medium or metadata.thumbnail_default
            if thumb_url:
                try:
                    target_thumb = video_dir / "thumbnail.jpg"
                    thumbnail_path = self.thumbnail_downloader.download_thumbnail(
                        thumb_url,
                        target_thumb,
                        video_id=canonical_id,
                    )
                except Exception as thumb_err:
                    logger.warning("Thumbnail acquisition failed for %s (non-fatal): %s", canonical_id, thumb_err)

            # 9. Optional transcript retrieval
            transcript_text = None
            transcript_lang = None
            transcript_avail = False
            transcript_prov = None
            if self.settings.TRANSCRIPT_ENABLED:
                try:
                    res = self.transcript_provider.get_transcript(canonical_id)
                    transcript_avail = res.available
                    transcript_lang = res.language
                    transcript_prov = "youtube-transcript-api"
                    if res.available and res.text:
                        transcript_text = res.text
                except Exception as trans_err:
                    logger.info("Transcript retrieval skipped/failed for %s: %s", canonical_id, trans_err)

            # 10. Extract video frames (bounded, deterministic)
            frames_dir = video_dir / "frames"
            frames_meta = []
            try:
                frames_meta = self.frame_sampler.extract_and_save_frames(
                    video_path=video_path,
                    output_dir=frames_dir,
                    num_frames=8,
                    video_id=canonical_id,
                )
                if frames_meta:
                    self.storage.save_json(f"{canonical_id}/frames_metadata.json", frames_meta)
            except Exception as frame_err:
                logger.warning("Frame extraction encountered non-fatal error for %s: %s", canonical_id, frame_err)

            # 11. Save normalized metadata json alongside media
            self.storage.save_json(
                f"{canonical_id}/metadata.json",
                metadata.to_api_response().model_dump() if hasattr(metadata.to_api_response(), "model_dump") else metadata.to_api_response().dict(),
            )

            # 12. Register multimodal assets in database
            session = self._get_db_session()
            try:
                repo = YouTubeVideoRepository(session)
                asset_repo = VideoAssetRepository(session)
                ingestion_record = repo.get_ingestion(canonical_id)
                internal_vid = ingestion_record.video_id if ingestion_record and ingestion_record.video_id else f"yt_{canonical_id}"

                # Register Video Asset
                if video_path and Path(video_path).exists():
                    v_p = Path(video_path)
                    v_checksum = self.validator.compute_sha256(v_p)
                    asset_repo.upsert_asset(
                        video_id=internal_vid,
                        youtube_video_id=canonical_id,
                        asset_type=AssetType.VIDEO,
                        storage_path=f"{canonical_id}/video.mp4",
                        mime_type="video/mp4",
                        file_size_bytes=v_p.stat().st_size,
                        checksum_sha256=v_checksum,
                        status=AssetStatus.READY,
                        pipeline_version="12A",
                    )

                # Register Audio Asset
                if audio_path and Path(audio_path).exists():
                    a_p = Path(audio_path)
                    a_checksum = self.validator.compute_sha256(a_p)
                    a_val = self.validator.validate_audio_file(a_p, video_id=canonical_id)
                    asset_repo.upsert_asset(
                        video_id=internal_vid,
                        youtube_video_id=canonical_id,
                        asset_type=AssetType.AUDIO,
                        storage_path=f"{canonical_id}/audio.wav",
                        mime_type="audio/wav",
                        file_size_bytes=a_p.stat().st_size,
                        checksum_sha256=a_checksum,
                        status=AssetStatus.READY,
                        pipeline_version="12A",
                        extra_metadata=a_val,
                    )

                # Register Thumbnail Asset
                if thumbnail_path and Path(thumbnail_path).exists():
                    t_p = Path(thumbnail_path)
                    t_checksum = self.validator.compute_sha256(t_p)
                    t_val = self.validator.validate_image_file(t_p, video_id=canonical_id)
                    asset_repo.upsert_asset(
                        video_id=internal_vid,
                        youtube_video_id=canonical_id,
                        asset_type=AssetType.THUMBNAIL,
                        source_url=thumb_url,
                        storage_path=f"{canonical_id}/thumbnail.jpg",
                        mime_type="image/jpeg",
                        file_size_bytes=t_p.stat().st_size,
                        checksum_sha256=t_checksum,
                        status=AssetStatus.READY,
                        pipeline_version="12A",
                        extra_metadata=t_val,
                    )

                # Register Frame Assets
                for f_info in frames_meta:
                    asset_repo.upsert_asset(
                        video_id=internal_vid,
                        youtube_video_id=canonical_id,
                        asset_type=AssetType.FRAME,
                        storage_path=f"{canonical_id}/frames/{f_info['filename']}",
                        mime_type="image/jpeg",
                        file_size_bytes=f_info["file_size_bytes"],
                        checksum_sha256=f_info["checksum_sha256"],
                        status=AssetStatus.READY,
                        pipeline_version="12A",
                        extra_metadata={
                            "frame_index": f_info["frame_index"],
                            "timestamp_seconds": f_info["timestamp_seconds"],
                            "width": f_info["width"],
                            "height": f_info["height"],
                        },
                    )

                # Register Metadata Asset
                meta_json_file = video_dir / "metadata.json"
                if meta_json_file.exists():
                    m_checksum = self.validator.compute_sha256(meta_json_file)
                    asset_repo.upsert_asset(
                        video_id=internal_vid,
                        youtube_video_id=canonical_id,
                        asset_type=AssetType.METADATA,
                        storage_path=f"{canonical_id}/metadata.json",
                        mime_type="application/json",
                        file_size_bytes=meta_json_file.stat().st_size,
                        checksum_sha256=m_checksum,
                        status=AssetStatus.READY,
                        pipeline_version="12A",
                    )

                # 13. Update final ingestion record to MEDIA_READY
                ingestion = repo.update_ingestion_state(
                    youtube_video_id=canonical_id,
                    status=IngestionStatus.MEDIA_READY,
                    video_path=str(video_path),
                    audio_path=str(audio_path),
                    thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
                    transcript_text=transcript_text,
                    transcript_language=transcript_lang,
                    transcript_available=transcript_avail,
                    transcript_provider=transcript_prov,
                    frames_path=str(frames_dir) if frames_meta else None,
                    frames_count=len(frames_meta),
                    pipeline_version="12A",
                )
                logger.info("Ingestion completed successfully for %s (status=MEDIA_READY, assets=%d frames)", canonical_id, len(frames_meta))
                return ingestion.to_dict() if ingestion else {}
            finally:
                session.close()

        except Exception as e:
            logger.error("Ingestion failed for %s: %s", canonical_id, e, exc_info=True)

            # Cleanup incomplete files on error
            if hasattr(self.storage, "clean_incomplete_files"):
                self.storage.clean_incomplete_files(canonical_id)

            # Check if terminal failure
            is_terminal = isinstance(e, TERMINAL_EXCEPTIONS)
            error_message = str(e)

            session = self._get_db_session()
            try:
                session.rollback()
                repo = YouTubeVideoRepository(session)
                ingestion = repo.get_ingestion(canonical_id)
                current_attempts = ingestion.attempt_count if ingestion else 1
                max_attempts = ingestion.max_attempts if ingestion else self.settings.INGESTION_MAX_RETRIES

                final_status = IngestionStatus.FAILED
                if not is_terminal and current_attempts < max_attempts:
                    # Retryable transient failure
                    error_message = f"Transient error (attempt {current_attempts}/{max_attempts}): {error_message}"
                else:
                    error_message = f"Permanent failure: {error_message}"

                updated = repo.update_ingestion_state(
                    youtube_video_id=canonical_id,
                    status=final_status,
                    error_message=error_message,
                )
                return updated.to_dict() if updated else {}
            except Exception as db_err:
                logger.error("Failed to update failure state in DB for %s: %s", canonical_id, db_err)
                session.rollback()
                raise
            finally:
                session.close()

    def get_aligned_assets(self, video_id: str) -> list[dict[str, Any]]:
        """
        Query all registered multimodal assets (video, audio, thumbnail, frame)
        associated with an internal video ID without exposing local absolute paths.
        """
        session = self._get_db_session()
        try:
            repo = VideoAssetRepository(session)
            assets = repo.get_assets_for_video(video_id)
            return [a.to_dict() for a in assets]
        finally:
            session.close()

    def ingest_local_video(
        self,
        video_id: str,
        local_video_path: Path | str,
        title: Optional[str] = None,
        youtube_video_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Ingest a local video asset (e.g. from existing catalog or dataset files)
        into the unified multimodal asset repository. Extracts audio, frames, thumbnail,
        computes SHA-256 checksums, and registers all aligned assets in the database.
        """
        src_path = Path(local_video_path)
        if not src_path.exists() or not src_path.is_file():
            raise MediaValidationError(f"Source video file not found: {src_path}", video_id=video_id)

        target_id = youtube_video_id or f"local_{video_id}"
        video_dir = self.storage.get_video_dir(target_id)
        dest_video_path = video_dir / "video.mp4"

        # Copy source video if not already in destination
        if src_path.resolve() != dest_video_path.resolve():
            shutil.copy2(src_path, dest_video_path)

        # Validate video
        vid_meta = self.validator.validate_video_file(dest_video_path, video_id=video_id)

        # Extract audio (16 kHz mono WAV)
        audio_path = video_dir / "audio.wav"
        audio_meta = None
        audio_error_category = None
        audio_error_message = None
        try:
            self.audio_extractor.extract_and_save_wav(dest_video_path, audio_path, video_id=video_id)
            audio_meta = self.validator.validate_audio_file(audio_path, video_id=video_id, check_silence=True)
        except Exception as e:
            err_str = str(e)
            logger.warning("Audio extraction failed for local video %s: %s", video_id, err_str)
            audio_path = None
            if "NO_AUDIO_STREAM" in err_str:
                audio_error_category = "NO_AUDIO_STREAM"
            elif "SILENT_AUDIO_REJECTED" in err_str or "silent" in err_str.lower():
                audio_error_category = "SILENT_AUDIO_REJECTED"
            else:
                audio_error_category = "EXTRACTION_FAILURE"
            audio_error_message = err_str

        # Extract frames (8 deterministic frames)
        frames_dir = video_dir / "frames"
        frames_meta = []
        try:
            frames_meta = self.frame_sampler.extract_and_save_frames(
                video_path=dest_video_path,
                output_dir=frames_dir,
                num_frames=8,
                video_id=video_id,
            )
            if frames_meta:
                self.storage.save_json(f"{target_id}/frames_metadata.json", frames_meta)
        except Exception as e:
            logger.warning("Frame extraction failed for local video %s: %s", video_id, e)

        # Thumbnail from first frame if available
        thumbnail_path = video_dir / "thumbnail.jpg"
        if not thumbnail_path.exists() and frames_meta and Path(frames_meta[0]["storage_path"]).exists():
            shutil.copy2(frames_meta[0]["storage_path"], thumbnail_path)

        # Save metadata JSON
        meta_dict = {
            "video_id": video_id,
            "youtube_video_id": youtube_video_id,
            "title": title or f"Catalog Video {video_id}",
            "duration_seconds": vid_meta.get("duration_seconds", 0.0),
            "video_format": "mp4",
            "frame_count": vid_meta.get("frame_count", 0),
            "fps": vid_meta.get("fps", 0.0),
        }
        self.storage.save_json(f"{target_id}/metadata.json", meta_dict)

        # Register assets in DB
        session = self._get_db_session()
        try:
            repo = YouTubeVideoRepository(session)
            asset_repo = VideoAssetRepository(session)

            # Update existing Video record if present
            video_rec = repo.find_by_video_id(video_id)
            if video_rec:
                video_rec.video_path = str(dest_video_path)
                video_rec.duration = int(vid_meta.get("duration_seconds", 0))
                video_rec.ingestion_status = IngestionStatus.MEDIA_READY.value
                session.commit()

            # Video Ingestion record
            repo.get_or_create_ingestion(
                youtube_video_id=target_id,
                video_id=video_id,
            )

            # Register Video Asset
            v_checksum = self.validator.compute_sha256(dest_video_path)
            asset_repo.upsert_asset(
                video_id=video_id,
                youtube_video_id=youtube_video_id,
                asset_type=AssetType.VIDEO,
                storage_path=f"{target_id}/video.mp4",
                mime_type="video/mp4",
                file_size_bytes=dest_video_path.stat().st_size,
                checksum_sha256=v_checksum,
                status=AssetStatus.READY,
                pipeline_version="13",
                extra_metadata=vid_meta,
            )

            # Register Audio Asset
            if audio_path and audio_path.exists():
                a_checksum = self.validator.compute_sha256(audio_path)
                asset_repo.upsert_asset(
                    video_id=video_id,
                    youtube_video_id=youtube_video_id,
                    asset_type=AssetType.AUDIO,
                    storage_path=f"{target_id}/audio.wav",
                    mime_type="audio/wav",
                    file_size_bytes=audio_path.stat().st_size,
                    checksum_sha256=a_checksum,
                    status=AssetStatus.READY,
                    pipeline_version="13",
                    extra_metadata=audio_meta,
                )
            else:
                asset_repo.upsert_asset(
                    video_id=video_id,
                    youtube_video_id=youtube_video_id,
                    asset_type=AssetType.AUDIO,
                    storage_path=f"{target_id}/audio.wav",
                    mime_type="audio/wav",
                    file_size_bytes=0,
                    checksum_sha256=None,
                    status=AssetStatus.FAILED,
                    error_category=audio_error_category or "EXTRACTION_FAILURE",
                    error_message=audio_error_message or "Audio unavailable or silent",
                    pipeline_version="13",
                )

            # Register Thumbnail Asset
            if thumbnail_path and thumbnail_path.exists():
                t_checksum = self.validator.compute_sha256(thumbnail_path)
                asset_repo.upsert_asset(
                    video_id=video_id,
                    youtube_video_id=youtube_video_id,
                    asset_type=AssetType.THUMBNAIL,
                    storage_path=f"{target_id}/thumbnail.jpg",
                    mime_type="image/jpeg",
                    file_size_bytes=thumbnail_path.stat().st_size,
                    checksum_sha256=t_checksum,
                    status=AssetStatus.READY,
                    pipeline_version="12A",
                )

            # Register Frame Assets
            for f_info in frames_meta:
                asset_repo.upsert_asset(
                    video_id=video_id,
                    youtube_video_id=youtube_video_id,
                    asset_type=AssetType.FRAME,
                    storage_path=f"{target_id}/frames/{f_info['filename']}",
                    mime_type="image/jpeg",
                    file_size_bytes=f_info["file_size_bytes"],
                    checksum_sha256=f_info["checksum_sha256"],
                    status=AssetStatus.READY,
                    pipeline_version="12A",
                    extra_metadata={
                        "frame_index": f_info["frame_index"],
                        "timestamp_seconds": f_info["timestamp_seconds"],
                        "width": f_info["width"],
                        "height": f_info["height"],
                    },
                )

            # Register Metadata Asset
            meta_json_file = video_dir / "metadata.json"
            if meta_json_file.exists():
                m_checksum = self.validator.compute_sha256(meta_json_file)
                asset_repo.upsert_asset(
                    video_id=video_id,
                    youtube_video_id=youtube_video_id,
                    asset_type=AssetType.METADATA,
                    storage_path=f"{target_id}/metadata.json",
                    mime_type="application/json",
                    file_size_bytes=meta_json_file.stat().st_size,
                    checksum_sha256=m_checksum,
                    status=AssetStatus.READY,
                    pipeline_version="12A",
                )

            # Update ingestion state
            repo.update_ingestion_state(
                youtube_video_id=target_id,
                status=IngestionStatus.MEDIA_READY,
                video_path=str(dest_video_path),
                audio_path=str(audio_path) if audio_path else None,
                thumbnail_path=str(thumbnail_path) if thumbnail_path.exists() else None,
                frames_path=str(frames_dir) if frames_meta else None,
                frames_count=len(frames_meta),
                pipeline_version="12A",
            )

            return {
                "video_id": video_id,
                "target_id": target_id,
                "status": "COMPLETED",
                "assets_count": len(frames_meta) + (1 if audio_path else 0) + (1 if thumbnail_path.exists() else 0) + 2,
                "frames_count": len(frames_meta),
                "has_audio": audio_path is not None,
                "has_video": True,
            }
        finally:
            session.close()


class AssetIngestionOrchestrator:
    """
    Production-grade, resumable, bounded-concurrency job orchestrator for batch asset ingestion.
    Supports:
    - Bounded concurrency (1 to 8 workers)
    - Resumable job checkpoints persisted to disk
    - Deduplication against existing catalog and asset records
    - Rate limiting delays between ingestion items
    - Storage budget enforcement (max_storage_gb)
    - Retry logic for transient extraction/network failures
    - Support for local files (MicroLens) and remote YouTube videos
    """

    def __init__(
        self,
        ingestion_service: Optional[YouTubeIngestionService] = None,
        max_concurrency: int = 2,
        batch_size: int = 100,
        rate_limit_delay_seconds: float = 0.0,
        max_storage_gb: float = 10.0,
        max_retries: int = 3,
        checkpoint_dir: Optional[Path | str] = None,
    ):
        self.service = ingestion_service or YouTubeIngestionService()
        self.max_concurrency = max(1, min(max_concurrency, 8))
        self.batch_size = max(1, min(batch_size, 1000))
        self.rate_limit_delay_seconds = max(0.0, rate_limit_delay_seconds)
        self.max_storage_gb = max_storage_gb
        self.max_retries = max_retries
        self.checkpoint_dir = Path(checkpoint_dir or "data/checkpoints/EXP_DATASET_READINESS_13")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency)

    def _get_storage_usage_bytes(self) -> int:
        """Calculate total storage consumed by media assets on disk."""
        total = 0
        storage_root = getattr(self.service.storage, "root_path", Path("data/raw/youtube"))
        if storage_root.exists():
            for root, _, files in os.walk(storage_root):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except (OSError, FileNotFoundError):
                        pass
        return total

    def _check_storage_budget(self) -> bool:
        """Verify whether remaining disk allocation allows further ingestion."""
        current_bytes = self._get_storage_usage_bytes()
        max_bytes = self.max_storage_gb * (1024 ** 3)
        return current_bytes < max_bytes

    def _is_already_ingested(self, video_id: str) -> bool:
        """Check if video is already fully ingested with ready assets."""
        try:
            session = self.service._get_db_session()
            try:
                asset_repo = VideoAssetRepository(session)
                assets = asset_repo.get_assets_for_video(video_id)
                ready_assets = [a for a in assets if a.status == AssetStatus.READY.value or a.status == "READY"]
                return len(ready_assets) >= 2  # Has video and frames or audio
            finally:
                session.close()
        except Exception as e:
            logger.warning("Error checking existing ingestion for %s: %s", video_id, e)
            return False

    def _save_checkpoint(self, job_id: str):
        """Persist current job status to disk checkpoint JSON."""
        job_data = self._jobs.get(job_id)
        if not job_data:
            return
        checkpoint_file = self.checkpoint_dir / f"job_{job_id}.json"
        try:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(job_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to save checkpoint for job %s: %s", job_id, e)

    def _process_item(self, item: Union[str, dict], force: bool = False) -> dict[str, Any]:
        """Process a single video item with deduplication, budget check, and retries."""
        if isinstance(item, dict):
            vid = item.get("video_id")
            local_path = item.get("local_video_path")
            title = item.get("title")
        else:
            vid = str(item)
            local_path = None
            title = None

        if not vid:
            return {"status": "FAILED", "video_id": "unknown", "error": "Missing video_id"}

        # 1. Deduplication Check
        if not force and self._is_already_ingested(vid):
            logger.info("Skipping %s: already ingested and MEDIA_READY (deduplicated)", vid)
            return {"status": "SKIPPED", "video_id": vid, "reason": "ALREADY_INGESTED"}

        # 2. Storage Limit Check
        if not self._check_storage_budget():
            current_gb = self._get_storage_usage_bytes() / (1024 ** 3)
            err = f"Storage budget exceeded: {current_gb:.2f} GB used / {self.max_storage_gb:.2f} GB limit"
            logger.error("Skipping %s: %s", vid, err)
            return {"status": "FAILED", "video_id": vid, "error_category": "STORAGE_LIMIT_EXCEEDED", "error": err}

        # 3. Rate Limiting Delay
        if self.rate_limit_delay_seconds > 0:
            time.sleep(self.rate_limit_delay_seconds)

        # 4. Check for local MicroLens media candidate
        candidate_local = None
        if local_path and Path(local_path).exists():
            candidate_local = Path(local_path)
        else:
            microlens_file = Path("data/raw/microlens/videos") / f"{vid}.mp4"
            if microlens_file.exists():
                candidate_local = microlens_file

        # 5. Process with bounded retries
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if candidate_local:
                    res = self.service.ingest_local_video(
                        video_id=vid,
                        local_video_path=candidate_local,
                        title=title,
                    )
                    return {"status": "SUCCESS", "video_id": vid, "source": "LOCAL_FILE", "details": res}
                else:
                    res = self.service.ingest_video(vid)
                    return {"status": "SUCCESS", "video_id": vid, "source": "YOUTUBE_API", "details": res}
            except Exception as e:
                last_error = e
                logger.warning("Attempt %d/%d failed for %s: %s", attempt, self.max_retries, vid, e)
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)

        return {"status": "FAILED", "video_id": vid, "error": str(last_error)}

    def submit_batch(
        self,
        video_inputs: list[Union[str, dict]],
        resume: bool = True,
        force: bool = False,
    ) -> str:
        """Submit a bounded batch of videos for asynchronous processing."""
        bounded_inputs = video_inputs[:self.batch_size]
        job_id = str(uuid.uuid4())[:8]

        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "RUNNING",
            "total": len(bounded_inputs),
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "results": [],
            "errors": [],
            "skipped_items": [],
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }
        self._save_checkpoint(job_id)

        def _worker():
            results = []
            errors = []
            skipped = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                future_to_item = {
                    pool.submit(self._process_item, item, force): item
                    for item in bounded_inputs
                }

                for future in concurrent.futures.as_completed(future_to_item):
                    item = future_to_item[future]
                    vid = item.get("video_id") if isinstance(item, dict) else str(item)
                    try:
                        res = future.result()
                        item_status = res.get("status")
                        if item_status == "SUCCESS":
                            results.append(res)
                            self._jobs[job_id]["completed"] += 1
                        elif item_status == "SKIPPED":
                            skipped.append(res)
                            self._jobs[job_id]["skipped"] += 1
                        else:
                            errors.append(res)
                            self._jobs[job_id]["failed"] += 1
                    except Exception as err:
                        logger.error("Batch worker exception for %s: %s", vid, err)
                        errors.append({"video_id": vid, "error": str(err)})
                        self._jobs[job_id]["failed"] += 1

                    self._jobs[job_id]["results"] = results
                    self._jobs[job_id]["errors"] = errors
                    self._jobs[job_id]["skipped_items"] = skipped
                    self._save_checkpoint(job_id)

            self._jobs[job_id]["status"] = "COMPLETED" if len(errors) == 0 else "PARTIAL_FAILURE"
            self._jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
            self._save_checkpoint(job_id)
            logger.info("Batch job %s finished (completed=%d, skipped=%d, failed=%d)", job_id, len(results), len(skipped), len(errors))

        # Submit background worker
        self._executor.submit(_worker)
        return job_id

    def resume_job(self, job_id: str) -> Optional[str]:
        """Resume an incomplete batch job from disk checkpoint."""
        status = self.get_job_status(job_id)
        if not status:
            return None

        # Collect failed or un-processed items
        failed_ids = [e.get("video_id") for e in status.get("errors", []) if e.get("video_id")]
        if not failed_ids:
            logger.info("Job %s has no failed items to resume.", job_id)
            return job_id

        logger.info("Resuming %d failed items from job %s", len(failed_ids), job_id)
        return self.submit_batch(failed_ids, resume=True, force=True)

    def get_job_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieve current progress of a batch job from memory or disk checkpoint."""
        if job_id in self._jobs:
            return self._jobs[job_id]

        checkpoint_file = self.checkpoint_dir / f"job_{job_id}.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._jobs[job_id] = data
                    return data
            except Exception as e:
                logger.warning("Failed to read checkpoint for job %s: %s", job_id, e)
        return None

    def list_jobs(self) -> list[dict[str, Any]]:
        """List recent batch jobs from memory and disk checkpoints."""
        jobs_map = dict(self._jobs)
        for cp in self.checkpoint_dir.glob("job_*.json"):
            jid = cp.stem.replace("job_", "")
            if jid not in jobs_map:
                try:
                    with open(cp, "r", encoding="utf-8") as f:
                        jobs_map[jid] = json.load(f)
                except Exception:
                    pass
        return list(jobs_map.values())
