# Phase 4 — YouTube Ingestion & Media Pipeline Documentation

## 1. Overview & Ingestion Lifecycle

The Phase 4 YouTube Ingestion & Media Pipeline establishes an asynchronous, idempotent, production-ready ingestion pipeline. It ingests validated YouTube video identifiers or URLs, retrieves and normalizes metadata via YouTube Data API v3, downloads bounded resolution media via `yt-dlp`, validates media constraints, extracts 16 kHz mono waveforms suitable for Wav2Vec2, downloads thumbnails, and handles captions/transcripts gracefully.

### Pipeline Lifecycle Stages

```
YouTube Video ID / URL
       ↓ (Parser validation)
Canonical YouTube ID (11 chars)
       ↓ (Registration)
VideoIngestion Record (PENDING)
       ↓ (Asynchronous Background Task)
Fetch & Normalize YouTube Metadata
       ↓ (Persistence)
Video Entity Created / Updated & Status (METADATA_READY)
       ↓ (Media Acquisition via yt-dlp)
Download Low-Res Video (<=360p) & Status (MEDIA_DOWNLOADING)
       ↓ (Media Validation)
Validate Size, Dimensions, Duration, and Decodability
       ↓ (Audio Extraction)
16 kHz Mono WAV Extraction via AudioProcessor & wave
       ↓ (Thumbnail & Transcript Acquisition)
Validate Domain & Download Thumbnail + Optional Captions
       ↓ (Ingestion Finalized)
Persist metadata.json & Ingestion Status: MEDIA_READY
```

### State Transitions

The pipeline transitions sequentially through the following states defined in `IngestionStatus`:
- `PENDING`: Ingestion request registered and scheduled for asynchronous background execution.
- `METADATA_READY`: YouTube Data API v3 metadata fetched, normalized, and persisted to PostgreSQL/SQLite `videos` table.
- `MEDIA_DOWNLOADING`: Video download initiated via controlled `yt-dlp` wrapper.
- `MEDIA_READY`: Video downloaded, container validated, 16 kHz audio extracted, thumbnail stored, and ready for feature extraction.
- `PROCESSING`: Reserved for multimodal feature extraction / embedding generation (Phase 5).
- `COMPLETED`: Ingestion and downstream indexing completed.
- `FAILED`: Unrecoverable failure or maximum retries exceeded. Terminal errors halt immediately with root-cause diagnostic message.

---

## 2. Database Schema & Migration Strategy

Phase 4 introduces minimal, production-oriented schema updates that maintain full backward compatibility with PostgreSQL + pgvector as well as SQLite in-memory test environments.

### `videos` Table Additions
Extended the existing `Video` entity to store rich YouTube metadata alongside recommendation embeddings:
- `channel_id` (`VARCHAR(255)`): YouTube channel identifier.
- `channel_title` (`VARCHAR(255)`): YouTube channel name.
- `published_at` (`TIMESTAMP`): Original video publication timestamp.
- `category_id` (`VARCHAR(64)`): YouTube content category identifier.
- `thumbnail_url` (`VARCHAR(1000)`): URL of high-resolution video thumbnail.
- `duration` (`INTEGER`): Video duration in seconds.
- `view_count` (`BIGINT`): Total video view count.
- `like_count` (`BIGINT`): Total video like count.
- `comment_count` (`BIGINT`): Total video comment count.
- `privacy_status` (`VARCHAR(50)`): Visibility status (`public`, `unlisted`, `private`).
- `ingestion_status` (`VARCHAR(50)`): Current lifecycle state (default `PENDING`).
- `ingestion_error` (`TEXT`): Diagnostic error message if ingestion fails.
- `last_ingestion_attempt` (`TIMESTAMP`): Timestamp of the most recent ingestion attempt.

### `video_ingestions` Table
Dedicated tracking table for idempotent lifecycle management and retry accounting:
- `id` (`UUID`, Primary Key): Unique ingestion job identifier.
- `video_id` (`VARCHAR(255)`): Internal reference video identifier (`yt_{video_id}`).
- `youtube_video_id` (`VARCHAR(64)`, Unique, Indexed): Canonical 11-character YouTube video ID.
- `status` (`VARCHAR(50)`, Indexed): State enum (`PENDING`, `METADATA_READY`, `MEDIA_DOWNLOADING`, `MEDIA_READY`, `PROCESSING`, `COMPLETED`, `FAILED`).
- `attempt_count` (`INTEGER`, Default `0`): Number of processing attempts made.
- `max_attempts` (`INTEGER`, Default `3`): Configurable retry ceiling.
- `error_message` (`TEXT`): Last encountered failure diagnostic.
- `video_path` (`VARCHAR(1000)`): Local path to acquired video file.
- `audio_path` (`VARCHAR(1000)`): Local path to extracted 16 kHz mono WAV file.
- `thumbnail_path` (`VARCHAR(1000)`): Local path to downloaded thumbnail image.
- `transcript_text` (`TEXT`): Full text transcript if available.
- `started_at` (`TIMESTAMP`): Execution start timestamp.
- `completed_at` (`TIMESTAMP`): Execution completion timestamp.
- `created_at` (`TIMESTAMP`): Record creation timestamp.
- `updated_at` (`TIMESTAMP`): Record update timestamp.

### Migration & Compatibility
- Cross-database compatibility is preserved using existing custom type decorators: `CompatibleUUID`, `CompatibleArray`, and `CompatibleEmbedding`.
- `scripts/init_db.sql` includes DDL for `video_ingestions`, new columns for `videos`, and corresponding indexes.

---

## 3. Media Storage Architecture

All acquired media assets are placed inside a tightly controlled, segregated directory tree:
```
data/
  raw/
    youtube/
      {video_id}/
        video.mp4      <- Bounded resolution container (<=360p)
        audio.wav      <- 16 kHz mono PCM WAV
        thumbnail.jpg  <- Validated thumbnail image
        metadata.json  <- Normalized metadata payload
```

### Path Traversal & Security Isolation
- Implemented `LocalStorage` under `src/storage/local.py` adhering to `BaseStorage` (`src/storage/base.py`).
- Strict validation via regex `^[a-zA-Z0-9_-]{1,64}$` for video IDs.
- Resolves all paths and verifies containment using `path.relative_to(self.root_path)`. Any traversal attempt (`../`, absolute foreign paths) raises `StorageSecurityError`.
- Media paths are completely excluded from Git via `.gitignore` (`data/raw/youtube/`, `*.mp4`, `*.wav`, `*.m4a`, `*.webm`).

---

## 4. Download Strategy

The multimodal pipeline requires only:
1. Sample video frames (ResNet-50 requires 8 frames sized 224x224).
2. Audio waveform (Wav2Vec2 requires 16 kHz mono audio).

### Format Specification
Downloading high-bitrate 1080p or 4K streams is wasteful and prohibited. `MediaDownloader` enforces the following `yt-dlp` format selection:
```python
ydl_opts = {
    "format": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360]/bestvideo[height<=360]+bestaudio/worst",
    "outtmpl": str(target_video_path),
    "merge_output_format": "mp4",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "socket_timeout": 60.0,
    "max_filesize": 100 * 1024 * 1024,
    "overwrites": True,
}
```
- Maximum video resolution is clamped at **360p**, which minimizes bandwidth consumption by ~90% while providing more than enough pixel density for the 224x224 frame extractor.
- `max_filesize` prevents disk exhaustion attacks from maliciously large video streams.
- `noplaylist` prevents inadvertent crawling of entire channels or playlists.

---

## 5. Media Validation

Implemented in `src/integrations/youtube/media/validator.py`:
- **File Existence & Non-Zero Size**: Verifies file exists and has size > 0.
- **Configurable Size Limits**:
  - `MAX_VIDEO_SIZE_MB` (Default: 100 MB)
  - `MAX_AUDIO_SIZE_MB` (Default: 50 MB)
- **Container & Readability**: Uses OpenCV (`cv2.VideoCapture`) to inspect stream headers, check frame count > 0, width > 0, height > 0, and decodes the initial frame.
- **Duration Checks**: Bounded by `MAX_VIDEO_DURATION_SECONDS` (Default: 600s / 10 minutes). Longer videos are rejected before processing.
- **Error Specialization**: Specific exceptions are raised: `MediaSizeExceededError`, `MediaDurationExceededError`, `MediaCorruptError`, and `MediaValidationError`.

---

## 6. Audio Extraction

Implemented in `src/integrations/youtube/media/audio.py`:
- Reuses existing `AudioProcessor` from `src.data.processors` to preserve standardized audio processing logic.
- Converts waveform tensor into standard 16-bit PCM mono WAV at 16 kHz.
- Uses Python's native `wave` library to serialize the output file without requiring external ffmpeg binary dependencies.
- Handles missing audio tracks, silent files, NaNs, and Infs defensively using `np.nan_to_num` and `np.clip`.

---

## 7. Thumbnail Handling

Implemented in `src/integrations/youtube/media/thumbnails.py`:
- Validates thumbnail URL hostname against allowed YouTube CDN domains (`ytimg.com`, `i.ytimg.com`, `youtube.com`, `img.youtube.com`, `googleusercontent.com`).
- Validates HTTP response status (`200 OK`) and `Content-Type` header (must begin with `image/`).
- Enforces `THUMBNAIL_MAX_SIZE_MB` (Default: 10 MB) during stream reading.
- Failures during thumbnail retrieval log warnings but do not fail the core media ingestion pipeline.

---

## 8. Transcript Strategy

Implemented in `src/integrations/youtube/transcript.py`:
- Abstract interface `TranscriptProvider` with concrete implementation `YouTubeTranscriptProvider`.
- Uses `youtube-transcript-api` to fetch subtitles in preferred languages (default `['en']`).
- Handles `TranscriptsDisabled`, `NoTranscriptFound`, and `VideoUnavailable` without raising fatal exceptions.
- **Transcript Availability is NOT Mandatory**: The multimodal recommendation system operates with vision + audio + metadata text even when transcripts are unavailable.

---

## 9. Asynchronous Execution & APIs

Long-running downloads are never executed in the FastAPI request event loop.
- `POST /api/youtube/ingest`:
  - Validates video ID via `extract_video_id`.
  - Creates or retrieves `VideoIngestion` record.
  - Schedules background job via FastAPI `BackgroundTasks`.
  - Returns immediately with `HTTP 202 Accepted` and status `PENDING`.
- `GET /api/youtube/ingest/{video_id}`:
  - Returns current ingestion progress, attempt count, and error state.
  - Strictly omits internal filesystem paths (`video_path`, `audio_path`, `thumbnail_path`) from response payloads.

---

## 10. Retry Policy & Idempotency

Implemented in `YouTubeIngestionService` and `YouTubeVideoRepository`:
- **Idempotency**: Repeated calls to `/api/youtube/ingest` for the same YouTube ID will not produce duplicate records in `videos` or `video_ingestions`.
- **Transient Failures**: Network timeouts, connection drops, or temporary ffmpeg/yt-dlp glitches increment `attempt_count` and remain retryable up to `INGESTION_MAX_RETRIES` (Default: 3).
- **Terminal Failures**: Permanently invalid requests (`YouTubeInvalidVideoIdError`, `YouTubeNotFoundError`, `YouTubePrivateVideoError`, `MediaSizeExceededError`, `MediaDurationExceededError`) immediately transition to `FAILED` without retrying.
- **Rollback Safety**: Database sessions are rolled back immediately on exception before recording the failure state, preventing half-committed transaction locks.

---

## 11. Cleanup Strategy

Implemented in `LocalStorage.clean_incomplete_files()` and `LocalStorage.delete_video_dir()`:
- Automatically scans and removes partial download artifacts (`*.part`, `*.ytdl`, `*.tmp`) prior to download attempts and upon failure.
- `delete_video_dir` enables atomic cleanup of an entire video folder when unrecoverable errors occur.

---

## 12. Security Controls Summary

1. **Input Validation**: Only valid canonical YouTube video IDs (11 base64 characters) are accepted.
2. **Path Sanitization**: Directory traversal is blocked using `path.resolve()` and `relative_to(root)`.
3. **SSRF & URL Injection Protection**: `yt-dlp` and `ThumbnailDownloader` only accept canonical YouTube watch URLs and whitelisted CDN domains.
4. **Information Leakage Prevention**: Internal filesystem paths are excluded from API schemas. YouTube API keys are masked and never logged.
5. **Resource Limiting**: Enforced bounded video resolution (<=360p), file size ceilings, and execution timeouts.

---

## 13. Test Results & Verification

- **Total Suite Passing**: 79/79 tests passed (100%).
- **Phase 4 Tests**: 16 dedicated unit and integration tests in `tests/test_youtube_ingestion.py`.
- **External Mocking**: Full test suite operates without internet access and without downloading real media.
