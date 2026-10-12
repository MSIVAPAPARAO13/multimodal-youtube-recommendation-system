# Multimodal Asset Ingestion and Alignment Pipeline (Phase 12A)

## 1. Executive Summary

Phase 12A establishes a production-grade, idempotent data and asset alignment layer for the **Multimodal YouTube Recommendation System**. It connects external YouTube video assets and local catalog video recordings with database metadata, preparing validated video frames, audio waveforms, thumbnails, and JSON metadata manifests for genuine multimodal representation learning in Phase 12B.

> [!IMPORTANT]
> **Data Preparation Boundary**: Phase 12A extracts, aligns, validates, and persists physical media assets. It **does not** compute multimodal embeddings or activate visual/acoustic weights in recommendation ranking. Feature extraction and recommendation scoring validation occur strictly in Phase 12B.

---

## 2. Identifier Mapping Architecture

The recommendation system manages four distinct identifier planes:

| Identifier Plane | Field / Key | Scope & Format | Example | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Database Primary Key** | `Video.id` | RFC 4122 UUID | `9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d` | Internal referential integrity |
| **Catalog Video ID** | `Video.video_id` | `VARCHAR(255)` string | `"1002"`, `"yt_dQw4w9WgXcQ"` | Unique catalog identifier used in retrieval & index |
| **YouTube Video ID** | `Video.youtube_video_id` | 11-char alphanumeric | `"dQw4w9WgXcQ"` | External YouTube Data API & video download identifier |
| **Ingestion Job ID** | `VideoIngestion.id` | RFC 4122 UUID | `4a3b8c1d-...` | Asynchronous worker tracking & attempt budgets |

### Alignment Rules:
1. When a YouTube URL or 11-character video ID is ingested, the system canonicalizes the ID via `extract_video_id` and records `youtube_video_id`. The internal catalog `video_id` defaults to `yt_{youtube_video_id}` unless explicitly mapped to an existing catalog ID.
2. For local catalog videos (e.g. MicroLens items), `video_id` retains the catalog integer identifier (e.g., `"1002"`), while storage and ingestion records use `local_{video_id}` or an aligned YouTube ID.
3. Every physical asset is linked to `video_id` in the `video_assets` table with an explicit `asset_type` enum.

---

## 3. Storage Hierarchy & Asset Layout

All media assets are stored safely within `MEDIA_STORAGE_DIR` (default: `data/raw/youtube/`) with directory traversal protection:

```text
data/raw/youtube/
└── {target_id}/                       # e.g., dQw4w9WgXcQ or local_1002
    ├── video.mp4                      # Low-resolution source video (<=360p)
    ├── audio.wav                      # Extracted 16 kHz mono PCM WAV
    ├── thumbnail.jpg                  # Verified JPEG thumbnail (224x224+)
    ├── metadata.json                  # Normalized metadata manifest
    ├── frames_metadata.json           # Extracted frame index & timestamps
    └── frames/                        # Deterministically sampled frames
        ├── frame_000.jpg              # Timestamp 0.00s, SHA-256 verified
        ├── frame_001.jpg              # Timestamp 0.25s, SHA-256 verified
        ├── ...
        └── frame_007.jpg              # Timestamp 1.75s, SHA-256 verified
```

---

## 4. Media Processing & Validation Specifications

### 4.1 Video Container Validation (`MediaValidator`)
- **Format:** Container format verified via OpenCV `VideoCapture`.
- **Dimensions:** Width and height must be $> 0$; frame count $> 0$.
- **Size Limit:** Max video size bounded by `MAX_VIDEO_SIZE_MB` (default 100 MB).
- **Duration Limit:** Max duration bounded by `MAX_VIDEO_DURATION_SECONDS` (default 600 seconds).
- **Decodability:** Initial frame decode read is verified prior to asset approval.

### 4.2 Audio Extraction & Alignment (`AudioExtractor`)
- **Waveform Format:** 16-bit PCM WAV.
- **Channels:** Single channel (**mono**). Rejects multi-channel audio without downmixing.
- **Sample Rate:** **16,000 Hz** (native requirement for Wav2Vec2 audio models).
- **Signal Integrity:** Verified non-zero frames using standard library `wave`. Empty audio triggers `MediaCorruptError`.
- **Missing Audio Policy:** Videos without readable audio tracks log `AUDIO_EXTRACTION_UNAVAILABLE` rather than fabricating synthetic noise or silence.

### 4.3 Visual Frame Extraction (`FrameSampler`)
- **Sampling Strategy:** Evenly spaced deterministic frame selection:
  $$\text{frame\_indices} = \text{linspace}(0, N_{\text{total}} - 1, N_{\text{target}})$$
- **Default Frame Count:** 8 frames (matches ResNet-50 / VideoProcessor expectations).
- **Timestamps:** Computed precisely as $t_i = \frac{\text{frame\_index}_i}{\text{fps}}$.
- **Checksums:** Individual SHA-256 hashes generated and stored per frame.

---

## 5. Database Schema (`video_assets` Table)

```sql
CREATE TABLE video_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id VARCHAR(255) NOT NULL,
    youtube_video_id VARCHAR(64),
    asset_type VARCHAR(32) NOT NULL,          -- 'VIDEO', 'AUDIO', 'THUMBNAIL', 'FRAME', 'METADATA'
    source_url VARCHAR(1000),
    storage_path VARCHAR(1000) NOT NULL,       -- Relative path inside media storage root
    mime_type VARCHAR(64),
    file_size_bytes BIGINT,
    checksum_sha256 VARCHAR(64),
    status VARCHAR(32) NOT NULL DEFAULT 'READY',
    attempt_count INTEGER NOT NULL DEFAULT 1,
    error_category VARCHAR(64),
    error_message TEXT,
    pipeline_version VARCHAR(32) NOT NULL DEFAULT '12A',
    extra_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 6. Job Orchestration & Bounded Concurrency

Batch operations are governed by `AssetIngestionOrchestrator`:
- **Worker Concurrency:** Bounded `ThreadPoolExecutor(max_workers=2)` prevents memory exhaustion and storage I/O thrashing.
- **Batch Size Limit:** Capped at 50 videos per batch request (`BatchIngestRequest.video_ids`).
- **Failure Isolation:** Each video execution is wrapped in isolated transaction boundaries; a failure in video $i$ never halts execution for video $i+1$.
- **Idempotency:** When source checksum matches existing valid assets, reprocessing is safely skipped without duplicate database records.

---

## 7. API Endpoints

| Method | Route | Description | Request / Response |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/youtube/ingest` | Ingest single video asynchronously | `YouTubeIngestRequest` $\to$ `202 Accepted` |
| `GET` | `/api/youtube/ingest/{id}` | Ingestion lifecycle & error status | `YouTubeIngestStatusResponse` |
| `GET` | `/api/youtube/ingest` | List recent ingestion jobs (paginated) | `limit`, `offset`, `status_filter` $\to$ `IngestionListResponse` |
| `POST` | `/api/youtube/ingest/batch` | Submit bounded batch of videos | `BatchIngestRequest` $\to$ `BatchIngestResponse` |
| `GET` | `/api/youtube/assets/{id}` | List registered multimodal assets | `VideoAssetsResponse` |
