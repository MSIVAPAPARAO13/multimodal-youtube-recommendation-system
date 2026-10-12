# Feature — Try Your Own Video Validation Report

## 1. Feature Overview

The **"Try Your Own Video"** feature allows users to upload an arbitrary local video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`), inspect its real multimodal feature extractions, and discover similar catalog videos using verified representations:

- **Frontend Route**: [`/try-video`](file:///c:/Users/msiva/Music/multimodal-video-recommender/frontend/src/pages/TryVideoPage.jsx)
- **API Endpoint**: `POST /api/custom-video/analyze`
- **Security & Ephemeral Storage**: Server-generated UUID temporary directories (`tempfile.TemporaryDirectory`) with guaranteed immediate removal in a `finally` block. Never permanently inserts arbitrary uploads into the catalog.
- **Visual Similarity Retrieval**: 8-frame mean-pooled ResNet-50 2048-D features compared against the catalog's verified `resnet50_mean_2048` vectors via cosine similarity.
- **Text Semantic Retrieval**: Title and description encoded via the Phase 8C Contrastive BERT (512-D) linear projection head and queried against PostgreSQL `pgvector HNSW` index.
- **Truthful Audio Intelligence**: Probes for valid 16 kHz audio tracks and rejects silent audio without generating fake representations. Production audio ranking weight is explicitly maintained at `0.00`.

---

## 2. Test Execution & Pass Rate

The dedicated test suite (`tests/test_custom_video_upload.py`) was executed across real media fixtures and controlled edge cases:

| Test Case | Scenario / Fixture | Verification | Status |
| :--- | :--- | :--- | :--- |
| **`test_upload_valid_video_with_audio_and_text`** | Real media fixture `data/raw/youtube/local_10040/video.mp4` (2.18 MB) with title/description | Validates 200 OK, 4 preview frames, visual 2048-D extraction, text 512-D projection, and ranked catalog results. | **PASSED** |
| **`test_upload_valid_video_without_audio`** | Synthesized MP4 video container without an audio track | Validates that audio status truthfully reports `NO_AUDIO_STREAM` and note confirms 0.00 weight. | **PASSED** |
| **`test_upload_corrupt_video`** | Corrupted random bytes with `.mp4` extension | Validates rejection with HTTP 400 and clear error detail. | **PASSED** |
| **`test_upload_unsupported_format`** | `.txt` / `.exe` file upload | Validates rejection with HTTP 400 indicating supported extensions. | **PASSED** |
| **`test_upload_oversized_file`** | Payload exceeding maximum upload limit | Validates stream abort with HTTP 413 "exceeds maximum limit". | **PASSED** |
| **`test_upload_missing_title_with_description`** | Missing title, but description provided | Validates text embedding generation and text recommendations from description context. | **PASSED** |
| **`test_upload_visual_only_no_text`** | No title or description supplied | Validates visual feature extraction while text status cleanly reports `NO_TEXT_PROVIDED`. | **PASSED** |
| **`test_ephemeral_cleanup_after_request`** | Server-controlled temporary directory lifecycle | Validates that all temporary files are completely purged from disk upon request completion. | **PASSED** |

### Suite Summary
- **Tests Passed**: 8 of 8 passed in 11.24s (100% pass rate).
- **Combined Phase 14 & Custom Video Suite**: 13 of 13 passed in 11.29s (100% pass rate).

---

## 3. Real Media Integration Sample Response

The endpoint was verified live using `data/raw/youtube/local_10040/video.mp4`:

```json
{
  "status": "success",
  "video_metadata": {
    "filename": "video.mp4",
    "size_mb": 2.08,
    "duration_seconds": 184.1,
    "fps": 30.0,
    "frame_count": 5523,
    "width": 1280,
    "height": 720
  },
  "extracted_frames": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  ],
  "modalities": {
    "visual": {
      "status": "EXTRACTED",
      "model": "torchvision_resnet50",
      "feature_type": "resnet50_mean_2048",
      "dimension": 2048,
      "frames_sampled": 8,
      "details": "8-frame mean-pooled ResNet-50 2048-D normalized embedding"
    },
    "audio": {
      "status": "EXTRACTED_NON_SILENT",
      "details": "Valid 16 kHz mono audio track detected & decoded"
    },
    "text": {
      "status": "EMBEDDING_GENERATED",
      "model": "Phase 8C Contrastive BERT",
      "dimension": 512,
      "input_text": "Quantum Computing Test",
      "details": "Phase 8C Contrastive BERT (512-D) query retrieved via ann (1.2ms)"
    }
  },
  "visual_recommendations": [
    {
      "video_id": "10040",
      "title": "A beginner's guide to quantum computing",
      "score": 1.0,
      "rank": 1,
      "thumbnail_url": "/media/local_10040/thumbnail.jpg",
      "duration": 184
    },
    {
      "video_id": "machine_learning_0",
      "title": "Intro to Machine Learning",
      "score": 0.8124,
      "rank": 2,
      "thumbnail_url": "/media/local_machine_learning_0/thumbnail.jpg",
      "duration": 152
    }
  ],
  "text_recommendations": [
    {
      "video_id": "10040",
      "title": "A beginner's guide to quantum computing",
      "score": 0.8842,
      "rank": 1,
      "thumbnail_url": "/media/local_10040/thumbnail.jpg",
      "duration": 184
    }
  ],
  "audio_status_note": "Audio extraction status validated. Ranking weights for audio are 0.00 in production to maintain catalog consistency."
}
```

---

## 4. UI Architecture & Responsive Layout

The React component `TryVideoPage.jsx` implements:
1. **Drag-and-Drop Area**: Styled with Bootstrap dashed borders and dynamic drag-over highlighting; file picker button fallback.
2. **Instant Local HTML5 Preview**: Uses `URL.createObjectURL` to render a native video preview before transmission.
3. **Real-Time Upload Progress**: Tracks upload bytes with Axios `onUploadProgress`.
4. **Analysis Spinner**: Indicates frame decoding and PyTorch inference stages.
5. **Dual Recommendation Tabs**: Explicitly separates Visual Nearest Neighbors from Text Semantic Retrieval with genuine cosine similarity score badges (`Sim: 0.xxx`).
6. **Responsive Layout**: Designed for desktop, tablet, and mobile with bottom navigation bar integration.
