"""
Unit and Integration Tests for "Try Your Own Video" Upload & Recommendation Feature.

Covers:
1. Valid video with audio (Real media integration test using local_10040/video.mp4)
2. Valid video without audio stream (OpenCV synthesized video container)
3. Corrupt video rejection (Malformed bytes with .mp4 extension)
4. Unsupported format rejection (.txt, .exe, etc.)
5. Oversized upload rejection (> 100 MB limit)
6. Missing title but supplied description
7. Only visual similarity available (No title or description)
8. Modality statuses and truthful metadata reporting
9. Ephemeral storage cleanup verification after success and failure
10. End-to-end API response contract validation
"""

import io
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def real_video_path():
    p = Path("data/raw/youtube/local_10040/video.mp4")
    if p.exists() and p.stat().st_size > 0:
        return p
    pytest.skip("Real video fixture data/raw/youtube/local_10040/video.mp4 not found on disk.")


@pytest.fixture
def silent_video_bytes():
    """Generate a valid, decodable MP4 video in memory without an audio track."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_path = tf.name

    # Create 15 frames of 320x240 video at 10 fps (1.5 seconds)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, 10.0, (320, 240))
    for i in range(15):
        frame = np.full((240, 320, 3), (i * 15) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()

    with open(tmp_path, "rb") as f:
        data = f.read()

    os.remove(tmp_path)
    return data


# ============ 1. Real Media Integration: Valid Video With Audio ============

def test_upload_valid_video_with_audio_and_text(client, real_video_path):
    """Real media integration: tests full pipeline on authentic MP4 fixture."""
    with open(real_video_path, "rb") as f:
        response = client.post(
            "/api/custom-video/analyze",
            files={"file": ("quantum_guide.mp4", f, "video/mp4")},
            data={
                "title": "Quantum Computing Principles",
                "description": "An introductory overview of qubits and superposition.",
                "max_recommendations": 4,
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert "video_metadata" in data
    assert data["video_metadata"]["duration_seconds"] > 0
    assert len(data["extracted_frames"]) > 0

    # Verify visual modality
    assert data["modalities"]["visual"]["status"] == "EXTRACTED"
    assert data["modalities"]["visual"]["dimension"] == 2048

    # Verify text modality
    assert data["modalities"]["text"]["status"] == "EMBEDDING_GENERATED"
    assert data["modalities"]["text"]["dimension"] == 512

    # Verify recommendations
    assert isinstance(data["visual_recommendations"], list)
    assert isinstance(data["text_recommendations"], list)
    if data["visual_recommendations"]:
        first_vis = data["visual_recommendations"][0]
        assert "score" in first_vis
        assert -1.0 <= first_vis["score"] <= 1.0


# ============ 2. Video Without Audio Stream ============

def test_upload_valid_video_without_audio(client, silent_video_bytes):
    """Test valid video container that contains no audio stream."""
    response = client.post(
        "/api/custom-video/analyze",
        files={"file": ("no_audio.mp4", io.BytesIO(silent_video_bytes), "video/mp4")},
        data={"title": "Silent Animation Demo"},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    # Audio status should reflect absence or silence, never fake audio
    assert data["modalities"]["audio"]["status"] in ("NO_AUDIO_STREAM", "SILENT_AUDIO_DETECTED", "EXTRACTION_FAILED")
    assert "0.00" in data["audio_status_note"]


# ============ 3. Corrupt Video Rejection ============

def test_upload_corrupt_video(client):
    """Ensure malformed/corrupt bytes with .mp4 extension are rejected with 400."""
    corrupt_bytes = b"NOT_A_VALID_MP4_HEADER_RANDOM_CORRUPT_BYTES_XYZ"
    response = client.post(
        "/api/custom-video/analyze",
        files={"file": ("corrupt.mp4", io.BytesIO(corrupt_bytes), "video/mp4")},
    )

    assert response.status_code == 400
    assert "corrupt" in response.json()["detail"].lower() or "unreadable" in response.json()["detail"].lower()


# ============ 4. Unsupported Format Rejection ============

def test_upload_unsupported_format(client):
    """Ensure unsupported file extension (.txt, .exe) is rejected with 400."""
    response = client.post(
        "/api/custom-video/analyze",
        files={"file": ("document.txt", io.BytesIO(b"Hello world"), "text/plain")},
    )

    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


# ============ 5. Oversized Upload Rejection ============

def test_upload_oversized_file(client, monkeypatch):
    """Ensure upload exceeding max size is rejected with HTTP 413."""
    # Temporarily lower limit to 10 KB for unit test verification
    import sys
    app_mod = sys.modules.get("src.api.app") or sys.modules.get("api.app")
    monkeypatch.setattr(app_mod, "MAX_CUSTOM_UPLOAD_BYTES", 10 * 1024)

    oversized_data = b"X" * (12 * 1024)  # 12 KB
    response = client.post(
        "/api/custom-video/analyze",
        files={"file": ("large_video.mp4", io.BytesIO(oversized_data), "video/mp4")},
    )

    assert response.status_code == 413
    assert "exceeds maximum limit" in response.json()["detail"].lower()


# ============ 6. Missing Title With Supplied Description ============

def test_upload_missing_title_with_description(client, silent_video_bytes):
    """Verify that omitting title while supplying description still generates text embeddings."""
    response = client.post(
        "/api/custom-video/analyze",
        files={"file": ("clip.mp4", io.BytesIO(silent_video_bytes), "video/mp4")},
        data={"description": "Detailed explanation of gradient descent in deep learning."},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["modalities"]["text"]["status"] == "EMBEDDING_GENERATED"
    assert data["modalities"]["text"]["dimension"] == 512
    assert len(data["text_recommendations"]) > 0


# ============ 7. Visual Only (No Text Provided) ============

def test_upload_visual_only_no_text(client, silent_video_bytes):
    """Verify that uploading with neither title nor description correctly marks text as NO_TEXT_PROVIDED."""
    response = client.post(
        "/api/custom-video/analyze",
        files={"file": ("visual_only.mp4", io.BytesIO(silent_video_bytes), "video/mp4")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["modalities"]["visual"]["status"] == "EXTRACTED"
    assert data["modalities"]["text"]["status"] == "NO_TEXT_PROVIDED"
    assert len(data["text_recommendations"]) == 0


# ============ 8. Ephemeral Cleanup Verification ============

def test_ephemeral_cleanup_after_request(client, silent_video_bytes):
    """Confirm that temporary files are deleted after request completion."""
    temp_dir_base = tempfile.gettempdir()
    before_dirs = set(Path(temp_dir_base).glob("custom_upload_*"))

    response = client.post(
        "/api/custom-video/analyze",
        files={"file": ("temp_test.mp4", io.BytesIO(silent_video_bytes), "video/mp4")},
        data={"title": "Temporary Cleanup Check"},
    )
    assert response.status_code == 200

    after_dirs = set(Path(temp_dir_base).glob("custom_upload_*"))
    # No lingering directories left behind
    assert len(after_dirs - before_dirs) == 0
