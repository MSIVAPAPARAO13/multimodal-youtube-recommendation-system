"""
FastAPI integration tests for GET /api/youtube/video/{video_id}.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.integrations.youtube.schemas import YouTubeVideoMetadata
from src.integrations.youtube.exceptions import (
    YouTubeAuthenticationError,
    YouTubeNotFoundError,
    YouTubePrivateVideoError,
    YouTubeQuotaExceededError,
)

SAMPLE_METADATA = YouTubeVideoMetadata(
    video_id="dQw4w9WgXcQ",
    title="Rick Astley - Never Gonna Give You Up",
    description="Official music video description",
    channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
    channel_title="Rick Astley",
    published_at="2009-10-25T06:57:33Z",
    category_id="10",
    tags=["rick astley", "80s"],
    thumbnail_default="https://i.ytimg.com/vi/dQw4w9WgXcQ/default.jpg",
    thumbnail_medium="https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg",
    thumbnail_high="https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
    duration="PT3M33S",
    duration_seconds=213,
    view_count=1500000000,
    like_count=17000000,
    comment_count=2500000,
    privacy_status="public",
)


def test_youtube_video_endpoint_success(client):
    """Verify GET /api/youtube/video/{video_id} returns 200 and matches expected structure."""
    with patch(
        "src.integrations.youtube.service.YouTubeMetadataService.get_video_metadata",
        return_value=SAMPLE_METADATA,
    ):
        response = client.get("/api/youtube/video/dQw4w9WgXcQ")
        assert response.status_code == 200

        data = response.json()
        assert data["video_id"] == "dQw4w9WgXcQ"
        assert data["title"] == "Rick Astley - Never Gonna Give You Up"
        assert data["channel_title"] == "Rick Astley"
        assert data["tags"] == ["rick astley", "80s"]
        assert "thumbnail" in data
        assert data["thumbnail"]["high"] == "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
        assert "statistics" in data
        assert data["statistics"]["views"] == 1500000000
        assert data["statistics"]["likes"] == 17000000
        assert data["statistics"]["comments"] == 2500000


def test_youtube_video_endpoint_with_full_url(client):
    """Verify endpoint accepts encoded full URLs."""
    with patch(
        "src.integrations.youtube.service.YouTubeMetadataService.get_video_metadata",
        return_value=SAMPLE_METADATA,
    ):
        url_input = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        response = client.get(f"/api/youtube/video/{url_input}")
        assert response.status_code == 200
        assert response.json()["video_id"] == "dQw4w9WgXcQ"


def test_youtube_video_endpoint_invalid_id(client):
    """Verify endpoint returns 400 Bad Request on invalid ID."""
    response = client.get("/api/youtube/video/too_short")
    assert response.status_code == 400
    assert "invalid youtube video" in response.json()["detail"].lower()


def test_youtube_video_endpoint_not_found(client):
    """Verify endpoint returns 404 Not Found when video does not exist."""
    with patch(
        "src.integrations.youtube.service.YouTubeMetadataService.get_video_metadata",
        side_effect=YouTubeNotFoundError("nonExist123"),
    ):
        response = client.get("/api/youtube/video/nonExist123")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def test_youtube_video_endpoint_private_video(client):
    """Verify endpoint returns 403 Forbidden on private or restricted video."""
    with patch(
        "src.integrations.youtube.service.YouTubeMetadataService.get_video_metadata",
        side_effect=YouTubePrivateVideoError("privateId123"),
    ):
        response = client.get("/api/youtube/video/privateId123")
        assert response.status_code == 403
        assert "private" in response.json()["detail"].lower()


def test_youtube_video_endpoint_auth_failure(client):
    """Verify endpoint returns 401 Unauthorized on invalid API key."""
    with patch(
        "src.integrations.youtube.service.YouTubeMetadataService.get_video_metadata",
        side_effect=YouTubeAuthenticationError(),
    ):
        response = client.get("/api/youtube/video/dQw4w9WgXcQ")
        assert response.status_code == 401
        assert "authentication failed" in response.json()["detail"].lower()


def test_youtube_video_endpoint_quota_exceeded(client):
    """Verify endpoint returns 429 Too Many Requests when quota is exhausted."""
    with patch(
        "src.integrations.youtube.service.YouTubeMetadataService.get_video_metadata",
        side_effect=YouTubeQuotaExceededError(),
    ):
        response = client.get("/api/youtube/video/dQw4w9WgXcQ")
        assert response.status_code == 429
        assert "quota" in response.json()["detail"].lower()


def test_youtube_video_endpoint_disabled(client):
    """Verify endpoint returns 503 when YouTube integration is disabled."""
    with patch("src.api.app.settings.YOUTUBE_API_ENABLED", False):
        response = client.get("/api/youtube/video/dQw4w9WgXcQ")
        assert response.status_code == 503
        assert "disabled" in response.json()["detail"].lower()


def test_api_key_never_leaked_in_response(client):
    """Verify secret API key never appears in response body or headers."""
    secret_key = "AIzaSySuperSecretKeyNotForPublic"
    with patch("src.api.app.settings.YOUTUBE_API_KEY", secret_key), \
         patch(
             "src.integrations.youtube.service.YouTubeMetadataService.get_video_metadata",
             return_value=SAMPLE_METADATA,
         ):
        response = client.get("/api/youtube/video/dQw4w9WgXcQ")
        response_text = response.text
        assert secret_key not in response_text
        assert secret_key not in str(response.headers)
