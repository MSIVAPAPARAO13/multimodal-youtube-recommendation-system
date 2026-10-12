"""
Unit tests for YouTubeAPIClient with mocked HTTP requests.
"""

from unittest.mock import MagicMock, patch
import pytest
import httpx

from src.integrations.youtube.client import YouTubeAPIClient
from src.integrations.youtube.exceptions import (
    YouTubeAPIError,
    YouTubeAuthenticationError,
    YouTubeConfigurationError,
    YouTubeNotFoundError,
    YouTubePrivateVideoError,
    YouTubeQuotaExceededError,
)

SAMPLE_SUCCESS_ITEM = {
    "id": "dQw4w9WgXcQ",
    "snippet": {
        "title": "Rick Astley - Never Gonna Give You Up",
        "description": "Music video description",
        "channelId": "UCuAXFkgsw1L7xaCfnd5JJOw",
        "channelTitle": "Rick Astley",
        "publishedAt": "2009-10-25T06:57:33Z",
        "categoryId": "10",
        "tags": ["Rick Astley", "Never Gonna Give You Up"],
        "thumbnails": {
            "default": {"url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/default.jpg"},
            "medium": {"url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg"},
            "high": {"url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"},
        },
    },
    "contentDetails": {
        "duration": "PT3M33S",
    },
    "statistics": {
        "viewCount": "1500000000",
        "likeCount": "17000000",
        "commentCount": "2500000",
    },
    "status": {
        "privacyStatus": "public",
    },
}


def test_missing_api_key_raises_configuration_error():
    """Verify client rejects empty or missing API keys without making network requests."""
    client = YouTubeAPIClient(api_key=None)
    with pytest.raises(YouTubeConfigurationError):
        client.fetch_video_data("dQw4w9WgXcQ")

    client_empty = YouTubeAPIClient(api_key="   ")
    with pytest.raises(YouTubeConfigurationError):
        client_empty.fetch_video_data("dQw4w9WgXcQ")


def test_successful_api_response():
    """Verify client correctly returns video item on HTTP 200."""
    fake_key = "AIzaSyFakeKey12345"
    client = YouTubeAPIClient(api_key=fake_key, max_retries=0)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"items": [SAMPLE_SUCCESS_ITEM]}

    with patch("httpx.Client.get", return_value=mock_resp):
        item = client.fetch_video_data("dQw4w9WgXcQ")
        assert item["id"] == "dQw4w9WgXcQ"
        assert item["snippet"]["title"] == "Rick Astley - Never Gonna Give You Up"


def test_video_not_found_raises_not_found_error():
    """Verify empty items array from YouTube API raises YouTubeNotFoundError."""
    client = YouTubeAPIClient(api_key="valid_key", max_retries=0)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"items": []}

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(YouTubeNotFoundError) as exc_info:
            client.fetch_video_data("nonExist123")
        assert "nonExist123" in str(exc_info.value)


def test_private_video_in_status_raises_private_error():
    """Verify videos with privacyStatus='private' raise YouTubePrivateVideoError."""
    client = YouTubeAPIClient(api_key="valid_key", max_retries=0)

    private_item = {
        "id": "dQw4w9WgXcQ",
        "snippet": {"title": "Private Video"},
        "status": {"privacyStatus": "private"}
    }
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"items": [private_item]}

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(YouTubePrivateVideoError):
            client.fetch_video_data("dQw4w9WgXcQ")


def test_invalid_api_key_raises_authentication_error():
    """Verify HTTP 400 with keyInvalid raises YouTubeAuthenticationError."""
    secret_key = "MY_SUPER_SECRET_KEY_999"
    client = YouTubeAPIClient(api_key=secret_key, max_retries=0)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 400
    mock_resp.json.return_value = {
        "error": {
            "errors": [{"reason": "keyInvalid", "message": "API key not valid."}],
            "message": "API key not valid. Please pass a valid API key."
        }
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(YouTubeAuthenticationError) as exc_info:
            client.fetch_video_data("dQw4w9WgXcQ")
        # Ensure the secret key is never leaked in the exception text
        assert secret_key not in str(exc_info.value)


def test_quota_exceeded_raises_quota_error():
    """Verify HTTP 403 quotaExceeded raises YouTubeQuotaExceededError."""
    client = YouTubeAPIClient(api_key="valid_key", max_retries=0)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 403
    mock_resp.json.return_value = {
        "error": {
            "errors": [{"reason": "quotaExceeded", "message": "The request cannot be completed because you have exceeded your quota."}],
            "message": "Quota exceeded."
        }
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(YouTubeQuotaExceededError):
            client.fetch_video_data("dQw4w9WgXcQ")


def test_timeout_and_network_error_retries_and_raises():
    """Verify client catches timeout exceptions and raises YouTubeAPIError with 504."""
    client = YouTubeAPIClient(api_key="valid_key", max_retries=1, backoff_factor=0.01)

    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Connection timed out")):
        with pytest.raises(YouTubeAPIError) as exc_info:
            client.fetch_video_data("dQw4w9WgXcQ")
        assert exc_info.value.status_code == 504


def test_malformed_json_response_raises_api_error():
    """Verify malformed JSON responses raise YouTubeAPIError with 502 Bad Gateway."""
    client = YouTubeAPIClient(api_key="valid_key", max_retries=0)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Invalid JSON")

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(YouTubeAPIError) as exc_info:
            client.fetch_video_data("dQw4w9WgXcQ")
        assert exc_info.value.status_code == 502
