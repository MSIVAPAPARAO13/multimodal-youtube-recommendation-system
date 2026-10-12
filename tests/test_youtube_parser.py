"""
Unit tests for YouTube URL and video ID parser.
"""

import pytest
from src.integrations.youtube.parser import extract_video_id, is_valid_youtube_id
from src.integrations.youtube.exceptions import YouTubeInvalidVideoIdError


def test_valid_youtube_watch_url():
    """Verify parsing standard youtube.com/watch?v=ID URLs."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"

    url_no_www = "https://youtube.com/watch?v=dQw4w9WgXcQ"
    assert extract_video_id(url_no_www) == "dQw4w9WgXcQ"


def test_valid_youtube_watch_url_with_parameters():
    """Verify parsing watch URLs with timestamps, playlist IDs, and feature flags."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&feature=shared"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_valid_youtu_be_short_url():
    """Verify parsing youtu.be/ID short URLs."""
    url = "https://youtu.be/dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"

    url_with_query = "https://youtu.be/dQw4w9WgXcQ?si=abcdef12345"
    assert extract_video_id(url_with_query) == "dQw4w9WgXcQ"


def test_valid_youtube_shorts_url():
    """Verify parsing youtube.com/shorts/ID URLs."""
    url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_valid_youtube_embed_url():
    """Verify parsing youtube.com/embed/ID URLs."""
    url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_valid_raw_video_id():
    """Verify accepting raw 11-character video IDs."""
    video_id = "dQw4w9WgXcQ"
    assert extract_video_id(video_id) == "dQw4w9WgXcQ"
    assert is_valid_youtube_id(video_id) is True

    video_id_special = "a1B2_c3-D4e"
    assert extract_video_id(video_id_special) == "a1B2_c3-D4e"


def test_invalid_urls_rejected():
    """Verify non-YouTube URLs and malformed URLs are rejected without making API calls."""
    with pytest.raises(YouTubeInvalidVideoIdError):
        extract_video_id("https://vimeo.com/123456789")

    with pytest.raises(YouTubeInvalidVideoIdError):
        extract_video_id("https://example.com/watch?v=dQw4w9WgXcQ")

    with pytest.raises(YouTubeInvalidVideoIdError):
        extract_video_id("https://www.youtube.com/about")


def test_invalid_video_ids_rejected():
    """Verify wrong length, empty, and invalid characters are rejected."""
    # Too short (10 chars)
    with pytest.raises(YouTubeInvalidVideoIdError):
        extract_video_id("dQw4w9WgXc")

    # Too long (12 chars)
    with pytest.raises(YouTubeInvalidVideoIdError):
        extract_video_id("dQw4w9WgXcQQ")

    # Invalid characters ($ and !)
    with pytest.raises(YouTubeInvalidVideoIdError):
        extract_video_id("dQw4w9WgXc$")

    # Empty string or spaces
    with pytest.raises(YouTubeInvalidVideoIdError):
        extract_video_id("")

    with pytest.raises(YouTubeInvalidVideoIdError):
        extract_video_id("   ")
