"""
Unit tests for YouTubeMetadataService and in-memory caching.
"""

from unittest.mock import MagicMock
import pytest

from src.integrations.youtube.service import (
    YouTubeMetadataService,
    InMemoryMetadataCache,
)
from src.integrations.youtube.schemas import (
    YouTubeVideoMetadata,
    parse_iso8601_duration,
)
from src.integrations.youtube.exceptions import YouTubeInvalidVideoIdError

SAMPLE_RAW_ITEM = {
    "id": "dQw4w9WgXcQ",
    "snippet": {
        "title": "Rick Astley - Never Gonna Give You Up",
        "description": "The official music video",
        "channelId": "UCuAXFkgsw1L7xaCfnd5JJOw",
        "channelTitle": "Rick Astley",
        "publishedAt": "2009-10-25T06:57:33Z",
        "categoryId": "10",
        "tags": ["80s", "pop", "dance"],
        "thumbnails": {
            "default": {"url": "https://img.youtube.com/vi/dQw4w9WgXcQ/default.jpg"},
            "medium": {"url": "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg"},
            "high": {"url": "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg"},
        },
    },
    "contentDetails": {
        "duration": "PT3M33S",
    },
    "statistics": {
        "viewCount": "1450000000",
        "likeCount": "16500000",
        "commentCount": "2400000",
    },
    "status": {
        "privacyStatus": "public",
    },
}


def test_parse_iso8601_duration():
    """Verify ISO 8601 duration parser for various formats."""
    assert parse_iso8601_duration("PT3M33S") == 213
    assert parse_iso8601_duration("PT1H2M10S") == 3730
    assert parse_iso8601_duration("PT45S") == 45
    assert parse_iso8601_duration("P1DT2H") == 93600
    assert parse_iso8601_duration(None) is None
    assert parse_iso8601_duration("invalid") is None


def test_metadata_normalization():
    """Verify service transforms raw YouTube API dictionary into YouTubeVideoMetadata."""
    mock_client = MagicMock()
    mock_client.fetch_video_data.return_value = SAMPLE_RAW_ITEM

    cache = InMemoryMetadataCache()
    service = YouTubeMetadataService(client=mock_client, cache=cache)

    metadata = service.get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert isinstance(metadata, YouTubeVideoMetadata)
    assert metadata.video_id == "dQw4w9WgXcQ"
    assert metadata.title == "Rick Astley - Never Gonna Give You Up"
    assert metadata.channel_title == "Rick Astley"
    assert metadata.tags == ["80s", "pop", "dance"]
    assert metadata.duration == "PT3M33S"
    assert metadata.duration_seconds == 213
    assert metadata.view_count == 1450000000
    assert metadata.like_count == 16500000
    assert metadata.comment_count == 2400000
    assert metadata.thumbnail_high == "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg"


def test_caching_behavior():
    """Verify repeated requests for the same video hit cache and do not invoke client."""
    mock_client = MagicMock()
    mock_client.fetch_video_data.return_value = SAMPLE_RAW_ITEM

    cache = InMemoryMetadataCache()
    service = YouTubeMetadataService(client=mock_client, cache=cache)

    # First call: hits client
    res1 = service.get_video_metadata("dQw4w9WgXcQ")
    assert mock_client.fetch_video_data.call_count == 1

    # Second call with URL variant: hits cache, client call count stays 1
    res2 = service.get_video_metadata("https://youtu.be/dQw4w9WgXcQ")
    assert mock_client.fetch_video_data.call_count == 1
    assert res1.video_id == res2.video_id
    assert res1.title == res2.title


def test_cache_eviction_on_capacity():
    """Verify in-memory cache evicts oldest items when max_size is reached."""
    cache = InMemoryMetadataCache(max_size=2)
    meta1 = YouTubeVideoMetadata(video_id="id111111111", title="Video 1")
    meta2 = YouTubeVideoMetadata(video_id="id222222222", title="Video 2")
    meta3 = YouTubeVideoMetadata(video_id="id333333333", title="Video 3")

    cache.set("id111111111", meta1)
    cache.set("id222222222", meta2)
    assert len(cache) == 2

    # Inserting 3rd item should evict oldest (id111111111)
    cache.set("id333333333", meta3)
    assert len(cache) == 2
    assert cache.get("id111111111") is None
    assert cache.get("id222222222") is not None
    assert cache.get("id333333333") is not None


def test_invalid_input_rejected_before_client_call():
    """Verify invalid input raises error immediately without hitting client."""
    mock_client = MagicMock()
    service = YouTubeMetadataService(client=mock_client)

    with pytest.raises(YouTubeInvalidVideoIdError):
        service.get_video_metadata("not_a_valid_id")

    assert mock_client.fetch_video_data.call_count == 0
