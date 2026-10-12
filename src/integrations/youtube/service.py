"""
YouTube Metadata Service Layer.
Coordinates URL parsing, API client calls, caching, and schema normalization.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Optional

from src.db.config import Settings, get_settings
from src.integrations.youtube.client import YouTubeAPIClient
from src.integrations.youtube.parser import extract_video_id
from src.integrations.youtube.schemas import (
    YouTubeVideoMetadata,
    parse_iso8601_duration,
)

logger = logging.getLogger(__name__)


class InMemoryMetadataCache:
    """
    Thread-safe application-level in-memory cache with TTL.
    Prevents duplicate calls to YouTube Data API for frequently accessed videos.
    Note: Persistent database caching will be added in Phase 4.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self._cache: dict[str, tuple[YouTubeVideoMetadata, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size = max_size

    def get(self, video_id: str) -> Optional[YouTubeVideoMetadata]:
        """Retrieve cached metadata if present and not expired."""
        if video_id in self._cache:
            metadata, timestamp = self._cache[video_id]
            now = datetime.now(timezone.utc)
            if now - timestamp < self.ttl:
                return metadata
            # Expired
            del self._cache[video_id]
        return None

    def set(self, video_id: str, metadata: YouTubeVideoMetadata) -> None:
        """Store metadata in cache, evicting oldest item if capacity is reached."""
        if len(self._cache) >= self.max_size:
            # Evict first / oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[video_id] = (metadata, datetime.now(timezone.utc))

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class YouTubeMetadataService:
    """
    High-level service for fetching normalized YouTube video metadata.

    Workflow:
        1. Extract & validate video ID from URL or raw ID string.
        2. Check application-level cache.
        3. If cache miss, fetch data via YouTubeAPIClient.
        4. Normalize API response into YouTubeVideoMetadata.
        5. Cache result and return.
    """

    def __init__(
        self,
        client: Optional[YouTubeAPIClient] = None,
        cache: Optional[InMemoryMetadataCache] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.client = client or YouTubeAPIClient(
            api_key=self.settings.YOUTUBE_API_KEY,
            base_url=self.settings.YOUTUBE_API_BASE_URL,
            timeout=self.settings.YOUTUBE_API_TIMEOUT,
        )
        self.cache = cache or InMemoryMetadataCache(ttl_seconds=3600)

    def get_video_metadata(self, url_or_id: str) -> YouTubeVideoMetadata:
        """
        Fetch and normalize YouTube metadata for a video URL or ID.

        Args:
            url_or_id: YouTube URL (watch, embed, shorts, youtu.be) or 11-char ID.

        Returns:
            Normalized YouTubeVideoMetadata instance.

        Raises:
            YouTubeInvalidVideoIdError: If URL/ID cannot be parsed.
            YouTubeConfigurationError: If API key is missing.
            YouTubeNotFoundError: If video does not exist.
            YouTubePrivateVideoError: If video is private.
            YouTubeQuotaExceededError: If daily quota is exhausted.
            YouTubeAuthenticationError: If API key is invalid.
            YouTubeAPIError: For general API errors.
        """
        # 1. Parse and validate ID without making network calls
        video_id = extract_video_id(url_or_id)

        # 2. Check Cache
        cached_result = self.cache.get(video_id)
        if cached_result is not None:
            logger.debug("Cache hit for YouTube video '%s'", video_id)
            return cached_result

        # 3. Call API Client
        raw_item = self.client.fetch_video_data(video_id)

        # 4. Transform into normalized domain schema
        metadata = self._transform_item(video_id, raw_item)

        # 5. Store in Cache
        self.cache.set(video_id, metadata)
        return metadata

    def _transform_item(self, video_id: str, item: dict[str, Any]) -> YouTubeVideoMetadata:
        """Transform raw YouTube Data API v3 item dictionary into YouTubeVideoMetadata."""
        snippet = item.get("snippet", {})
        content_details = item.get("contentDetails", {})
        statistics = item.get("statistics", {})
        status = item.get("status", {})

        thumbnails = snippet.get("thumbnails", {})
        thumb_default = thumbnails.get("default", {}).get("url")
        thumb_medium = thumbnails.get("medium", {}).get("url")
        thumb_high = thumbnails.get("high", {}).get("url")

        raw_duration = content_details.get("duration")
        duration_sec = parse_iso8601_duration(raw_duration)

        return YouTubeVideoMetadata(
            video_id=video_id,
            title=snippet.get("title", ""),
            description=snippet.get("description", ""),
            channel_id=snippet.get("channelId"),
            channel_title=snippet.get("channelTitle"),
            published_at=snippet.get("publishedAt"),
            category_id=snippet.get("categoryId"),
            tags=snippet.get("tags", []) or [],
            thumbnail_default=thumb_default,
            thumbnail_medium=thumb_medium,
            thumbnail_high=thumb_high,
            duration=raw_duration,
            duration_seconds=duration_sec,
            view_count=int(statistics.get("viewCount", 0) or 0),
            like_count=int(statistics.get("likeCount", 0) or 0),
            comment_count=int(statistics.get("commentCount", 0) or 0),
            privacy_status=status.get("privacyStatus", "public"),
        )
