"""
YouTube Integration Package for Multimodal Video Recommendation System.
"""

from src.integrations.youtube.client import YouTubeAPIClient
from src.integrations.youtube.parser import extract_video_id, is_valid_youtube_id
from src.integrations.youtube.schemas import (
    YouTubeVideoMetadata,
    YouTubeVideoResponse,
    YouTubeThumbnails,
    YouTubeStatistics,
    parse_iso8601_duration,
)
from src.integrations.youtube.service import (
    YouTubeMetadataService,
    InMemoryMetadataCache,
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

from src.integrations.youtube.transcript import (
    TranscriptProvider,
    YouTubeTranscriptProvider,
    TranscriptResult,
)
from src.integrations.youtube.media import (
    MediaError,
    MediaDownloadError,
    MediaValidationError,
    MediaSizeExceededError,
    MediaDurationExceededError,
    MediaCorruptError,
    AudioExtractionError,
    ThumbnailDownloadError,
    MediaValidator,
    MediaDownloader,
    ThumbnailDownloader,
    AudioExtractor,
)

__all__ = [
    "YouTubeAPIClient",
    "YouTubeMetadataService",
    "InMemoryMetadataCache",
    "extract_video_id",
    "is_valid_youtube_id",
    "YouTubeVideoMetadata",
    "YouTubeVideoResponse",
    "YouTubeThumbnails",
    "YouTubeStatistics",
    "parse_iso8601_duration",
    "YouTubeAPIError",
    "YouTubeAuthenticationError",
    "YouTubeConfigurationError",
    "YouTubeInvalidVideoIdError",
    "YouTubeNotFoundError",
    "YouTubePrivateVideoError",
    "YouTubeQuotaExceededError",
    "TranscriptProvider",
    "YouTubeTranscriptProvider",
    "TranscriptResult",
    "MediaError",
    "MediaDownloadError",
    "MediaValidationError",
    "MediaSizeExceededError",
    "MediaDurationExceededError",
    "MediaCorruptError",
    "AudioExtractionError",
    "ThumbnailDownloadError",
    "MediaValidator",
    "MediaDownloader",
    "ThumbnailDownloader",
    "AudioExtractor",
]
