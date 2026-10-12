"""
Media acquisition, validation, thumbnail, and audio extraction package.
"""

from .exceptions import (
    MediaError,
    MediaDownloadError,
    MediaValidationError,
    MediaSizeExceededError,
    MediaDurationExceededError,
    MediaCorruptError,
    AudioExtractionError,
    ThumbnailDownloadError,
)
from .validator import MediaValidator
from .downloader import MediaDownloader
from .thumbnails import ThumbnailDownloader
from .audio import AudioExtractor

__all__ = [
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
