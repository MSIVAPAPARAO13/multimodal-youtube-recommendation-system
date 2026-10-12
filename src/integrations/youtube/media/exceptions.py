"""
Exceptions for YouTube media acquisition, validation, and processing.
"""

from typing import Optional


class MediaError(Exception):
    """Base exception for all media acquisition and processing errors."""

    def __init__(self, message: str, video_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.video_id = video_id


class MediaDownloadError(MediaError):
    """Raised when downloading video or audio stream fails."""
    pass


class MediaValidationError(MediaError):
    """Base exception for media validation errors."""
    pass


class MediaSizeExceededError(MediaValidationError):
    """Raised when downloaded media exceeds configured size limit."""
    pass


class MediaDurationExceededError(MediaValidationError):
    """Raised when media duration exceeds configured duration limit."""
    pass


class MediaCorruptError(MediaValidationError):
    """Raised when video or audio container is corrupt or unreadable."""
    pass


class AudioExtractionError(MediaError):
    """Raised when audio waveform extraction or conversion fails."""
    pass


class ThumbnailDownloadError(MediaError):
    """Raised when fetching thumbnail image fails or fails security checks."""
    pass
