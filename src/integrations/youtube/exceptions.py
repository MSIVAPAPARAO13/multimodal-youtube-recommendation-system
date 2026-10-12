"""
Exceptions for YouTube Data API v3 integration.
"""

from typing import Optional


class YouTubeAPIError(Exception):
    """Base exception for YouTube API operations."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class YouTubeConfigurationError(YouTubeAPIError):
    """Raised when YouTube API key or settings are missing or invalid."""
    def __init__(self, message: str = "YouTube API is not properly configured. Missing or invalid API key."):
        super().__init__(message, status_code=500)


class YouTubeInvalidVideoIdError(YouTubeAPIError):
    """Raised when a provided URL or ID is not a valid YouTube video identifier."""
    def __init__(self, video_id: str, message: Optional[str] = None):
        msg = message or f"Invalid YouTube video identifier: '{video_id}'"
        super().__init__(msg, status_code=400)
        self.video_id = video_id


class YouTubeNotFoundError(YouTubeAPIError):
    """Raised when a video is not found on YouTube."""
    def __init__(self, video_id: str, message: Optional[str] = None):
        msg = message or f"YouTube video not found: '{video_id}'"
        super().__init__(msg, status_code=404)
        self.video_id = video_id


class YouTubePrivateVideoError(YouTubeAPIError):
    """Raised when a video is private, restricted, or unavailable."""
    def __init__(self, video_id: str, message: Optional[str] = None):
        msg = message or f"YouTube video '{video_id}' is private, deleted, or unavailable."
        super().__init__(msg, status_code=403)
        self.video_id = video_id


class YouTubeQuotaExceededError(YouTubeAPIError):
    """Raised when the YouTube Data API quota has been exceeded."""
    def __init__(self, message: str = "YouTube API quota exceeded. Please try again later."):
        super().__init__(message, status_code=429)


class YouTubeAuthenticationError(YouTubeAPIError):
    """Raised when the YouTube API key is rejected or unauthorized."""
    def __init__(self, message: str = "YouTube API authentication failed. Invalid API key."):
        super().__init__(message, status_code=401)
