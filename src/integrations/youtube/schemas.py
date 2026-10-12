"""
Pydantic Schemas for YouTube Data API v3 metadata and endpoint responses.
"""

import re
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


def parse_iso8601_duration(duration_str: Optional[str]) -> Optional[int]:
    """
    Parse an ISO 8601 duration string (e.g., 'PT1H2M10S', 'PT4M13S', 'PT30S') into total seconds.
    """
    if not duration_str or not isinstance(duration_str, str):
        return None

    pattern = re.compile(
        r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
    )
    match = pattern.match(duration_str)
    if not match:
        return None

    parts = match.groupdict()
    total_seconds = (
        int(parts.get("days") or 0) * 86400
        + int(parts.get("hours") or 0) * 3600
        + int(parts.get("minutes") or 0) * 60
        + int(parts.get("seconds") or 0)
    )
    return total_seconds


class YouTubeThumbnails(BaseModel):
    """Structured thumbnail URLs for a YouTube video."""
    default: Optional[str] = Field(None, description="Default thumbnail URL (120x90)")
    medium: Optional[str] = Field(None, description="Medium thumbnail URL (320x180)")
    high: Optional[str] = Field(None, description="High thumbnail URL (480x360)")

    model_config = ConfigDict(extra="ignore")


class YouTubeStatistics(BaseModel):
    """Engagement statistics for a YouTube video."""
    views: int = Field(0, description="Total view count")
    likes: int = Field(0, description="Total like count")
    comments: int = Field(0, description="Total comment count")

    model_config = ConfigDict(extra="ignore")


class YouTubeVideoMetadata(BaseModel):
    """
    Normalized internal schema for YouTube video metadata.
    Decoupled from Google API raw structures.
    """
    video_id: str = Field(..., description="11-character YouTube video ID")
    title: str = Field(..., description="Video title")
    description: str = Field("", description="Video description")
    channel_id: Optional[str] = Field(None, description="Channel ID")
    channel_title: Optional[str] = Field(None, description="Channel title")
    published_at: Optional[str] = Field(None, description="ISO timestamp of publish date")
    category_id: Optional[str] = Field(None, description="YouTube Category ID")
    tags: list[str] = Field(default_factory=list, description="Video tags")
    thumbnail_default: Optional[str] = Field(None, description="Default thumbnail URL")
    thumbnail_medium: Optional[str] = Field(None, description="Medium thumbnail URL")
    thumbnail_high: Optional[str] = Field(None, description="High thumbnail URL")
    duration: Optional[str] = Field(None, description="ISO 8601 duration string (e.g. PT4M13S)")
    duration_seconds: Optional[int] = Field(None, description="Parsed video duration in seconds")
    view_count: int = Field(0, description="View count")
    like_count: int = Field(0, description="Like count")
    comment_count: int = Field(0, description="Comment count")
    privacy_status: Optional[str] = Field("public", description="Privacy status (public, unlisted, private)")

    model_config = ConfigDict(extra="ignore")

    @property
    def thumbnails(self) -> YouTubeThumbnails:
        """Convenience property for nested thumbnail access."""
        return YouTubeThumbnails(
            default=self.thumbnail_default,
            medium=self.thumbnail_medium,
            high=self.thumbnail_high,
        )

    @property
    def statistics(self) -> YouTubeStatistics:
        """Convenience property for nested statistics access."""
        return YouTubeStatistics(
            views=self.view_count,
            likes=self.like_count,
            comments=self.comment_count,
        )

    def to_api_response(self) -> "YouTubeVideoResponse":
        """Transform into nested public API response representation."""
        return YouTubeVideoResponse(
            video_id=self.video_id,
            title=self.title,
            description=self.description,
            channel_title=self.channel_title,
            published_at=self.published_at,
            category_id=self.category_id,
            tags=self.tags,
            thumbnail=YouTubeThumbnails(
                default=self.thumbnail_default,
                medium=self.thumbnail_medium,
                high=self.thumbnail_high
            ),
            duration=self.duration,
            statistics=YouTubeStatistics(
                views=self.view_count,
                likes=self.like_count,
                comments=self.comment_count
            )
        )


class YouTubeVideoResponse(BaseModel):
    """Public API response schema for GET /api/youtube/video/{video_id}."""
    video_id: str
    title: str
    description: str
    channel_title: Optional[str] = None
    published_at: Optional[str] = None
    category_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    thumbnail: YouTubeThumbnails
    duration: Optional[str] = None
    statistics: YouTubeStatistics

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "dQw4w9WgXcQ",
                "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
                "description": "The official video for Never Gonna Give You Up by Rick Astley",
                "channel_title": "Rick Astley",
                "published_at": "2009-10-25T06:57:33Z",
                "category_id": "10",
                "tags": ["rick astley", "never gonna give you up"],
                "thumbnail": {
                    "default": "https://i.ytimg.com/vi/dQw4w9WgXcQ/default.jpg",
                    "medium": "https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg",
                    "high": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
                },
                "duration": "PT3M33S",
                "statistics": {
                    "views": 1500000000,
                    "likes": 17000000,
                    "comments": 2500000
                }
            }
        }
    )
