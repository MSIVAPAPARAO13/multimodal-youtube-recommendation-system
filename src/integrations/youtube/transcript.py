"""
Transcript provider abstraction and YouTube implementation.
Handles absent transcripts, disabled captions, and provider errors gracefully.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    HAS_TRANSCRIPT_API = True
except ImportError:
    YouTubeTranscriptApi = None
    TranscriptsDisabled = Exception
    NoTranscriptFound = Exception
    VideoUnavailable = Exception
    HAS_TRANSCRIPT_API = False


@dataclass
class TranscriptResult:
    """Represents the outcome of a transcript retrieval attempt."""
    text: Optional[str]
    available: bool
    language: Optional[str] = None
    error: Optional[str] = None


class TranscriptProvider(ABC):
    """Abstract interface for video transcript retrieval."""

    @abstractmethod
    def get_transcript(self, video_id: str, languages: Optional[list[str]] = None) -> TranscriptResult:
        """
        Attempt to retrieve transcript text for a video.
        Must not raise on unavailable or disabled transcripts.
        """
        pass


class YouTubeTranscriptProvider(TranscriptProvider):
    """
    Retrieves YouTube captions using youtube-transcript-api where available.
    Degrades gracefully when transcripts are disabled, missing, or blocked.
    """

    def __init__(self, default_languages: Optional[list[str]] = None):
        self.default_languages = default_languages or ["en"]

    def get_transcript(self, video_id: str, languages: Optional[list[str]] = None) -> TranscriptResult:
        langs = languages or self.default_languages

        if not HAS_TRANSCRIPT_API or YouTubeTranscriptApi is None:
            logger.info("youtube-transcript-api not available; skipping transcript for %s", video_id)
            return TranscriptResult(
                text=None,
                available=False,
                error="Transcript API library is not installed",
            )

        try:
            # First try requested languages
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            combined_text = " ".join(item.get("text", "") for item in transcript_list).strip()
            return TranscriptResult(
                text=combined_text if combined_text else None,
                available=bool(combined_text),
                language=langs[0] if langs else "en",
            )

        except TranscriptsDisabled:
            logger.info("Subtitles/transcripts are disabled for video %s", video_id)
            return TranscriptResult(
                text=None,
                available=False,
                error="Transcripts are disabled by video owner",
            )

        except NoTranscriptFound:
            logger.info("No transcript found in languages %s for video %s", langs, video_id)
            return TranscriptResult(
                text=None,
                available=False,
                error="No transcript found for requested languages",
            )

        except VideoUnavailable:
            logger.warning("Video %s is unavailable for transcript extraction", video_id)
            return TranscriptResult(
                text=None,
                available=False,
                error="Video is unavailable",
            )

        except Exception as e:
            logger.warning("Error fetching transcript for %s: %s", video_id, e)
            return TranscriptResult(
                text=None,
                available=False,
                error=f"Transcript extraction failed: {str(e)}",
            )
