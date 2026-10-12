"""
YouTube URL and Video ID parser.
Extracts and validates YouTube 11-character video IDs without making API requests.
"""

import re
from urllib.parse import urlparse, parse_qs
from typing import Optional

from src.integrations.youtube.exceptions import YouTubeInvalidVideoIdError

# Standard YouTube video ID format: 11 base64-like characters [a-zA-Z0-9_-]
YOUTUBE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11}$")

# Recognized YouTube hostnames
VALID_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def is_valid_youtube_id(video_id: str) -> bool:
    """Check whether a string matches the standard 11-char YouTube ID pattern."""
    if not isinstance(video_id, str):
        return False
    return bool(YOUTUBE_ID_REGEX.match(video_id.strip()))


def extract_video_id(url_or_id: str) -> str:
    """
    Extract and validate an 11-character YouTube video ID from a URL or raw ID string.

    Supported Formats:
        - Raw ID: 'dQw4w9WgXcQ'
        - Standard watch: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        - Short domain: 'https://youtu.be/dQw4w9WgXcQ'
        - Shorts: 'https://www.youtube.com/shorts/dQw4w9WgXcQ'
        - Embed: 'https://www.youtube.com/embed/dQw4w9WgXcQ'
        - URLs with query params/timestamps: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s'

    Args:
        url_or_id: Raw string (URL or video ID)

    Returns:
        Canonical 11-character video ID string.

    Raises:
        YouTubeInvalidVideoIdError: If input cannot be parsed into a valid YouTube video ID.
    """
    if not url_or_id or not isinstance(url_or_id, str):
        raise YouTubeInvalidVideoIdError(str(url_or_id), "Video ID or URL must be a non-empty string.")

    cleaned = url_or_id.strip()

    # 1. Check if input is already a raw 11-char YouTube ID
    if is_valid_youtube_id(cleaned):
        return cleaned

    # 2. Check if input looks like a URL
    if "://" not in cleaned and not any(cleaned.startswith(h) for h in ("youtube.com", "www.youtube.com", "youtu.be")):
        # If it's not a URL and failed the 11-char regex, it's invalid
        raise YouTubeInvalidVideoIdError(cleaned, f"Invalid YouTube video ID format: '{cleaned}'")

    # Add scheme if missing
    if "://" not in cleaned:
        cleaned = "https://" + cleaned

    try:
        parsed = urlparse(cleaned)
    except Exception as e:
        raise YouTubeInvalidVideoIdError(cleaned, f"Malformed URL: {e}")

    hostname = (parsed.hostname or "").lower()
    if not any(hostname == h or hostname.endswith("." + h) for h in ("youtube.com", "youtu.be")):
        raise YouTubeInvalidVideoIdError(cleaned, f"Not a recognized YouTube URL: '{cleaned}'")

    candidate_id: Optional[str] = None

    # Handle youtu.be/VIDEO_ID
    if "youtu.be" in hostname:
        path_parts = [p for p in parsed.path.split("/") if p]
        if path_parts:
            candidate_id = path_parts[0]

    # Handle youtube.com
    elif "youtube.com" in hostname:
        # 1. /watch?v=VIDEO_ID
        if parsed.path == "/watch" or parsed.path == "/watch/":
            query_params = parse_qs(parsed.query)
            v_values = query_params.get("v")
            if v_values and len(v_values) > 0:
                candidate_id = v_values[0]

        # 2. /shorts/VIDEO_ID
        elif parsed.path.startswith("/shorts/"):
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] == "shorts":
                candidate_id = parts[1]

        # 3. /embed/VIDEO_ID
        elif parsed.path.startswith("/embed/"):
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] == "embed":
                candidate_id = parts[1]

        # 4. /v/VIDEO_ID
        elif parsed.path.startswith("/v/"):
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] == "v":
                candidate_id = parts[1]

    if candidate_id and is_valid_youtube_id(candidate_id):
        return candidate_id

    raise YouTubeInvalidVideoIdError(
        url_or_id,
        f"Could not extract a valid 11-character video ID from: '{url_or_id}'"
    )
