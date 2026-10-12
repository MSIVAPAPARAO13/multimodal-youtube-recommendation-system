"""
Controlled thumbnail fetcher and validator for YouTube metadata thumbnails.
"""

import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import requests

from .exceptions import ThumbnailDownloadError

logger = logging.getLogger(__name__)

ALLOWED_THUMBNAIL_DOMAINS = {
    "ytimg.com",
    "i.ytimg.com",
    "i1.ytimg.com",
    "i2.ytimg.com",
    "i3.ytimg.com",
    "i4.ytimg.com",
    "youtube.com",
    "img.youtube.com",
    "googleusercontent.com",
    "lh3.googleusercontent.com",
    "ggpht.com",
}


class ThumbnailDownloader:
    """
    Safely validates thumbnail hostnames, verifies image content types,
    and downloads thumbnails into controlled storage.
    """

    def __init__(self, max_size_mb: int = 10, timeout: float = 10.0):
        self.max_size_mb = max_size_mb
        self.timeout = timeout

    def is_safe_url(self, url: str) -> bool:
        """Verify URL uses HTTP/HTTPS and hostname belongs to trusted YouTube/Google domains."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            hostname = parsed.hostname
            if not hostname:
                return False
            hostname = hostname.lower()
            return any(
                hostname == domain or hostname.endswith("." + domain)
                for domain in ALLOWED_THUMBNAIL_DOMAINS
            )
        except Exception:
            return False

    def download_thumbnail(self, url: str, destination_path: Path | str, video_id: Optional[str] = None) -> Path:
        """
        Download and validate thumbnail image from safe URL.
        """
        if not url:
            raise ThumbnailDownloadError("Empty thumbnail URL provided", video_id=video_id)

        if not self.is_safe_url(url):
            raise ThumbnailDownloadError(f"Untrusted or invalid thumbnail domain: '{url}'", video_id=video_id)

        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(url, timeout=self.timeout, stream=True)
            if response.status_code != 200:
                raise ThumbnailDownloadError(
                    f"Thumbnail HTTP {response.status_code} for URL: {url}",
                    video_id=video_id,
                )

            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise ThumbnailDownloadError(
                    f"Invalid thumbnail Content-Type: '{content_type}' (expected image/*)",
                    video_id=video_id,
                )

            max_bytes = self.max_size_mb * 1024 * 1024
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > max_bytes:
                    raise ThumbnailDownloadError(
                        f"Thumbnail exceeds max size of {self.max_size_mb} MB",
                        video_id=video_id,
                    )

            with open(dest, "wb") as f:
                f.write(content)

            return dest

        except requests.RequestException as e:
            raise ThumbnailDownloadError(f"Network error fetching thumbnail: {e}", video_id=video_id) from e
