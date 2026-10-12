"""
Controlled YouTube media downloader using yt-dlp.
Ensures low-resolution downloads (<=360p), strict URL/ID validation,
and bounded resource consumption.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    import yt_dlp
    HAS_YT_DLP = True
except ImportError:
    yt_dlp = None
    HAS_YT_DLP = False

from ..parser import extract_video_id
from ..exceptions import YouTubeInvalidVideoIdError
from .exceptions import MediaDownloadError

logger = logging.getLogger(__name__)


class MediaDownloader:
    """
    Controlled media acquisition wrapper for YouTube.
    Downloads the smallest viable media format suitable for multimodal processing.
    """

    def __init__(
        self,
        timeout: float = 60.0,
        max_size_mb: int = 100,
        max_duration_seconds: int = 600,
    ):
        self.timeout = timeout
        self.max_size_mb = max_size_mb
        self.max_duration_seconds = max_duration_seconds

    def download_video(self, video_input: str, destination_dir: Path | str) -> Path:
        """
        Download YouTube video at low resolution (<=360p) into destination_dir.

        Args:
            video_input: Canonical video ID or valid YouTube URL
            destination_dir: Directory where video.mp4 will be saved

        Returns:
            Path to downloaded mp4 file
        """
        if not HAS_YT_DLP or yt_dlp is None:
            raise MediaDownloadError("yt-dlp is not installed or available in the environment.")

        # Strict validation: video_input must resolve to canonical 11-char YouTube ID
        try:
            canonical_id = extract_video_id(video_input)
        except YouTubeInvalidVideoIdError as e:
            raise MediaDownloadError(f"Rejected invalid YouTube video input: {e.message}") from e

        dest_dir = Path(destination_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        target_video_path = dest_dir / "video.mp4"

        # Canonical YouTube URL only
        canonical_url = f"https://www.youtube.com/watch?v={canonical_id}"

        ydl_opts = {
            # Format strategy: <=360p to minimize network and disk overhead while preserving 224x224 frame quality
            "format": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360]/bestvideo[height<=360]+bestaudio/worst",
            "outtmpl": str(target_video_path),
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": self.timeout,
            "max_filesize": self.max_size_mb * 1024 * 1024,
            "overwrites": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(canonical_url, download=True)
                if not info:
                    raise MediaDownloadError("Failed to retrieve media stream information", video_id=canonical_id)

                duration = info.get("duration")
                if duration and duration > self.max_duration_seconds:
                    # Clean up file if downloaded
                    if target_video_path.exists():
                        target_video_path.unlink()
                    raise MediaDownloadError(
                        f"Video duration ({duration}s) exceeds maximum configured limit of {self.max_duration_seconds}s",
                        video_id=canonical_id,
                    )

            if not target_video_path.exists():
                # yt-dlp might have saved with slightly different extension if remuxing
                candidates = list(dest_dir.glob("video.*"))
                if candidates:
                    target_video_path = candidates[0]
                else:
                    raise MediaDownloadError("Expected video file not found after download completion", video_id=canonical_id)

            return target_video_path

        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            logger.warning("yt-dlp download failed for %s: %s", canonical_id, error_msg)
            raise MediaDownloadError(f"YouTube media download failed: {error_msg}", video_id=canonical_id) from e
        except Exception as e:
            if isinstance(e, MediaDownloadError):
                raise
            logger.error("Unexpected error during media download for %s: %s", canonical_id, e)
            raise MediaDownloadError(f"Unexpected media download failure: {e}", video_id=canonical_id) from e
