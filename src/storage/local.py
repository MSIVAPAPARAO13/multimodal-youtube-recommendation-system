"""
Local filesystem storage implementation with directory traversal protection.
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from .base import BaseStorage

logger = logging.getLogger(__name__)

SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class StorageSecurityError(Exception):
    """Raised when an illegal or out-of-bounds storage path is requested."""
    pass


class LocalStorage(BaseStorage):
    """
    Stores acquired media and artifacts in a controlled local directory
    under project_root / data / raw / youtube / {video_id} /
    """

    def __init__(self, root_dir: Path | str = "data/raw/youtube"):
        self.root_path = Path(root_dir).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _validate_safe_id(self, video_id: str) -> str:
        """Ensure video_id only contains alphanumeric and safe characters."""
        if not video_id or not SAFE_ID_REGEX.match(video_id):
            raise StorageSecurityError(f"Invalid or unsafe video_id: '{video_id}'")
        return video_id

    def get_path(self, relative_path: str | Path) -> Path:
        """
        Resolve relative path and strictly verify it resides inside self.root_path.
        """
        # Ensure path does not contain traversal attempts
        target = (self.root_path / relative_path).resolve()
        try:
            target.relative_to(self.root_path)
        except ValueError:
            raise StorageSecurityError(
                f"Path traversal detected: '{relative_path}' resolves outside storage root."
            )
        return target

    def get_video_dir(self, video_id: str) -> Path:
        """
        Get or create directory for video_id strictly inside root storage.
        """
        safe_id = self._validate_safe_id(video_id)
        video_dir = self.get_path(safe_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    def save_bytes(self, relative_path: str, content: bytes) -> Path:
        """Safely write bytes to relative path."""
        target_path = self.get_path(relative_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(content)
        return target_path

    def save_json(self, relative_path: str, data: dict[str, Any]) -> Path:
        """Safely serialize dict to JSON file."""
        target_path = self.get_path(relative_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return target_path

    def exists(self, relative_path: str) -> bool:
        """Check if relative path exists."""
        target_path = self.get_path(relative_path)
        return target_path.exists()

    def delete_file(self, relative_path: str) -> bool:
        """Safely delete file if present."""
        target_path = self.get_path(relative_path)
        if target_path.exists() and target_path.is_file():
            target_path.unlink()
            return True
        return False

    def delete_video_dir(self, video_id: str) -> bool:
        """
        Remove the entire directory for a video ID.
        Used for cleanup of failed or incomplete download attempts.
        """
        safe_id = self._validate_safe_id(video_id)
        target_dir = self.get_path(safe_id)
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir, ignore_errors=True)
            return True
        return False

    def clean_incomplete_files(self, video_id: str) -> int:
        """
        Remove temporary download files (.part, .ytdl, etc.) in a video folder.
        """
        safe_id = self._validate_safe_id(video_id)
        target_dir = self.get_path(safe_id)
        if not target_dir.exists() or not target_dir.is_dir():
            return 0

        cleaned_count = 0
        for pattern in ["*.part", "*.ytdl", "*.temp", "*.tmp"]:
            for temp_file in target_dir.glob(pattern):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned_count += 1
                except Exception as e:
                    logger.warning("Could not clean temporary file %s: %s", temp_file, e)

        return cleaned_count
