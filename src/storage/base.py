"""
Storage abstraction base interface for media assets and metadata.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class BaseStorage(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def get_path(self, relative_path: str) -> Path:
        """Resolve and validate a relative path within the controlled storage root."""
        pass

    @abstractmethod
    def get_video_dir(self, video_id: str) -> Path:
        """Get or create the storage directory for a specific video ID."""
        pass

    @abstractmethod
    def save_bytes(self, relative_path: str, content: bytes) -> Path:
        """Save raw bytes to a target relative path."""
        pass

    @abstractmethod
    def save_json(self, relative_path: str, data: dict[str, Any]) -> Path:
        """Save a dictionary as JSON to a target relative path."""
        pass

    @abstractmethod
    def exists(self, relative_path: str) -> bool:
        """Check if a relative path exists."""
        pass

    @abstractmethod
    def delete_file(self, relative_path: str) -> bool:
        """Delete a file if it exists."""
        pass

    @abstractmethod
    def delete_video_dir(self, video_id: str) -> bool:
        """Delete an entire video folder and its contents."""
        pass
