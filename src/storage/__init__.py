"""
Storage package for multimodal video recommender.
"""

from .base import BaseStorage
from .local import LocalStorage, StorageSecurityError

__all__ = ["BaseStorage", "LocalStorage", "StorageSecurityError"]
