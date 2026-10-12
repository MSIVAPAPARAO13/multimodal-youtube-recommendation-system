"""
ThumbnailEncoder: Extracts Visual Embeddings from YouTube Thumbnails.

Uses the stored validated thumbnail from the ingestion pipeline (thumbnail.jpg)
and applies identical ImageNet preprocessing before passing through ResNet-50.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Optional, Union
import torch
import torch.nn as nn

from src.features.visual_processor import VisualProcessor
from src.features.visual_encoder import BaseVisualEncoder, ResNet50VisualEncoder

logger = logging.getLogger(__name__)


@dataclass
class ThumbnailEncodingResult:
    """Structured result and diagnostics from ThumbnailEncoder."""
    embedding: torch.Tensor
    is_available: bool
    thumbnail_available: bool
    thumbnail_path: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self, include_tensor: bool = False) -> dict[str, Any]:
        res = {
            "is_available": self.is_available,
            "thumbnail_available": self.thumbnail_available,
            "thumbnail_path": self.thumbnail_path,
            "error_message": self.error_message,
        }
        if include_tensor:
            res["embedding"] = self.embedding
        else:
            res["embedding"] = self.embedding[0].cpu().tolist() if self.embedding.numel() > 0 else []
        return res

    def __iter__(self):
        """Enable backward-compatible unpacking: emb, ok = thumb_encoder.encode(...)"""
        yield self.embedding
        yield self.is_available

    def __getitem__(self, key: Union[str, int]) -> Any:
        if key == "embedding" or key == 0:
            return self.embedding
        if key == "is_available" or key == 1:
            return self.is_available
        if key == "thumbnail_available":
            return self.thumbnail_available
        if key == "thumbnail_path":
            return self.thumbnail_path
        if key == "error_message":
            return self.error_message
        raise KeyError(key)


class ThumbnailEncoder:
    """
    Extracts 512-D embeddings from validated local YouTube thumbnails.
    """

    def __init__(
        self,
        encoder: Optional[BaseVisualEncoder] = None,
        processor: Optional[VisualProcessor] = None,
        device: Optional[torch.device] = None,
        feature_dim: int = 512,
    ):
        self.device = device or torch.device("cpu")
        self.feature_dim = feature_dim
        self.processor = processor or VisualProcessor(target_size=(224, 224), normalize=True)
        self.encoder = encoder or ResNet50VisualEncoder(
            feature_dim=feature_dim,
            pretrained=True,
            freeze_backbone=True,
        )
        self.encoder.to(self.device)
        self.encoder.eval()

    def encode_thumbnail(
        self,
        thumbnail_input: Optional[Union[str, Path, torch.Tensor]] = None,
    ) -> ThumbnailEncodingResult:
        """
        Encode thumbnail image into a 512-dimensional embedding.

        Args:
            thumbnail_input: Path to local thumbnail file or preprocessed tensor (1, 3, H, W)

        Returns:
            ThumbnailEncodingResult with 512-D embedding and diagnostic flags.
        """
        if thumbnail_input is None:
            return ThumbnailEncodingResult(
                embedding=torch.zeros(1, self.feature_dim, device=self.device),
                is_available=False,
                thumbnail_available=False,
                thumbnail_path=None,
                error_message="No thumbnail provided",
            )

        with torch.no_grad():
            # If already a tensor
            if isinstance(thumbnail_input, torch.Tensor):
                tensor = thumbnail_input.to(self.device)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                try:
                    emb = self.encoder.encode_single_image(tensor)
                    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
                    return ThumbnailEncodingResult(
                        embedding=emb,
                        is_available=True,
                        thumbnail_available=True,
                        thumbnail_path=None,
                        error_message=None,
                    )
                except Exception as e:
                    logger.warning("Failed to encode thumbnail tensor: %s", e)
                    return ThumbnailEncodingResult(
                        embedding=torch.zeros(1, self.feature_dim, device=self.device),
                        is_available=False,
                        thumbnail_available=True,
                        thumbnail_path=None,
                        error_message=str(e),
                    )

            # Local file path
            path_str = str(thumbnail_input)
            thumb_path_obj = Path(thumbnail_input)

            if not thumb_path_obj.exists() or not thumb_path_obj.is_file():
                return ThumbnailEncodingResult(
                    embedding=torch.zeros(1, self.feature_dim, device=self.device),
                    is_available=False,
                    thumbnail_available=False,
                    thumbnail_path=path_str,
                    error_message=f"Thumbnail file not found: {path_str}",
                )

            if thumb_path_obj.stat().st_size == 0:
                return ThumbnailEncodingResult(
                    embedding=torch.zeros(1, self.feature_dim, device=self.device),
                    is_available=False,
                    thumbnail_available=False,
                    thumbnail_path=path_str,
                    error_message=f"Thumbnail file is empty (0 bytes): {path_str}",
                )

            try:
                # Standardized ImageNet preprocessing
                img_tensor = self.processor.preprocess_thumbnail(thumb_path_obj)
                img_tensor = img_tensor.to(self.device)

                emb = self.encoder.encode_single_image(img_tensor)
                emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

                return ThumbnailEncodingResult(
                    embedding=emb,
                    is_available=True,
                    thumbnail_available=True,
                    thumbnail_path=path_str,
                    error_message=None,
                )
            except Exception as e:
                logger.warning("Corrupted or unreadable thumbnail file %s: %s", path_str, e)
                return ThumbnailEncodingResult(
                    embedding=torch.zeros(1, self.feature_dim, device=self.device),
                    is_available=False,
                    thumbnail_available=True,
                    thumbnail_path=path_str,
                    error_message=f"Corrupted thumbnail: {e}",
                )

    def encode(
        self,
        thumbnail_input: Optional[Union[str, Path, torch.Tensor]] = None,
    ) -> ThumbnailEncodingResult:
        """Alias for encode_thumbnail."""
        return self.encode_thumbnail(thumbnail_input)
