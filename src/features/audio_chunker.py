"""
AudioChunker: Bounded Windowing & Chunk Aggregation for Long Audio Recordings.

Responsibilities:
- Configurable chunk duration (default 20s) and overlap (default 2s)
- Preserves temporal order
- Deterministic chunk boundaries
- Safe final partial-chunk padding with attention masks
- Bounded processing to avoid memory explosion (capped by max_chunks)
- Sequence and chunk aggregation (mean, weighted/decay, max)
"""

from dataclasses import dataclass, field
import logging
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AudioChunkResult:
    """Diagnostic and payload result from AudioChunker."""
    chunks: torch.Tensor  # Shape: (num_chunks, chunk_samples)
    attention_mask: torch.Tensor  # Shape: (num_chunks, chunk_samples)
    num_chunks: int = 0
    chunk_duration_seconds: float = 20.0
    overlap_seconds: float = 2.0
    is_truncated: bool = False
    chunk_times: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunking metadata to dict without raw tensors."""
        return {
            "num_chunks": self.num_chunks,
            "chunk_duration_seconds": self.chunk_duration_seconds,
            "overlap_seconds": self.overlap_seconds,
            "is_truncated": self.is_truncated,
            "chunk_times": [(round(s, 2), round(e, 2)) for s, e in self.chunk_times],
        }


class AudioChunker:
    """
    Splits continuous 1D audio waveforms into deterministic, overlapping chunks.
    """

    def __init__(
        self,
        chunk_duration_seconds: float = 20.0,
        overlap_seconds: float = 2.0,
        max_chunks: int = 30,
        sample_rate: int = 16000,
    ):
        self.chunk_duration_seconds = max(1.0, float(chunk_duration_seconds))
        # Ensure overlap is strictly less than duration
        self.overlap_seconds = max(0.0, min(float(overlap_seconds), self.chunk_duration_seconds - 0.5))
        self.max_chunks = max(1, int(max_chunks))
        self.sample_rate = max(1, int(sample_rate))

        self.chunk_samples = int(self.chunk_duration_seconds * self.sample_rate)
        self.hop_samples = int((self.chunk_duration_seconds - self.overlap_seconds) * self.sample_rate)
        if self.hop_samples <= 0:
            self.hop_samples = self.chunk_samples

    def chunk_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> AudioChunkResult:
        """
        Split 1D waveform into overlapping (chunk_samples) windows with binary attention masks.

        Args:
            waveform: 1D Tensor of shape (num_samples,) or (1, num_samples)
            sample_rate: Sampling rate (defaults to self.sample_rate)

        Returns:
            AudioChunkResult with chunks tensor (C, chunk_samples) and attention_mask.
        """
        sr = sample_rate or self.sample_rate
        chunk_samples = int(self.chunk_duration_seconds * sr)
        hop_samples = int((self.chunk_duration_seconds - self.overlap_seconds) * sr)
        if hop_samples <= 0:
            hop_samples = chunk_samples

        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        if waveform.dim() != 1 or waveform.numel() == 0:
            return AudioChunkResult(
                chunks=torch.zeros((0, chunk_samples), dtype=torch.float32),
                attention_mask=torch.zeros((0, chunk_samples), dtype=torch.float32),
                num_chunks=0,
                chunk_duration_seconds=self.chunk_duration_seconds,
                overlap_seconds=self.overlap_seconds,
                is_truncated=False,
                chunk_times=[],
            )

        total_samples = waveform.size(0)

        # Single chunk case
        if total_samples <= chunk_samples:
            chunk = torch.zeros(chunk_samples, dtype=waveform.dtype, device=waveform.device)
            mask = torch.zeros(chunk_samples, dtype=torch.float32, device=waveform.device)

            chunk[:total_samples] = waveform
            mask[:total_samples] = 1.0

            return AudioChunkResult(
                chunks=chunk.unsqueeze(0),
                attention_mask=mask.unsqueeze(0),
                num_chunks=1,
                chunk_duration_seconds=self.chunk_duration_seconds,
                overlap_seconds=self.overlap_seconds,
                is_truncated=False,
                chunk_times=[(0.0, float(total_samples / sr))],
            )

        # Multi-chunk slicing
        chunk_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        time_list: List[Tuple[float, float]] = []

        start = 0
        is_truncated = False

        while start < total_samples and len(chunk_list) < self.max_chunks:
            end = min(start + chunk_samples, total_samples)
            actual_len = end - start

            chunk = torch.zeros(chunk_samples, dtype=waveform.dtype, device=waveform.device)
            mask = torch.zeros(chunk_samples, dtype=torch.float32, device=waveform.device)

            chunk[:actual_len] = waveform[start:end]
            mask[:actual_len] = 1.0

            chunk_list.append(chunk)
            mask_list.append(mask)
            time_list.append((float(start / sr), float(end / sr)))

            if end == total_samples:
                break

            start += hop_samples

        if start < total_samples and len(chunk_list) >= self.max_chunks:
            is_truncated = True

        chunks_tensor = torch.stack(chunk_list, dim=0)
        masks_tensor = torch.stack(mask_list, dim=0)

        return AudioChunkResult(
            chunks=chunks_tensor,
            attention_mask=masks_tensor,
            num_chunks=len(chunk_list),
            chunk_duration_seconds=self.chunk_duration_seconds,
            overlap_seconds=self.overlap_seconds,
            is_truncated=is_truncated,
            chunk_times=time_list,
        )

    def aggregate_chunk_embeddings(
        self,
        chunk_embeddings: torch.Tensor,
        method: str = "mean",
        mode: Optional[str] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate C chunk embeddings into a single video-level audio embedding.

        Args:
            chunk_embeddings: (C, feature_dim) or (1, C, feature_dim)
            method: 'mean', 'decay', 'weighted', or 'max'
            mode: Alias for method
            weights: Optional tensor of weights (C,)

        Returns:
            aggregated_embedding: (1, feature_dim)
        """
        if chunk_embeddings.dim() == 3:
            chunk_embeddings = chunk_embeddings.squeeze(0)

        C, D = chunk_embeddings.shape
        if C == 1:
            return chunk_embeddings.view(1, D)

        pooling = (mode or method).lower().strip()

        if pooling == "mean":
            return chunk_embeddings.mean(dim=0, keepdim=True)

        elif pooling in ("decay", "weighted"):
            if weights is not None and weights.numel() == C:
                w = weights.view(C, 1).to(chunk_embeddings.device)
                w = w / w.sum()
            else:
                # Temporal decay giving slight recency weighting
                steps = torch.arange(C, dtype=chunk_embeddings.dtype, device=chunk_embeddings.device)
                w = torch.exp(0.1 * (steps - (C - 1))).view(C, 1)
                w = w / w.sum()
            return (chunk_embeddings * w).sum(dim=0, keepdim=True)

        elif pooling == "max":
            return chunk_embeddings.max(dim=0, keepdim=True)[0]

        else:
            logger.warning("Unknown chunk pooling method '%s'; defaulting to mean.", pooling)
            return chunk_embeddings.mean(dim=0, keepdim=True)
