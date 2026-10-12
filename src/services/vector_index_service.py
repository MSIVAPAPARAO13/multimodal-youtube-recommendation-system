"""
Vector Index Service for Candidate Retrieval.

Experiment / Phase 9 Mode: TEXT_COENGAGEMENT_V2
Provides both exact cosine search (correctness baseline) and pgvector-backed
retrieval (with automatic exact fallback), top-K ranking, and latency measurement.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sqlalchemy.orm import Session

from src.db.models import HAS_PGVECTOR, Video
from src.db.repository import VideoRepository

logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 512


class VectorIndexService:
    """
    Manages vector candidate retrieval, exact vs approximate search,
    embedding validation, and self-match exclusion.
    """

    def __init__(
        self,
        default_top_k: int = 10,
        max_top_k: int = 50,
        backend: str = "auto",
        ef_search: Optional[int] = None,
    ):
        self.default_top_k = default_top_k
        self.max_top_k = max_top_k
        self.backend = backend.lower()  # "auto", "exact", "ann"
        self.ef_search = ef_search

    @property
    def is_pgvector_available(self) -> bool:
        """Check whether pgvector extension is available in the current environment."""
        return HAS_PGVECTOR and (self.backend in ("auto", "ann", "pgvector"))

    def validate_embedding(
        self,
        embedding: Union[List[float], torch.Tensor, np.ndarray],
    ) -> List[float]:
        """
        Validate dimension, finiteness, and normalize to unit L2 sphere.
        Rejects NaN, Inf, and incorrect dimensionality.
        """
        if isinstance(embedding, torch.Tensor):
            arr = embedding.detach().cpu().squeeze().numpy().astype(np.float32)
        elif isinstance(embedding, list):
            arr = np.array(embedding, dtype=np.float32)
        elif isinstance(embedding, np.ndarray):
            arr = embedding.squeeze().astype(np.float32)
        else:
            raise TypeError(f"Unsupported embedding type: {type(embedding)}")

        if arr.ndim != 1:
            raise ValueError(f"Embedding must be a 1D vector, got shape {arr.shape}")

        if arr.shape[0] != EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EXPECTED_EMBEDDING_DIM}, got {arr.shape[0]}"
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError("Embedding contains non-finite values (NaN or Inf)")

        norm = float(np.linalg.norm(arr))
        if norm <= 1e-8:
            raise ValueError("Embedding vector norm is effectively zero; cannot normalize")

        # Normalize to unit sphere
        normalized = arr / norm
        return normalized.tolist()

    def search_exact_in_memory(
        self,
        query_embedding: List[float],
        candidate_embeddings: Dict[str, List[float]],
        top_k: int = 10,
        exclude_video_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Deterministic exact nearest-neighbor search using cosine dot-product in NumPy.

        Args:
            query_embedding: Validated 512-D normalized query vector
            candidate_embeddings: Map of candidate_id -> 512-D vector
            top_k: Number of candidates to return
            exclude_video_ids: IDs to exclude (e.g., query item or already-seen items)

        Returns:
            List of (video_id, cosine_similarity_score) sorted descending
        """
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0 or np.isnan(q_norm):
            return []

        exclude = exclude_video_ids or set()
        scores: List[Tuple[str, float]] = []

        for vid, c_vec in candidate_embeddings.items():
            if vid in exclude or c_vec is None:
                continue
            c = np.array(c_vec, dtype=np.float32)
            c_norm = np.linalg.norm(c)
            if c_norm == 0 or np.isnan(c_norm):
                continue
            sim = float(np.dot(q, c) / (q_norm * c_norm))
            scores.append((vid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        bounded_k = min(max(1, top_k), self.max_top_k)
        return scores[:bounded_k]

    def search_database(
        self,
        session: Session,
        query_embedding: List[float],
        top_k: int = 10,
        exclude_video_ids: Optional[List[str]] = None,
        category_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        backend: Optional[str] = None,
        ef_search: Optional[int] = None,
    ) -> Tuple[List[Tuple[Video, float]], str, float]:
        """
        Retrieve nearest candidates from the database.
        Uses pgvector HNSW when available; automatically falls back to exact search otherwise.

        Returns:
            (results, backend_used, latency_ms)
        """
        validated_emb = self.validate_embedding(query_embedding)
        bounded_k = min(max(1, top_k), self.max_top_k)
        repo = VideoRepository(session)

        active_backend = (backend or self.backend).lower()
        if active_backend == "pgvector":
            active_backend = "ann"
        elif active_backend == "fallback":
            active_backend = "exact"
        active_ef_search = ef_search or self.ef_search

        t0 = time.perf_counter()
        results, backend_used = repo.find_similar_with_filters(
            embedding=validated_emb,
            limit=bounded_k,
            exclude_video_ids=exclude_video_ids,
            category_id=category_id,
            channel_id=channel_id,
            backend=active_backend,
            ef_search=active_ef_search,
            return_backend=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return results, backend_used, latency_ms

    def compare_exact_vs_approximate(
        self,
        exact_results: List[Tuple[str, float]],
        approx_results: List[Tuple[str, float]],
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate Recall@K of approximate retrieval relative to exact ground truth.
        """
        exact_top_k = [vid for vid, _ in exact_results[:k]]
        approx_top_k = [vid for vid, _ in approx_results[:k]]

        if not exact_top_k:
            return {"k": k, "recall_at_k": 1.0, "exact_count": 0, "approx_count": 0}

        overlap = len(set(exact_top_k).intersection(set(approx_top_k)))
        recall = overlap / float(len(exact_top_k))

        return {
            "k": k,
            "recall_at_k": round(recall, 4),
            "exact_count": len(exact_top_k),
            "approx_count": len(approx_top_k),
            "overlap_count": overlap,
        }
