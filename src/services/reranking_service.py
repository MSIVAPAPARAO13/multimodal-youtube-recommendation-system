"""
Candidate Re-Ranking Service.

Phase 11B: Multimodal Candidate Re-Ranking Engine
Experiment ID: EXP_MICROLENS_MULTIMODAL_RERANK_V1

Provides a transparent, modular multi-signal re-ranking pipeline operating on
bounded candidate sets retrieved by Stage 1 HNSW ANN retrieval.
Supports active-weight renormalization across available modalities (text, visual,
audio, metadata), deterministic tie-breaking, and safe fallback.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from src.db.config import get_settings, Settings
from src.db.models import Video

logger = logging.getLogger(__name__)

RERANKING_EXPERIMENT_ID = "EXP_MICROLENS_MULTIMODAL_RERANK_V1"


class ReRankingWeights(BaseModel):
    """
    Configuration weights for candidate re-ranking signals.
    All weights must be finite and non-negative.
    """
    w_text: float = Field(0.70, description="Weight for text semantic relevance")
    w_visual: float = Field(0.15, description="Weight for visual / thumbnail relevance")
    w_audio: float = Field(0.05, description="Weight for audio relevance")
    w_metadata: float = Field(0.10, description="Weight for category / channel metadata match")

    @field_validator("w_text", "w_visual", "w_audio", "w_metadata")
    @classmethod
    def validate_weight_value(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Weight value must be finite, got {v}")
        if v < 0.0:
            raise ValueError(f"Weight value must be non-negative, got {v}")
        return float(v)

    def total_weight(self) -> float:
        return self.w_text + self.w_visual + self.w_audio + self.w_metadata

    def validate_positive_total(self) -> None:
        if self.total_weight() <= 0.0:
            raise ValueError("Total sum of re-ranking weights must be strictly positive.")


class CandidateReRankingService:
    """
    Stage 2 Re-Ranking Service.
    Re-scores a bounded set of candidates from Stage 1 retrieval using
    verified multimodal signals and metadata.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        default_weights: Optional[ReRankingWeights] = None,
        expected_dimension: int = 512,
    ):
        self.settings = settings or get_settings()
        self.expected_dimension = expected_dimension
        if default_weights is not None:
            default_weights.validate_positive_total()
            self.default_weights = default_weights
        else:
            self.default_weights = ReRankingWeights(
                w_text=self.settings.RERANKING_WEIGHT_TEXT,
                w_visual=self.settings.RERANKING_WEIGHT_VISUAL,
                w_audio=self.settings.RERANKING_WEIGHT_AUDIO,
                w_metadata=self.settings.RERANKING_WEIGHT_METADATA,
            )

    def validate_vector(
        self,
        vector: Union[List[float], np.ndarray],
        name: str = "vector",
    ) -> np.ndarray:
        """
        Validate vector dimension, finiteness, and non-zero norm.
        Returns L2-normalized float32 numpy array.
        """
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.flatten()

        if len(arr) != self.expected_dimension:
            raise ValueError(
                f"{name} dimension mismatch: expected {self.expected_dimension}, got {len(arr)}"
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values")

        norm = float(np.linalg.norm(arr))
        if norm < 1e-7:
            raise ValueError(f"{name} has zero or negligible norm ({norm})")

        return arr / norm

    def compute_cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two unit vectors, bounded in [-1.0, 1.0]."""
        dot = float(np.dot(vec_a, vec_b))
        return max(-1.0, min(1.0, dot))

    def score_candidate(
        self,
        candidate_video: Video,
        query_text_norm: np.ndarray,
        query_visual_norm: Optional[np.ndarray] = None,
        query_audio_norm: Optional[np.ndarray] = None,
        candidate_visual_vector: Optional[Union[List[float], np.ndarray]] = None,
        candidate_audio_vector: Optional[Union[List[float], np.ndarray]] = None,
        target_category_id: Optional[str] = None,
        target_channel_id: Optional[str] = None,
        weights: Optional[ReRankingWeights] = None,
        stage1_score: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate composite multi-signal score for a single candidate.

        Signals:
        - text_score: cosine similarity in [-1.0, 1.0]
        - visual_score: cosine similarity in [-1.0, 1.0] (active only if both query and cand exist)
        - audio_score: cosine similarity in [-1.0, 1.0] (active only if both query and cand exist)
        - metadata_score: category match (0.6) + channel match (0.4) in [0.0, 1.0]

        Missing modalities are excluded from active weight sum, and remaining
        weights are dynamically renormalized.
        """
        active_w = weights or self.default_weights
        active_w.validate_positive_total()

        signal_scores: Dict[str, float] = {}
        active_weights: Dict[str, float] = {}

        # 1. Text signal
        cand_text_raw = candidate_video.embedding
        if cand_text_raw is not None:
            try:
                cand_text_norm = self.validate_vector(cand_text_raw, "candidate_text")
                sim_text = self.compute_cosine_similarity(query_text_norm, cand_text_norm)
                signal_scores["text"] = sim_text
                active_weights["text"] = active_w.w_text
            except Exception as e:
                logger.debug("Candidate %s text validation failed: %s", candidate_video.video_id, e)

        # 2. Visual signal
        if query_visual_norm is not None and candidate_visual_vector is not None:
            try:
                cand_vis_norm = self.validate_vector(candidate_visual_vector, "candidate_visual")
                sim_vis = self.compute_cosine_similarity(query_visual_norm, cand_vis_norm)
                signal_scores["visual"] = sim_vis
                active_weights["visual"] = active_w.w_visual
            except Exception as e:
                logger.debug("Candidate %s visual validation failed: %s", candidate_video.video_id, e)

        # 3. Audio signal
        if query_audio_norm is not None and candidate_audio_vector is not None:
            try:
                cand_aud_norm = self.validate_vector(candidate_audio_vector, "candidate_audio")
                sim_aud = self.compute_cosine_similarity(query_audio_norm, cand_aud_norm)
                signal_scores["audio"] = sim_aud
                active_weights["audio"] = active_w.w_audio
            except Exception as e:
                logger.debug("Candidate %s audio validation failed: %s", candidate_video.video_id, e)

        # 4. Metadata signal
        meta_active = False
        meta_score = 0.0
        if target_category_id or target_channel_id:
            cat_match = 0.0
            chan_match = 0.0
            if target_category_id and candidate_video.category_id:
                meta_active = True
                cat_match = 1.0 if str(candidate_video.category_id) == str(target_category_id) else 0.0
            if target_channel_id and candidate_video.channel_id:
                meta_active = True
                chan_match = 1.0 if str(candidate_video.channel_id) == str(target_channel_id) else 0.0

            if meta_active:
                meta_score = (0.6 * cat_match) + (0.4 * chan_match)
                signal_scores["metadata"] = meta_score
                active_weights["metadata"] = active_w.w_metadata

        # 5. Composite score calculation with dynamic weight renormalization
        total_active_weight = sum(active_weights.values())
        if total_active_weight > 1e-7:
            composite_score = sum(
                (weight / total_active_weight) * signal_scores[sig]
                for sig, weight in active_weights.items()
            )
        else:
            # All signals missing; safely fallback to stage 1 score or 0.0
            composite_score = stage1_score if stage1_score is not None else 0.0

        telemetry = {
            "signal_scores": signal_scores,
            "active_weights": active_weights,
            "renormalized_weight_sum": total_active_weight,
            "modalities_present": list(active_weights.keys()),
        }

        return float(composite_score), telemetry

    def rerank_candidates(
        self,
        candidates: List[Tuple[Video, float]],
        query_text_embedding: Union[List[float], np.ndarray],
        query_visual_embedding: Optional[Union[List[float], np.ndarray]] = None,
        query_audio_embedding: Optional[Union[List[float], np.ndarray]] = None,
        candidate_visual_embeddings: Optional[Dict[str, Union[List[float], np.ndarray]]] = None,
        candidate_audio_embeddings: Optional[Dict[str, Union[List[float], np.ndarray]]] = None,
        target_category_id: Optional[str] = None,
        target_channel_id: Optional[str] = None,
        weights: Optional[ReRankingWeights] = None,
        top_k: int = 10,
        ranking_mode: str = "rerank",
    ) -> List[Tuple[Video, float]]:
        """
        Re-rank a retrieved candidate list.

        Args:
            candidates: List of (Video, stage1_score) tuples from Stage 1 retrieval
            query_text_embedding: Required 512-D query representation
            query_visual_embedding: Optional 512-D visual query vector
            query_audio_embedding: Optional 512-D audio query vector
            candidate_visual_embeddings: Optional mapping of video_id -> 512-D visual vector
            candidate_audio_embeddings: Optional mapping of video_id -> 512-D audio vector
            target_category_id: Optional target category ID for metadata matching
            target_channel_id: Optional target channel ID for metadata matching
            weights: Optional custom ReRankingWeights
            top_k: Number of re-ranked candidates to return
            ranking_mode: 'baseline' (preserves Stage 1 ranking) or 'rerank'

        Returns:
            List of (Video, final_score) tuples sorted descending by score.
        """
        if not candidates:
            return []

        # In baseline mode, preserve Stage 1 candidates directly
        if ranking_mode == "baseline":
            return candidates[:top_k]

        # Validate query embeddings
        query_text_norm = self.validate_vector(query_text_embedding, "query_text")

        query_vis_norm = None
        if query_visual_embedding is not None:
            try:
                query_vis_norm = self.validate_vector(query_visual_embedding, "query_visual")
            except Exception as e:
                logger.warning("Query visual embedding invalid: %s", e)

        query_aud_norm = None
        if query_audio_embedding is not None:
            try:
                query_aud_norm = self.validate_vector(query_audio_embedding, "query_audio")
            except Exception as e:
                logger.warning("Query audio embedding invalid: %s", e)

        active_weights = weights or self.default_weights
        active_weights.validate_positive_total()

        cand_vis_map = candidate_visual_embeddings or {}
        cand_aud_map = candidate_audio_embeddings or {}

        scored_candidates: List[Tuple[Video, float]] = []
        for cand_video, s1_score in candidates:
            cand_id = cand_video.video_id
            cand_vis = cand_vis_map.get(cand_id)
            cand_aud = cand_aud_map.get(cand_id)

            score, _ = self.score_candidate(
                candidate_video=cand_video,
                query_text_norm=query_text_norm,
                query_visual_norm=query_vis_norm,
                query_audio_norm=query_aud_norm,
                candidate_visual_vector=cand_vis,
                candidate_audio_vector=cand_aud,
                target_category_id=target_category_id,
                target_channel_id=target_channel_id,
                weights=active_weights,
                stage1_score=s1_score,
            )
            scored_candidates.append((cand_video, score))

        # Deterministic sorting: primary key is -score (descending), tie-breaker is video_id (ascending)
        scored_candidates.sort(key=lambda item: (-item[1], str(item[0].video_id)))

        return scored_candidates[:top_k]

    def hydrate_candidate_features(
        self,
        candidate_ids: List[str],
        session: Session,
        modality: str = "VISUAL_FRAMES",
        feature_type: str = "resnet50_projected_512",
    ) -> Dict[str, List[float]]:
        """
        Hydrate multimodal candidate embeddings from video_features table in a single batch query.
        Avoids N+1 database queries.
        Only returns verified, active features.
        """
        from src.db.repository import VideoFeatureRepository
        feat_repo = VideoFeatureRepository(session)
        return feat_repo.get_features_by_video_ids(candidate_ids, modality, feature_type)

    def retrieve_visual_candidates(
        self,
        query_video_id: str,
        session: Session,
        top_k: int = 5,
        modality: str = "VISUAL_FRAMES",
        feature_type: str = "resnet50_mean_2048",
        expected_dim: int = 2048,
    ) -> List[Dict[str, Any]]:
        """
        Phase 12C Experiment A: Direct Visual Video-to-Video Retrieval.

        Retrieves top-K visually similar candidate videos using raw feature representations
        within the same feature space.

        Guarantees:
        - Validates query and candidate feature dimensions and finiteness.
        - Computes cosine similarity only within the identical feature space.
        - Excludes query item from candidate results.
        - Applies deterministic tie-breaking (by score descending, video_id ascending).
        """
        from src.db.repository import VideoFeatureRepository
        feat_repo = VideoFeatureRepository(session)

        # 1. Fetch query feature
        query_feat = feat_repo.get_feature(query_video_id, modality, feature_type)
        if not query_feat or not query_feat.embedding:
            raise ValueError(
                f"Visual feature not found for query video '{query_video_id}' "
                f"(modality='{modality}', feature_type='{feature_type}')"
            )

        # 2. Validate query vector
        query_arr = np.asarray(query_feat.embedding, dtype=np.float32)
        if len(query_arr) != expected_dim:
            raise ValueError(
                f"Query vector dimension mismatch: expected {expected_dim}, got {len(query_arr)}"
            )
        if not np.all(np.isfinite(query_arr)):
            raise ValueError("Query vector contains NaN or infinite values")
        q_norm_val = float(np.linalg.norm(query_arr))
        if q_norm_val < 1e-7:
            raise ValueError(f"Query vector has zero or negligible norm ({q_norm_val})")
        q_unit = query_arr / q_norm_val

        # 3. Retrieve all candidate features of the exact same modality and feature_type
        all_features = feat_repo.get_features_by_modality(modality, feature_type)

        scored_candidates: List[Dict[str, Any]] = []
        for cand_feat in all_features:
            cand_id = cand_feat.video_id
            # Query exclusion guarantee
            if cand_id == query_video_id:
                continue

            if not cand_feat.embedding:
                continue

            cand_arr = np.asarray(cand_feat.embedding, dtype=np.float32)
            if len(cand_arr) != expected_dim or not np.all(np.isfinite(cand_arr)):
                continue

            c_norm_val = float(np.linalg.norm(cand_arr))
            if c_norm_val < 1e-7:
                continue
            c_unit = cand_arr / c_norm_val

            sim = float(np.dot(q_unit, c_unit))
            sim_bounded = max(-1.0, min(1.0, sim))

            scored_candidates.append({
                "video_id": cand_id,
                "score": sim_bounded,
                "feature_type": feature_type,
                "dimension": expected_dim,
            })

        # 4. Deterministic tie-breaking: primary key = -score (descending), secondary = video_id (ascending)
        scored_candidates.sort(key=lambda item: (-item["score"], str(item["video_id"])))

        # Assign ranks
        results = scored_candidates[:top_k]
        for idx, item in enumerate(results, start=1):
            item["rank"] = idx

        return results

    @classmethod
    def get_production_safe_weights(cls) -> ReRankingWeights:
        """
        Return safe production ranking weights.
        Visual and audio weights are strictly 0.0 because Phase 12C establishes that
        cross-modal alignment is exploratory and catalog audio features are unavailable.
        """
        return ReRankingWeights(
            w_text=0.90,
            w_visual=0.0,
            w_audio=0.0,
            w_metadata=0.10,
        )

