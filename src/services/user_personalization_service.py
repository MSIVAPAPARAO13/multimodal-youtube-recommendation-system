"""
User Personalization Service.

Experiment / Phase 10 Mode: EXP_MICROLENS_USER_PERSONALIZATION_V1
Retrieval Representation: TEXT_COENGAGEMENT_V2 (512-D)

Constructs recency-weighted user profile vectors from historical interactions,
enforces strict cutoff timestamp filtering to prevent future-data leakage,
and scores candidates using the existing Phase 9 vector retrieval engine.
"""

from datetime import datetime, timezone
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sqlalchemy.orm import Session

from src.db.config import get_settings, Settings
from src.db.models import UserInteraction, Video
from src.db.repository import InteractionRepository, VideoRepository
from src.schemas.recommendations import (
    RecommendationItem,
    UserRecommendationResponse,
)
from src.services.recommendation_service import (
    MODEL_VERSION,
    RETRIEVAL_MODE,
    RecommendationService,
)
from src.services.vector_index_service import VectorIndexService

logger = logging.getLogger(__name__)

PERSONALIZATION_EXPERIMENT_ID = "EXP_MICROLENS_USER_PERSONALIZATION_V1"


class UserPersonalizationService:
    """
    Personalization service synthesizing user interest profiles from historical interactions
    and scoring video candidates against the Phase 9 vector index.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        recommendation_service: Optional[RecommendationService] = None,
        vector_index_service: Optional[VectorIndexService] = None,
        decay_factor: float = 0.85,
        half_life_seconds: float = 86400.0 * 7.0,  # 7 days
        max_history_length: int = 50,
    ):
        self.settings = settings or get_settings()
        self.vector_index_service = vector_index_service or VectorIndexService(
            default_top_k=self.settings.VECTOR_DEFAULT_TOP_K,
            max_top_k=self.settings.VECTOR_MAX_TOP_K,
            backend=self.settings.VECTOR_SEARCH_BACKEND,
            ef_search=self.settings.VECTOR_HNSW_EF_SEARCH,
        )
        self.recommendation_service = recommendation_service or RecommendationService(
            settings=self.settings,
            vector_index_service=self.vector_index_service,
        )
        self.decay_factor = decay_factor
        self.half_life_seconds = half_life_seconds
        self.max_history_length = max_history_length

    def build_user_profile(
        self,
        item_embeddings: List[List[float]],
        timestamps: Optional[List[float]] = None,
        cutoff_timestamp: Optional[float] = None,
        use_time_decay: bool = False,
    ) -> Optional[List[float]]:
        """
        Build a normalized 512-D user profile vector from historical item embeddings.

        Formula:
            profile_raw = sum_{k=0}^{m-1} (w_k * embedding_k)
            profile = normalize(profile_raw)

        Args:
            item_embeddings: Chronologically ordered embeddings (oldest to newest)
            timestamps: Optional timestamps (in seconds) corresponding to items
            cutoff_timestamp: Optional evaluation cutoff timestamp in seconds
            use_time_decay: If True and timestamps are present, uses exponential elapsed time decay;
                           otherwise uses exponential position decay.

        Returns:
            512-D L2-normalized profile vector, or None if no valid embeddings.
        """
        if not item_embeddings:
            return None

        m = len(item_embeddings)
        weights = []

        if use_time_decay and timestamps and cutoff_timestamp:
            lam = math.log(2.0) / max(self.half_life_seconds, 1.0)
            for ts in timestamps:
                delta = max(0.0, cutoff_timestamp - ts)
                w = math.exp(-lam * delta)
                weights.append(max(w, 1e-4))
        else:
            # Position decay: most recent item (k = m - 1) has weight 1.0; preceding items decay
            for k in range(m):
                w = self.decay_factor ** (m - 1 - k)
                weights.append(w)

        # Weighted sum of vectors
        profile_raw = np.zeros(512, dtype=np.float32)
        total_weight = 0.0

        for emb, w in zip(item_embeddings, weights):
            vec = np.array(emb, dtype=np.float32)
            if vec.shape[0] != 512 or not np.all(np.isfinite(vec)):
                continue
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
                profile_raw += w * vec
                total_weight += w

        if total_weight <= 1e-8:
            return None

        p_norm = float(np.linalg.norm(profile_raw))
        if p_norm <= 1e-8 or not np.isfinite(p_norm):
            return None

        profile = (profile_raw / p_norm).tolist()
        return profile

    def get_personalized_recommendations(
        self,
        session: Session,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True,
        cutoff_timestamp: Optional[datetime] = None,
        seed_video_id: Optional[str] = None,
        category_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        blend_weight: float = 0.7,
        ranking_mode: str = "baseline",
        candidate_pool_size: int = 50,
        weights: Optional[Dict[str, float]] = None,
    ) -> UserRecommendationResponse:
        """
        Generate personalized recommendations for a user.

        Guarantees:
        1. No future leakage: only interactions strictly earlier than cutoff_timestamp are used.
        2. Graceful cold start: falls back to content-based or catalog recommendations.
        3. Excludes seen items when exclude_seen=True.
        """
        interaction_repo = InteractionRepository(session)
        video_repo = VideoRepository(session)

        # 1. Fetch user history before cutoff
        raw_history = interaction_repo.get_user_history_before(
            user_id=user_id,
            cutoff_timestamp=cutoff_timestamp,
            limit=self.max_history_length * 2,  # allow headroom for deduplication
        )

        # 2. Deduplicate history (preserve chronologically, keeping latest event per item)
        seen_items: Set[str] = set()
        deduped_history: List[UserInteraction] = []

        # raw_history is ordered asc by timestamp
        for ev in raw_history:
            vid = ev.video_id
            seen_items.add(vid)
            # Remove previous occurrence if exists to keep latest position
            deduped_history = [h for h in deduped_history if h.video_id != vid]
            deduped_history.append(ev)

        # Enforce max history length
        if len(deduped_history) > self.max_history_length:
            deduped_history = deduped_history[-self.max_history_length:]

        # 3. Resolve historical item embeddings
        historical_embeddings: List[List[float]] = []
        historical_timestamps: List[float] = []

        if deduped_history:
            hist_vids = [h.video_id for h in deduped_history]
            hydrated_videos = video_repo.get_by_video_ids(hist_vids)

            for h in deduped_history:
                v = hydrated_videos.get(h.video_id)
                emb = None
                if v and v.embedding is not None:
                    emb = v.embedding
                elif v and v.title:
                    try:
                        emb = self.recommendation_service.encode_text(v.title)
                        video_repo.update_embedding(v.video_id, emb)
                    except Exception as e:
                        logger.debug("Could not encode historical video %s: %s", v.video_id, e)

                if emb is not None:
                    try:
                        valid_emb = self.vector_index_service.validate_embedding(emb)
                        historical_embeddings.append(valid_emb)
                        ts = h.timestamp.replace(tzinfo=timezone.utc).timestamp() if h.timestamp else 0.0
                        historical_timestamps.append(ts)
                    except Exception:
                        pass

        # 4. Check for cold-start condition
        cutoff_sec = (
            cutoff_timestamp.replace(tzinfo=timezone.utc).timestamp()
            if cutoff_timestamp
            else None
        )
        profile = self.build_user_profile(
            item_embeddings=historical_embeddings,
            timestamps=historical_timestamps,
            cutoff_timestamp=cutoff_sec,
            use_time_decay=bool(cutoff_sec is not None),
        )

        cutoff_iso = cutoff_timestamp.isoformat() if cutoff_timestamp else None

        # COLD START FALLBACK
        if profile is None:
            return self._handle_cold_start(
                session=session,
                user_id=user_id,
                top_k=top_k,
                seed_video_id=seed_video_id,
                category_id=category_id,
                channel_id=channel_id,
                cutoff_iso=cutoff_iso,
                fallback_reason="no_history" if not raw_history else "no_embedded_items",
                ranking_mode=ranking_mode,
            )

        # 5. Candidate generation using profile vector
        exclusions = set()
        if exclude_seen:
            exclusions.update(seen_items)
        if seed_video_id:
            exclusions.add(seed_video_id)

        stage1_k = max(candidate_pool_size, top_k) if ranking_mode == "rerank" else (top_k * 2 if seed_video_id else top_k)
        candidates, backend_used, _ = self.vector_index_service.search_database(
            session=session,
            query_embedding=profile,
            top_k=stage1_k,
            exclude_video_ids=list(exclusions),
            category_id=category_id,
            channel_id=channel_id,
        )

        # 6. Optional blending or Re-Ranking
        reranking_applied = False
        if ranking_mode == "rerank" and candidates:
            custom_w = None
            if weights:
                from src.services.reranking_service import ReRankingWeights
                custom_w = ReRankingWeights(**weights)

            candidates = self.recommendation_service.reranking_service.rerank_candidates(
                candidates=candidates,
                query_text_embedding=profile,
                target_category_id=category_id,
                target_channel_id=channel_id,
                weights=custom_w,
                top_k=top_k,
                ranking_mode="rerank",
            )
            reranking_applied = True
        elif seed_video_id and candidates:
            seed_v = video_repo.get_by_video_id(seed_video_id)
            seed_emb = seed_v.embedding if (seed_v and seed_v.embedding) else None
            if seed_emb:
                s_arr = np.array(seed_emb, dtype=np.float32)
                s_norm = np.linalg.norm(s_arr)
                if s_norm > 1e-6:
                    s_arr = s_arr / s_norm
                    scored_candidates = []
                    for c_vid, p_score in candidates:
                        c_vec = np.array(c_vid.embedding, dtype=np.float32)
                        c_norm = np.linalg.norm(c_vec)
                        c_vec = c_vec / (c_norm + 1e-8)
                        content_sim = float(np.dot(s_arr, c_vec))
                        blended_score = blend_weight * p_score + (1.0 - blend_weight) * content_sim
                        scored_candidates.append((c_vid, blended_score))
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    candidates = scored_candidates[:top_k]
            else:
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        items: List[RecommendationItem] = []
        for rank, (cand_video, score) in enumerate(candidates, start=1):
            items.append(
                RecommendationItem(
                    video_id=cand_video.video_id,
                    title=cand_video.title or "",
                    score=round(float(score), 4),
                    rank=rank,
                    category_id=cand_video.category_id,
                    channel_title=cand_video.channel_title,
                    thumbnail_url=cand_video.thumbnail_url,
                    duration=cand_video.duration,
                )
            )

        return UserRecommendationResponse(
            user_id=user_id,
            recommendations=items,
            count=len(items),
            personalization_applied=True,
            fallback_reason=None,
            history_length=len(historical_embeddings),
            cutoff_timestamp=cutoff_iso,
            retrieval_mode=RETRIEVAL_MODE,
            retrieval_backend=backend_used,
            model_version=PERSONALIZATION_EXPERIMENT_ID,
            ranking_mode=ranking_mode,
            reranking_applied=reranking_applied,
        )

    def _handle_cold_start(
        self,
        session: Session,
        user_id: str,
        top_k: int,
        seed_video_id: Optional[str],
        category_id: Optional[str],
        channel_id: Optional[str],
        cutoff_iso: Optional[str],
        fallback_reason: str,
        ranking_mode: str = "baseline",
    ) -> UserRecommendationResponse:
        """Gracefully handle cold-start users via content-based or catalog fallback."""
        backend_used = "exact_fallback"
        items: List[RecommendationItem] = []

        if seed_video_id:
            try:
                base_resp = self.recommendation_service.get_recommendations_for_video(
                    session=session,
                    video_id=seed_video_id,
                    top_k=top_k,
                    category_id=category_id,
                    channel_id=channel_id,
                )
                items = base_resp.recommendations
                backend_used = base_resp.retrieval_backend
            except Exception as e:
                logger.debug("Seed fallback failed: %s", e)

        if not items:
            # Catalog fallback
            video_repo = VideoRepository(session)
            catalog = video_repo.get_all(limit=top_k)
            for rank, v in enumerate(catalog, start=1):
                items.append(
                    RecommendationItem(
                        video_id=v.video_id,
                        title=v.title or "",
                        score=0.0,
                        rank=rank,
                        category_id=v.category_id,
                        channel_title=v.channel_title,
                        thumbnail_url=v.thumbnail_url,
                        duration=v.duration,
                    )
                )

        return UserRecommendationResponse(
            user_id=user_id,
            recommendations=items,
            count=len(items),
            personalization_applied=False,
            fallback_reason=fallback_reason,
            history_length=0,
            cutoff_timestamp=cutoff_iso,
            retrieval_mode=RETRIEVAL_MODE,
            retrieval_backend=backend_used,
            model_version=PERSONALIZATION_EXPERIMENT_ID,
            ranking_mode=ranking_mode,
            reranking_applied=False,
        )
