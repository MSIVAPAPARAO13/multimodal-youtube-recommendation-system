"""
Tests for Phase 11B Candidate Re-Ranking Engine.

Covers:
1. ReRankingWeights validation (non-negative, finite, positive sum).
2. Vector validation (dimension, NaN/Inf, zero-norm).
3. CandidateReRankingService scoring logic:
   - Text relevance
   - Visual relevance (when present)
   - Audio relevance (when present)
   - Metadata matching (category, channel)
   - Missing modality dynamic weight renormalization
   - All-missing feature safe fallback
4. Deterministic tie-breaking by video_id.
5. End-to-end integration with RecommendationService and UserPersonalizationService.
6. API endpoint schema and execution with ranking_mode='rerank'.
"""

import math
from unittest.mock import MagicMock
import numpy as np
import pytest
from sqlalchemy.orm import Session

from src.db.models import Video, UserInteraction
from src.db.repository import VideoRepository, InteractionRepository
from src.schemas.recommendations import RecommendationResponse, UserRecommendationResponse
from src.services.recommendation_service import RecommendationService
from src.services.reranking_service import (
    CandidateReRankingService,
    ReRankingWeights,
)
from src.services.user_personalization_service import UserPersonalizationService
from src.services.vector_index_service import VectorIndexService


# =====================================================================
# 1. Weights and Configuration Validation
# =====================================================================

def test_weights_valid_construction():
    """Verify standard valid weight configuration."""
    w = ReRankingWeights(w_text=0.7, w_visual=0.15, w_audio=0.05, w_metadata=0.1)
    assert w.w_text == 0.7
    assert w.w_visual == 0.15
    assert w.total_weight() == pytest.approx(1.0)


def test_weights_rejection_negative():
    """Verify negative weights are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        ReRankingWeights(w_text=-0.1)


def test_weights_rejection_nan_inf():
    """Verify NaN and Inf weights are rejected."""
    with pytest.raises(ValueError, match="finite"):
        ReRankingWeights(w_text=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        ReRankingWeights(w_visual=float("inf"))


def test_weights_rejection_all_zero():
    """Verify weights summing to zero are rejected."""
    w = ReRankingWeights(w_text=0.0, w_visual=0.0, w_audio=0.0, w_metadata=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        w.validate_positive_total()


# =====================================================================
# 2. Vector Validation
# =====================================================================

def test_vector_validation_success():
    """Valid 512-D vector normalizes to unit norm."""
    service = CandidateReRankingService()
    vec = [2.0] * 512
    normed = service.validate_vector(vec)
    assert normed.shape == (512,)
    assert pytest.approx(float(np.linalg.norm(normed)), 1e-5) == 1.0


def test_vector_validation_dimension_mismatch():
    """Incorrect dimension raises ValueError."""
    service = CandidateReRankingService()
    with pytest.raises(ValueError, match="dimension mismatch"):
        service.validate_vector([1.0] * 256)


def test_vector_validation_nan_inf():
    """Vector with NaN or Inf raises ValueError."""
    service = CandidateReRankingService()
    vec_nan = [1.0] * 511 + [float("nan")]
    with pytest.raises(ValueError, match="NaN or infinite"):
        service.validate_vector(vec_nan)

    vec_inf = [1.0] * 511 + [float("inf")]
    with pytest.raises(ValueError, match="NaN or infinite"):
        service.validate_vector(vec_inf)


def test_vector_validation_zero_norm():
    """Zero vector raises ValueError."""
    service = CandidateReRankingService()
    with pytest.raises(ValueError, match="zero or negligible norm"):
        service.validate_vector([0.0] * 512)


# =====================================================================
# 3. Candidate Scoring and Active-Weight Renormalization
# =====================================================================

def test_score_text_only_renormalization():
    """When only text is available, text weight renormalizes to 1.0."""
    service = CandidateReRankingService()
    q_vec = service.validate_vector([1.0] + [0.0] * 511)
    cand_vec = [1.0] + [0.0] * 511

    v = Video(video_id="v1", title="Test Video", embedding=cand_vec)
    score, telemetry = service.score_candidate(
        candidate_video=v,
        query_text_norm=q_vec,
    )

    assert pytest.approx(score, 1e-4) == 1.0
    assert telemetry["modalities_present"] == ["text"]
    assert pytest.approx(telemetry["active_weights"]["text"], 1e-4) == service.default_weights.w_text


def test_score_multimodal_when_visual_present():
    """When both text and visual are present, composite score matches weighted sum."""
    w = ReRankingWeights(w_text=0.8, w_visual=0.2, w_audio=0.0, w_metadata=0.0)
    service = CandidateReRankingService(default_weights=w)

    q_text = service.validate_vector([1.0] + [0.0] * 511)
    q_vis = service.validate_vector([0.0, 1.0] + [0.0] * 510)

    # Candidate has identical text (sim=1.0) and orthogonal visual (sim=0.0)
    cand_text = [1.0] + [0.0] * 511
    cand_vis = [0.0, 0.0, 1.0] + [0.0] * 509

    v = Video(video_id="v1", title="Test Video", embedding=cand_text)
    score, telemetry = service.score_candidate(
        candidate_video=v,
        query_text_norm=q_text,
        query_visual_norm=q_vis,
        candidate_visual_vector=cand_vis,
    )

    # Expected: 0.8 * 1.0 + 0.2 * 0.0 = 0.8
    assert pytest.approx(score, 1e-4) == 0.8
    assert set(telemetry["modalities_present"]) == {"text", "visual"}


def test_score_metadata_matching():
    """Matching category and channel boosts metadata score."""
    w = ReRankingWeights(w_text=0.5, w_visual=0.0, w_audio=0.0, w_metadata=0.5)
    service = CandidateReRankingService(default_weights=w)

    q_text = service.validate_vector([1.0] + [0.0] * 511)
    cand_text = [1.0] + [0.0] * 511

    v = Video(
        video_id="v1",
        title="Test Video",
        embedding=cand_text,
        category_id="27",
        channel_id="chan_tech",
    )

    score, telemetry = service.score_candidate(
        candidate_video=v,
        query_text_norm=q_text,
        target_category_id="27",
        target_channel_id="chan_tech",
    )

    # text_sim = 1.0, meta_sim = (0.6*1.0 + 0.4*1.0) = 1.0 -> composite = 1.0
    assert pytest.approx(score, 1e-4) == 1.0
    assert "metadata" in telemetry["modalities_present"]


def test_score_all_missing_fallback():
    """If all modalities are missing/invalid, safely falls back to stage1_score."""
    service = CandidateReRankingService()
    q_text = service.validate_vector([1.0] + [0.0] * 511)

    v = Video(video_id="v_empty", title="No embedding", embedding=None)
    score, telemetry = service.score_candidate(
        candidate_video=v,
        query_text_norm=q_text,
        stage1_score=0.42,
    )

    assert pytest.approx(score, 1e-4) == 0.42
    assert telemetry["modalities_present"] == []


# =====================================================================
# 4. Deterministic Re-Ranking and Tie-Breaking
# =====================================================================

def test_rerank_deterministic_ordering_and_ties():
    """Candidates with equal scores are tie-broken deterministically by video_id."""
    service = CandidateReRankingService()
    q_text = [1.0] + [0.0] * 511

    # Two candidates with identical embeddings (identical cosine similarity)
    v_b = Video(video_id="v_beta", title="Beta", embedding=[1.0] + [0.0] * 511)
    v_a = Video(video_id="v_alpha", title="Alpha", embedding=[1.0] + [0.0] * 511)

    # Input has v_beta before v_alpha
    cands = [(v_b, 0.9), (v_a, 0.9)]
    reranked = service.rerank_candidates(
        candidates=cands,
        query_text_embedding=q_text,
        ranking_mode="rerank",
    )

    # Alphabetical tie-breaking puts v_alpha first
    assert len(reranked) == 2
    assert reranked[0][0].video_id == "v_alpha"
    assert reranked[1][0].video_id == "v_beta"


def test_rerank_baseline_mode_preserves_stage1():
    """When ranking_mode='baseline', Stage 1 ordering is strictly preserved."""
    service = CandidateReRankingService()
    v1 = Video(video_id="v1", title="First", embedding=[0.1] * 512)
    v2 = Video(video_id="v2", title="Second", embedding=[0.9] * 512)

    cands = [(v1, 0.95), (v2, 0.50)]
    reranked = service.rerank_candidates(
        candidates=cands,
        query_text_embedding=[1.0] + [0.0] * 511,
        ranking_mode="baseline",
    )

    assert len(reranked) == 2
    assert reranked[0][0].video_id == "v1"
    assert reranked[1][0].video_id == "v2"


# =====================================================================
# 5. Service & API Integration Tests
# =====================================================================

def test_recommendation_service_reranking_flow(in_memory_db: Session):
    """RecommendationService executes Stage 1 + Stage 2 re-ranking cleanly."""
    repo = VideoRepository(in_memory_db)
    v_seed = repo.create(video_id="seed", title="Seed Video", embedding=[1.0, 0.0] + [0.0] * 510)
    # cand1 has lower similarity (0.5), cand2 has higher similarity (1.0)
    v_c1 = repo.create(video_id="cand1", title="Candidate 1", embedding=[0.5, 0.866] + [0.0] * 510)
    v_c2 = repo.create(video_id="cand2", title="Candidate 2", embedding=[1.0, 0.0] + [0.0] * 510)
    in_memory_db.commit()

    service = RecommendationService()
    resp = service.get_recommendations_for_video(
        session=in_memory_db,
        video_id="seed",
        top_k=5,
        ranking_mode="rerank",
        candidate_pool_size=20,
    )

    assert resp.ranking_mode == "rerank"
    assert resp.reranking_applied is True
    assert len(resp.recommendations) == 2
    # cand2 has higher similarity to seed than cand1
    assert resp.recommendations[0].video_id == "cand2"
    assert resp.recommendations[1].video_id == "cand1"


def test_user_personalization_service_reranking_flow(in_memory_db: Session):
    """UserPersonalizationService applies re-ranking to user profile queries."""
    video_repo = VideoRepository(in_memory_db)
    int_repo = InteractionRepository(in_memory_db)

    # 1. Historical videos
    v1 = video_repo.create(video_id="h1", title="History 1", embedding=[1.0, 0.0] + [0.0] * 510)
    v2 = video_repo.create(video_id="h2", title="History 2", embedding=[0.9, 0.1] + [0.0] * 510)
    # 2. Candidate videos
    c1 = video_repo.create(video_id="c1", title="Candidate 1", embedding=[0.1, 0.99] + [0.0] * 510)
    c2 = video_repo.create(video_id="c2", title="Candidate 2", embedding=[1.0, 0.0] + [0.0] * 510)

    int_repo.create("user_100", "h1", "watch")
    int_repo.create("user_100", "h2", "watch")
    in_memory_db.commit()

    pers_service = UserPersonalizationService()
    resp = pers_service.get_personalized_recommendations(
        session=in_memory_db,
        user_id="user_100",
        top_k=5,
        ranking_mode="rerank",
        candidate_pool_size=20,
    )

    assert resp.personalization_applied is True
    assert resp.ranking_mode == "rerank"
    assert resp.reranking_applied is True
    assert len(resp.recommendations) > 0
    # c2 is far closer to user's history profile [1.0, 0.0, ...] than c1
    assert resp.recommendations[0].video_id == "c2"


def test_api_video_recommendations_with_reranking(client, in_memory_db: Session):
    """API endpoint GET /api/recommendations/video/{id} supports ranking_mode=rerank."""
    repo = VideoRepository(in_memory_db)
    repo.create(video_id="v_api_seed", title="API Seed Video", embedding=[1.0] + [0.0] * 511)
    repo.create(video_id="v_api_c1", title="API Cand 1", embedding=[0.9] + [0.0] * 511)
    in_memory_db.commit()

    # Query baseline
    r_base = client.get("/api/recommendations/video/v_api_seed?ranking_mode=baseline")
    assert r_base.status_code == 200
    data_base = r_base.json()
    assert data_base["ranking_mode"] == "baseline"
    assert data_base["reranking_applied"] is False

    # Query rerank
    r_rerank = client.get("/api/recommendations/video/v_api_seed?ranking_mode=rerank&candidate_pool_size=30")
    assert r_rerank.status_code == 200
    data_rerank = r_rerank.json()
    assert data_rerank["ranking_mode"] == "rerank"
    assert data_rerank["reranking_applied"] is True


def test_api_user_recommendations_with_reranking(client, in_memory_db: Session):
    """API endpoint GET /api/recommendations/user/{id} supports ranking_mode=rerank."""
    video_repo = VideoRepository(in_memory_db)
    int_repo = InteractionRepository(in_memory_db)

    video_repo.create(video_id="v_h", title="Watched", embedding=[1.0] + [0.0] * 511)
    video_repo.create(video_id="v_rec", title="Recommend Me", embedding=[0.95] + [0.0] * 511)
    int_repo.create("user_api", "v_h", "watch")
    in_memory_db.commit()

    r = client.get("/api/recommendations/user/user_api?ranking_mode=rerank&candidate_pool_size=25")
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == "user_api"
    assert data["ranking_mode"] == "rerank"
    assert data["reranking_applied"] is True
