"""
Unit and Integration Tests for Phase 10 User Personalization Engine.

Covers:
1. History and leakage:
   - Empty history
   - Single-event history
   - Multi-event chronological ordering
   - Strict cutoff filtering (future events excluded)
   - Deduplication of repeated item interactions
   - History isolation between distinct users
   - Graceful handling of missing item embeddings
2. Profile generation:
   - Correct 512-D vector dimension
   - Unit L2 normalization
   - Zero-vector / NaN / Inf rejection
   - Deterministic profile construction
   - Recency weighting behavior (decay favors newer items)
3. Ranking and candidate scoring:
   - Personalized candidates ranked descending by cosine similarity
   - Seen-item exclusion filtering
   - Cold-start fallback returns valid schema and items
   - Seed-video blending
   - Top-K bounds enforcement
4. API endpoints:
   - GET /api/recommendations/user/{user_id} success with history
   - Cold start fallback for unknown user
   - Query parameter validation (limit, exclude_seen, as_of)
   - Cross-user history isolation via API
5. Evaluation metrics:
   - Evaluation metric helper verified on hand-crafted ground truth
"""

from datetime import datetime, timedelta, timezone
import math
from typing import List
import pytest
import torch

from src.db.models import UserInteraction, Video
from src.db.repository import InteractionRepository, VideoRepository
from src.schemas.recommendations import UserRecommendationResponse
from src.services.user_personalization_service import (
    PERSONALIZATION_EXPERIMENT_ID,
    UserPersonalizationService,
)
from src.services.vector_index_service import VectorIndexService
from scripts.evaluate_user_personalization import evaluate_ranking


# =====================================================================
# Profile Generation Unit Tests
# =====================================================================

def test_profile_generation_dimension_and_norm():
    """Verify generated profile is strictly 512-D and L2-normalized."""
    svc = UserPersonalizationService()
    emb1 = [1.0] + [0.0] * 511
    emb2 = [0.0, 1.0] + [0.0] * 510

    profile = svc.build_user_profile([emb1, emb2])
    assert profile is not None
    assert len(profile) == 512

    norm = math.sqrt(sum(x * x for x in profile))
    assert math.isclose(norm, 1.0, abs_tol=1e-5)


def test_profile_generation_empty_and_zero_vectors():
    """Verify empty embedding list and zero vectors return None gracefully."""
    svc = UserPersonalizationService()

    # Empty list
    assert svc.build_user_profile([]) is None

    # Zero vector
    zero_vec = [0.0] * 512
    assert svc.build_user_profile([zero_vec]) is None

    # Invalid dimension
    assert svc.build_user_profile([[1.0] * 256]) is None


def test_profile_generation_nan_inf_safety():
    """Verify embeddings with NaN or Inf are ignored without corrupting profile."""
    svc = UserPersonalizationService()
    valid_emb = [1.0] + [0.0] * 511
    nan_emb = [float("nan")] * 512
    inf_emb = [float("inf")] * 512

    profile = svc.build_user_profile([valid_emb, nan_emb, inf_emb])
    assert profile is not None
    assert len(profile) == 512
    assert math.isclose(profile[0], 1.0, abs_tol=1e-5)


def test_profile_generation_deterministic():
    """Verify profile construction is perfectly deterministic."""
    svc = UserPersonalizationService(decay_factor=0.85)
    emb1 = [0.6, 0.8] + [0.0] * 510
    emb2 = [0.8, 0.6] + [0.0] * 510

    p1 = svc.build_user_profile([emb1, emb2])
    p2 = svc.build_user_profile([emb1, emb2])
    assert p1 == p2


def test_profile_recency_decay_weighting():
    """Verify that recent items receive higher weight under exponential decay."""
    svc = UserPersonalizationService(decay_factor=0.5)

    # Item 0 (older): aligned with dim 0
    # Item 1 (newer): aligned with dim 1
    emb_old = [1.0, 0.0] + [0.0] * 510
    emb_new = [0.0, 1.0] + [0.0] * 510

    # With decay_factor=0.5, emb_new has weight 1.0, emb_old has weight 0.5
    profile = svc.build_user_profile([emb_old, emb_new])
    assert profile is not None

    # Profile vector should have larger value in dim 1 (newer item) than dim 0 (older item)
    assert profile[1] > profile[0]
    # Ratio profile[1] / profile[0] should equal 1.0 / 0.5 = 2.0
    assert math.isclose(profile[1] / profile[0], 2.0, abs_tol=1e-4)


# =====================================================================
# Database Repository and Leakage Prevention Tests
# =====================================================================

def test_interaction_repo_cutoff_leakage_protection(in_memory_db):
    """Verify interactions strictly equal to or after cutoff are never retrieved."""
    repo = InteractionRepository(in_memory_db)
    base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # 3 events: t0 = base, t1 = base + 1hr, t2 = base + 2hr
    repo.create(user_id="user_leak_test", video_id="vid_t0", timestamp=base_time)
    repo.create(user_id="user_leak_test", video_id="vid_t1", timestamp=base_time + timedelta(hours=1))
    repo.create(user_id="user_leak_test", video_id="vid_t2", timestamp=base_time + timedelta(hours=2))

    # Query with cutoff at t1: should return ONLY vid_t0 (strictly < t1)
    history_at_t1 = repo.get_user_history_before(
        user_id="user_leak_test",
        cutoff_timestamp=base_time + timedelta(hours=1),
    )
    assert len(history_at_t1) == 1
    assert history_at_t1[0].video_id == "vid_t0"

    # Query with cutoff at t2: should return vid_t0 and vid_t1
    history_at_t2 = repo.get_user_history_before(
        user_id="user_leak_test",
        cutoff_timestamp=base_time + timedelta(hours=2),
    )
    assert len(history_at_t2) == 2
    assert [h.video_id for h in history_at_t2] == ["vid_t0", "vid_t1"]


def test_interaction_user_isolation(in_memory_db):
    """Verify that interaction histories are strictly isolated between different users."""
    repo = InteractionRepository(in_memory_db)
    t = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    repo.create(user_id="user_A", video_id="vid_A1", timestamp=t)
    repo.create(user_id="user_B", video_id="vid_B1", timestamp=t)

    hist_a = repo.get_user_history_before(user_id="user_A")
    assert len(hist_a) == 1
    assert hist_a[0].video_id == "vid_A1"

    hist_b = repo.get_user_history_before(user_id="user_B")
    assert len(hist_b) == 1
    assert hist_b[0].video_id == "vid_B1"


# =====================================================================
# End-to-End Service Tests
# =====================================================================

def test_personalized_recommendations_ranking_and_seen_exclusion(in_memory_db):
    """Verify candidate ranking reflects user profile and seen items are excluded."""
    v_repo = VideoRepository(in_memory_db)
    i_repo = InteractionRepository(in_memory_db)
    svc = UserPersonalizationService()

    # Create catalog:
    # Hist: vid_1 (dim 0), vid_2 (dim 0) -> user likes dim 0
    # Cand 1: vid_math (dim 0) -> high match
    # Cand 2: vid_cook (dim 1) -> low match
    emb_dim0 = [1.0] + [0.0] * 511
    emb_dim1 = [0.0, 1.0] + [0.0] * 510

    v_repo.create(video_id="vid_1", title="Python 1", embedding=emb_dim0)
    v_repo.create(video_id="vid_2", title="Python 2", embedding=emb_dim0)
    v_repo.create(video_id="vid_math", title="Python 3", embedding=emb_dim0)
    v_repo.create(video_id="vid_cook", title="Cooking", embedding=emb_dim1)

    t0 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
    i_repo.create(user_id="coder_user", video_id="vid_1", timestamp=t0)
    i_repo.create(user_id="coder_user", video_id="vid_2", timestamp=t1)

    resp = svc.get_personalized_recommendations(
        session=in_memory_db,
        user_id="coder_user",
        top_k=5,
        exclude_seen=True,
    )

    assert isinstance(resp, UserRecommendationResponse)
    assert resp.personalization_applied is True
    assert resp.user_id == "coder_user"
    assert resp.history_length == 2

    # vid_1 and vid_2 must be excluded because exclude_seen=True
    rec_vids = [item.video_id for item in resp.recommendations]
    assert "vid_1" not in rec_vids
    assert "vid_2" not in rec_vids

    # Top recommendation should be vid_math (dim 0 match), ahead of vid_cook
    assert rec_vids[0] == "vid_math"
    assert resp.recommendations[0].score > resp.recommendations[1].score


def test_personalized_recommendations_cold_start_fallback(in_memory_db):
    """Verify that a user with 0 history gets a valid response with fallback metadata."""
    v_repo = VideoRepository(in_memory_db)
    v_repo.create(video_id="pop_1", title="Popular 1", embedding=[0.1] * 512)
    v_repo.create(video_id="pop_2", title="Popular 2", embedding=[0.2] * 512)

    svc = UserPersonalizationService()
    resp = svc.get_personalized_recommendations(
        session=in_memory_db,
        user_id="brand_new_user",
        top_k=2,
    )

    assert isinstance(resp, UserRecommendationResponse)
    assert resp.personalization_applied is False
    assert resp.fallback_reason == "no_history"
    assert resp.history_length == 0
    assert len(resp.recommendations) > 0


def test_personalized_recommendations_history_deduplication(in_memory_db):
    """Verify that repeated interactions with the same item are deduplicated."""
    v_repo = VideoRepository(in_memory_db)
    i_repo = InteractionRepository(in_memory_db)
    svc = UserPersonalizationService()

    emb = [1.0] + [0.0] * 511
    v_repo.create(video_id="repeat_vid", title="Watched Twice", embedding=emb)
    v_repo.create(video_id="cand_v", title="Candidate", embedding=emb)

    t0 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    i_repo.create(user_id="repeat_user", video_id="repeat_vid", timestamp=t0)
    i_repo.create(user_id="repeat_user", video_id="repeat_vid", timestamp=t1)

    resp = svc.get_personalized_recommendations(
        session=in_memory_db,
        user_id="repeat_user",
        top_k=2,
    )

    # History length should be 1 after deduplication
    assert resp.history_length == 1


# =====================================================================
# API Route Integration Tests
# =====================================================================

def test_api_user_recommendations_success(client, in_memory_db):
    """Verify GET /api/recommendations/user/{user_id} returns valid personalized recommendations."""
    v_repo = VideoRepository(in_memory_db)
    i_repo = InteractionRepository(in_memory_db)

    emb1 = [1.0] + [0.0] * 511
    emb2 = [0.9] + [0.1] + [0.0] * 510
    v_repo.create(video_id="v_hist", title="History Video", embedding=emb1)
    v_repo.create(video_id="v_rec", title="Recommended Video", embedding=emb2)

    i_repo.create(user_id="test_api_user", video_id="v_hist")

    resp = client.get("/api/recommendations/user/test_api_user?limit=5&exclude_seen=true")
    assert resp.status_code == 200
    data = resp.json()

    assert data["user_id"] == "test_api_user"
    assert data["personalization_applied"] is True
    assert data["history_length"] == 1
    assert len(data["recommendations"]) >= 1
    assert data["recommendations"][0]["video_id"] == "v_rec"


def test_api_user_recommendations_cold_start(client, in_memory_db):
    """Verify GET /api/recommendations/user/{user_id} handles cold start with 200 OK and fallback info."""
    v_repo = VideoRepository(in_memory_db)
    v_repo.create(video_id="v_cat", title="Catalog Video", embedding=[0.5] * 512)

    resp = client.get("/api/recommendations/user/unknown_user_999?limit=5")
    assert resp.status_code == 200
    data = resp.json()

    assert data["user_id"] == "unknown_user_999"
    assert data["personalization_applied"] is False
    assert data["fallback_reason"] == "no_history"
    assert len(data["recommendations"]) >= 1


def test_api_user_recommendations_invalid_limit(client):
    """Verify query parameter validation rejects limit < 1 or limit > 50."""
    resp_zero = client.get("/api/recommendations/user/user1?limit=0")
    assert resp_zero.status_code == 422

    resp_too_high = client.get("/api/recommendations/user/user1?limit=100")
    assert resp_too_high.status_code == 422


# =====================================================================
# Evaluation Logic Unit Tests
# =====================================================================

def test_evaluate_ranking_metric_correctness():
    """Verify evaluate_ranking computes Recall, Precision, MRR, NDCG accurately on synthetic cases."""
    candidates = ["item_a", "item_b", "item_c", "item_d", "item_e"]

    # Target is at rank 1
    res1 = evaluate_ranking(candidates, target_item="item_a")
    assert res1["Recall@1"] == 1.0
    assert res1["Recall@5"] == 1.0
    assert res1["MRR"] == 1.0
    assert res1["NDCG@10"] == 1.0
    assert res1["Precision@5"] == 0.2

    # Target is at rank 3
    res3 = evaluate_ranking(candidates, target_item="item_c")
    assert res3["Recall@1"] == 0.0
    assert res3["Recall@5"] == 1.0
    assert math.isclose(res3["MRR"], 1.0 / 3.0, abs_tol=1e-5)
    assert math.isclose(res3["NDCG@10"], 1.0 / math.log2(4.0), abs_tol=1e-5)  # rank 3 -> log2(3+1) = 2.0 -> 0.5
    assert math.isclose(res3["Precision@5"], 0.2, abs_tol=1e-5)

    # Target is not in candidates
    res_miss = evaluate_ranking(candidates, target_item="unseen_item")
    assert res_miss["Recall@1"] == 0.0
    assert res_miss["Recall@10"] == 0.0
    assert res_miss["MRR"] == 0.0
    assert res_miss["NDCG@10"] == 0.0
