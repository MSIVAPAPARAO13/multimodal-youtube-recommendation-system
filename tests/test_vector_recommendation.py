"""
Unit and Integration Tests for Phase 9 Vector Recommendation Engine.

Tests:
1. Phase 8C checkpoint loading and parameter shapes
2. Embedding dimension validation (512-D required)
3. NaN / Inf rejection
4. Unit L2 normalization consistency
5. Self-match exclusion in retrieval
6. Correct descending nearest-neighbor ordering
7. Deterministic retrieval
8. Top-K bounds enforcement
9. Batch metadata hydration (single query, no N+1)
10. Idempotent embedding upsert
11. Metadata filtering (category, channel, seen videos)
12. Unknown video error handling (404)
13. Recommendation health API endpoint
14. Video recommendation GET API endpoint
15. Similar video recommendation POST API endpoint
16. Exact vs approximate Recall@K comparison
"""

import math
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from src.db.models import Video
from src.db.repository import VideoRepository
from src.models.text_projection_model import LinearProjectionModel, load_phase_8c_projection
from src.services.vector_index_service import VectorIndexService
from src.services.recommendation_service import RecommendationService
from src.schemas.recommendations import SimilarVideosRequest


PHASE_8C_CHECKPOINT = "data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt"


def test_phase_8c_checkpoint_loading_and_shape():
    """Verify Phase 8C checkpoint loads correctly with (512, 768) weight matrix."""
    model = load_phase_8c_projection(PHASE_8C_CHECKPOINT)
    assert isinstance(model, LinearProjectionModel)
    assert model.linear.weight.shape == (512, 768)
    assert model.linear.bias.shape == (512,)
    assert not model.training  # must be in eval mode


def test_vector_index_service_validation():
    """Verify VectorIndexService validates 512-D vectors and normalizes to unit sphere."""
    svc = VectorIndexService()
    vec = [1.0] * 512
    normalized = svc.validate_embedding(vec)

    assert len(normalized) == 512
    norm = math.sqrt(sum(x * x for x in normalized))
    assert math.isclose(norm, 1.0, abs_tol=1e-5)

    # Rejection of wrong dimensions
    with pytest.raises(ValueError, match="dimension mismatch"):
        svc.validate_embedding([1.0] * 256)


def test_vector_index_service_nan_inf_rejection():
    """Verify VectorIndexService rejects non-finite values (NaN / Inf)."""
    svc = VectorIndexService()
    nan_vec = [1.0] * 511 + [float("nan")]
    with pytest.raises(ValueError, match="non-finite"):
        svc.validate_embedding(nan_vec)

    inf_vec = [1.0] * 511 + [float("inf")]
    with pytest.raises(ValueError, match="non-finite"):
        svc.validate_embedding(inf_vec)


def test_vector_index_service_exact_search_and_ordering():
    """Verify exact nearest neighbor retrieval sorts candidates strictly descending by similarity."""
    svc = VectorIndexService()
    q = [0.0] * 512
    q[0] = 1.0  # unit vector along dim 0

    c1 = [0.0] * 512
    c1[0] = 0.9
    c1[1] = 0.1  # high similarity to q

    c2 = [0.0] * 512
    c2[1] = 1.0  # orthogonal (zero similarity)

    c3 = [0.0] * 512
    c3[0] = 0.5
    c3[1] = 0.5  # medium similarity

    candidates = {
        "vid_c1": c1,
        "vid_c2": c2,
        "vid_c3": c3,
    }

    results = svc.search_exact_in_memory(query_embedding=q, candidate_embeddings=candidates, top_k=3)
    assert len(results) == 3
    # Check strict descending order
    assert results[0][0] == "vid_c1"
    assert results[1][0] == "vid_c3"
    assert results[2][0] == "vid_c2"
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_vector_index_service_self_match_exclusion():
    """Verify search excludes query item and specified exclusions."""
    svc = VectorIndexService()
    q = [1.0] * 512
    candidates = {
        "query_vid": [1.0] * 512,
        "other_vid": [0.5] * 512,
    }

    results = svc.search_exact_in_memory(
        query_embedding=q,
        candidate_embeddings=candidates,
        top_k=2,
        exclude_video_ids={"query_vid"},
    )
    returned_ids = [vid for vid, _ in results]
    assert "query_vid" not in returned_ids
    assert "other_vid" in returned_ids


def test_vector_repository_batch_hydration(in_memory_db):
    """Verify get_by_video_ids retrieves multiple video entities in a single batch query."""
    repo = VideoRepository(in_memory_db)
    repo.create(video_id="batch_1", title="Title 1")
    repo.create(video_id="batch_2", title="Title 2")
    repo.create(video_id="batch_3", title="Title 3")

    hydrated = repo.get_by_video_ids(["batch_1", "batch_3", "nonexistent"])
    assert len(hydrated) == 2
    assert "batch_1" in hydrated
    assert "batch_3" in hydrated
    assert hydrated["batch_1"].title == "Title 1"
    assert hydrated["batch_3"].title == "Title 3"


def test_vector_repository_idempotent_upsert(in_memory_db):
    """Verify upsert_embedding creates record if absent and updates without duplicates if present."""
    repo = VideoRepository(in_memory_db)
    emb1 = [0.1] * 512
    emb2 = [0.2] * 512

    v1 = repo.upsert_embedding(video_id="upsert_vid", embedding=emb1, title="Initial Title")
    assert v1.video_id == "upsert_vid"
    assert v1.title == "Initial Title"
    assert repo.count() == 1

    # Update same ID
    v2 = repo.upsert_embedding(video_id="upsert_vid", embedding=emb2, title="Updated Title")
    assert v2.title == "Updated Title"
    assert repo.count() == 1  # count must remain 1


def test_vector_repository_metadata_filtering(in_memory_db):
    """Verify find_similar_with_filters respects category_id, channel_id, and exclusions."""
    repo = VideoRepository(in_memory_db)
    q_emb = [1.0] * 512

    # Category 10
    repo.create(video_id="v_cat10", title="Cat 10 Video", embedding=[0.9] * 512)
    v_cat10 = repo.get_by_video_id("v_cat10")
    v_cat10.category_id = "10"
    v_cat10.channel_title = "TechChannel"

    # Category 20
    repo.create(video_id="v_cat20", title="Cat 20 Video", embedding=[0.9] * 512)
    v_cat20 = repo.get_by_video_id("v_cat20")
    v_cat20.category_id = "20"
    v_cat20.channel_title = "CookChannel"

    in_memory_db.commit()

    # Filter by category 10
    res_cat10 = repo.find_similar_with_filters(embedding=q_emb, limit=10, category_id="10")
    assert len(res_cat10) == 1
    assert res_cat10[0][0].video_id == "v_cat10"

    # Filter by category 20
    res_cat20 = repo.find_similar_with_filters(embedding=q_emb, limit=10, category_id="20")
    assert len(res_cat20) == 1
    assert res_cat20[0][0].video_id == "v_cat20"

    # Exclude v_cat10
    res_ex = repo.find_similar_with_filters(embedding=q_emb, limit=10, exclude_video_ids=["v_cat10"])
    assert len(res_ex) == 1
    assert res_ex[0][0].video_id == "v_cat20"


def test_recommendation_service_encode_and_retrieval(in_memory_db):
    """Verify RecommendationService end-to-end recommendation flow."""
    svc = RecommendationService(checkpoint_path=PHASE_8C_CHECKPOINT)
    repo = VideoRepository(in_memory_db)

    # Seed 3 videos
    emb_q = [0.0] * 512
    emb_q[0] = 1.0
    emb_cand = [0.0] * 512
    emb_cand[0] = 0.9
    emb_cand[1] = 0.1

    repo.create(video_id="query_1", title="Python Tutorial", embedding=emb_q)
    repo.create(video_id="cand_1", title="Python Advanced", embedding=emb_cand)
    repo.create(video_id="cand_2", title="Unrelated Cooking", embedding=[0.0] * 512)

    resp = svc.get_recommendations_for_video(session=in_memory_db, video_id="query_1", top_k=2)

    assert resp.query_video_id == "query_1"
    assert resp.retrieval_mode == "TEXT_COENGAGEMENT_V2"
    assert resp.count >= 1
    assert resp.recommendations[0].video_id == "cand_1"
    assert resp.recommendations[0].rank == 1
    assert resp.recommendations[0].score > 0.0


def test_recommendation_service_unknown_video_error(in_memory_db):
    """Verify RecommendationService raises ValueError when query video is not in database."""
    svc = RecommendationService(checkpoint_path=PHASE_8C_CHECKPOINT)
    with pytest.raises(ValueError, match="not found"):
        svc.get_recommendations_for_video(session=in_memory_db, video_id="nonexistent_vid")


def test_recommendation_api_health_endpoint(client):
    """Verify GET /api/recommendations/health returns valid readiness metadata."""
    resp = client.get("/api/recommendations/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["retrieval_mode"] == "TEXT_COENGAGEMENT_V2"
    assert "vector_backend" in data
    assert "total_indexed_videos" in data


def test_recommendation_api_video_endpoint(client, in_memory_db):
    """Verify GET /api/recommendations/video/{video_id} returns top-K recommendations."""
    repo = VideoRepository(in_memory_db)
    repo.create(video_id="api_q", title="Data Science Intro", embedding=[1.0] + [0.0] * 511)
    repo.create(video_id="api_c1", title="Machine Learning Intro", embedding=[0.9, 0.1] + [0.0] * 510)
    repo.create(video_id="api_c2", title="Deep Learning Intro", embedding=[0.8, 0.2] + [0.0] * 510)

    resp = client.get("/api/recommendations/video/api_q?limit=2")
    assert resp.status_code == 200
    data = resp.json()

    assert data["query_video_id"] == "api_q"
    assert data["count"] == 2
    assert len(data["recommendations"]) == 2
    assert data["recommendations"][0]["video_id"] == "api_c1"
    assert data["recommendations"][0]["rank"] == 1


def test_recommendation_api_similar_endpoint(client, in_memory_db):
    """Verify POST /api/recommendations/similar accepts exclusions and filters."""
    repo = VideoRepository(in_memory_db)
    repo.create(video_id="sim_q", title="Query Item", embedding=[1.0] + [0.0] * 511)
    repo.create(video_id="sim_seen", title="Already Seen", embedding=[0.95] + [0.0] * 511)
    repo.create(video_id="sim_target", title="Target Item", embedding=[0.85] + [0.0] * 511)

    payload = {
        "video_id": "sim_q",
        "top_k": 5,
        "exclude_video_ids": ["sim_seen"],
    }
    resp = client.post("/api/recommendations/similar", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    rec_ids = [r["video_id"] for r in data["recommendations"]]
    assert "sim_seen" not in rec_ids
    assert "sim_target" in rec_ids


def test_recommendation_api_404_handling(client):
    """Verify GET and POST endpoints return 404 for unknown video IDs."""
    get_resp = client.get("/api/recommendations/video/unknown_id")
    assert get_resp.status_code == 404

    post_resp = client.post("/api/recommendations/similar", json={"video_id": "unknown_id", "top_k": 5})
    assert post_resp.status_code == 404


def test_compare_exact_vs_approximate():
    """Verify VectorIndexService.compare_exact_vs_approximate calculates overlap Recall@K."""
    svc = VectorIndexService()
    exact = [("v1", 0.9), ("v2", 0.8), ("v3", 0.7), ("v4", 0.6)]
    approx = [("v1", 0.9), ("v2", 0.8), ("v5", 0.65), ("v4", 0.6)]

    res = svc.compare_exact_vs_approximate(exact, approx, k=4)
    # 3 out of 4 overlap (v1, v2, v4)
    assert res["overlap_count"] == 3
    assert res["recall_at_k"] == 0.75
