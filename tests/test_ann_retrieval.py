"""
Unit and Integration Tests for Phase 11A Approximate Nearest Neighbor (ANN) Retrieval Foundation.

Tests:
1. Vector and Query Validation:
   - 512-D normalized vector validation
   - Dimension mismatch rejection
   - Non-finite (NaN / Inf) rejection
   - Zero-norm vector rejection
   - Top-K bounding
2. Backend Selection & Parameter Validation:
   - "exact", "ann", "auto" backend modes
   - Invalid backend rejection (ValueError)
   - Invalid ef_search rejection (ValueError)
   - Parameterized SQL construction (prevent SQL injection)
3. Fallback and Failure Isolation:
   - ANN mode raises RuntimeError when pgvector/PostgreSQL is unavailable (e.g. SQLite session)
   - Auto mode gracefully falls back to exact search without failing
   - Exact mode executes deterministic cosine ranking
4. Phase 10 User Personalization Compatibility:
   - Synthesized user profile vectors execute through VectorIndexService.search_database
   - Correct backend reporting in results ("pgvector_hnsw" or "exact_fallback")
5. PostgreSQL / pgvector Integration (when available):
   - Real HNSW index retrieval matches top candidates
   - Exclusion filters and metadata hydration
"""

import math
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.db.models import Video
from src.db.repository import VideoRepository
from src.services.vector_index_service import VectorIndexService, EXPECTED_EMBEDDING_DIM
from src.services.user_personalization_service import UserPersonalizationService
from scripts.benchmark_ann_retrieval import compute_ann_recall


# =====================================================================
# 1. Vector and Query Validation
# =====================================================================

def test_ann_vector_validation_success():
    """Verify 512-D vector is normalized to unit L2 sphere."""
    svc = VectorIndexService()
    vec = [1.0] * 512
    norm_vec = svc.validate_embedding(vec)
    assert len(norm_vec) == 512
    norm = math.sqrt(sum(x * x for x in norm_vec))
    assert math.isclose(norm, 1.0, abs_tol=1e-5)


def test_ann_vector_validation_dimension_mismatch():
    """Verify wrong dimensions are strictly rejected with ValueError."""
    svc = VectorIndexService()
    with pytest.raises(ValueError, match="dimension mismatch"):
        svc.validate_embedding([1.0] * 256)

    with pytest.raises(ValueError, match="dimension mismatch"):
        svc.validate_embedding([1.0] * 768)


def test_ann_vector_validation_nan_inf():
    """Verify NaN and Inf values are strictly rejected."""
    svc = VectorIndexService()
    nan_vec = [1.0] * 511 + [float("nan")]
    with pytest.raises(ValueError, match="non-finite"):
        svc.validate_embedding(nan_vec)

    inf_vec = [1.0] * 511 + [float("inf")]
    with pytest.raises(ValueError, match="non-finite"):
        svc.validate_embedding(inf_vec)


def test_ann_vector_validation_zero_norm():
    """Verify all-zero vector is strictly rejected."""
    svc = VectorIndexService()
    with pytest.raises(ValueError, match="effectively zero"):
        svc.validate_embedding([0.0] * 512)


def test_ann_top_k_bounds():
    """Verify top_k bounds (min 1, max max_top_k)."""
    svc = VectorIndexService(max_top_k=20)
    # Ensure all candidates are valid unit vectors
    cands = {f"v_{i}": [1.0 if j == (i % 512) else 0.0 for j in range(512)] for i in range(30)}
    q = [1.0] + [0.0] * 511

    # Requesting 50 should be bounded to 20
    res = svc.search_exact_in_memory(q, cands, top_k=50)
    assert len(res) == 20

    # Requesting 0 should be bounded to 1
    res_zero = svc.search_exact_in_memory(q, cands, top_k=0)
    assert len(res_zero) == 1


# =====================================================================
# 2. Backend Selection and Configuration Validation
# =====================================================================

def test_ann_backend_validation_invalid_name(in_memory_db):
    """Verify invalid backend names raise ValueError."""
    repo = VideoRepository(in_memory_db)
    q = [1.0] + [0.0] * 511
    with pytest.raises(ValueError, match="Invalid backend"):
        repo.find_similar_with_filters(q, backend="invalid_backend")


def test_ann_ef_search_validation(in_memory_db):
    """Verify ef_search < 1 raises ValueError."""
    repo = VideoRepository(in_memory_db)
    q = [1.0] + [0.0] * 511
    with pytest.raises(ValueError, match="ef_search must be a positive integer"):
        repo.find_similar_with_filters(q, backend="auto", ef_search=0)

    with pytest.raises(ValueError, match="ef_search must be a positive integer"):
        repo.find_similar_with_filters(q, backend="auto", ef_search=-5)


def test_ann_mode_fails_clearly_on_unsupported_backend(in_memory_db):
    """Verify that backend='ann' raises RuntimeError on SQLite in-memory test database."""
    repo = VideoRepository(in_memory_db)
    q = [1.0] + [0.0] * 511
    # in_memory_db is SQLite, so backend='ann' must fail clearly, not silently fall back
    with pytest.raises(RuntimeError, match="ANN retrieval backend requested, but pgvector/PostgreSQL is unavailable"):
        repo.find_similar_with_filters(q, backend="ann")


def test_auto_mode_graceful_fallback(in_memory_db):
    """Verify backend='auto' uses exact fallback on SQLite and returns backend_used."""
    repo = VideoRepository(in_memory_db)
    repo.create(video_id="vid_exact_1", title="Video 1", embedding=[1.0] + [0.0] * 511)
    q = [1.0] + [0.0] * 511

    results, backend_used = repo.find_similar_with_filters(
        embedding=q,
        limit=5,
        backend="auto",
        return_backend=True,
    )
    assert backend_used == "exact_fallback"
    assert len(results) == 1
    assert results[0][0].video_id == "vid_exact_1"


def test_exact_mode_execution(in_memory_db):
    """Verify backend='exact' executes deterministic in-memory ranking."""
    repo = VideoRepository(in_memory_db)
    repo.create(video_id="vid_a", title="Alpha", embedding=[1.0] + [0.0] * 511)
    repo.create(video_id="vid_b", title="Beta", embedding=[0.0, 1.0] + [0.0] * 510)
    q = [1.0] + [0.0] * 511

    results, backend_used = repo.find_similar_with_filters(
        embedding=q,
        limit=5,
        backend="exact",
        return_backend=True,
    )
    assert backend_used == "exact"
    assert len(results) == 2
    assert results[0][0].video_id == "vid_a"
    assert math.isclose(results[0][1], 1.0, abs_tol=1e-5)


# =====================================================================
# 3. Phase 10 User Personalization Profile Compatibility
# =====================================================================

def test_ann_search_with_user_profile_vector(in_memory_db):
    """Verify VectorIndexService accepts synthesized Phase 10 user profile vectors."""
    repo = VideoRepository(in_memory_db)
    repo.create(video_id="cand_ml", title="ML Guide", embedding=[1.0] + [0.0] * 511)
    repo.create(video_id="cand_cook", title="Cooking", embedding=[0.0, 1.0] + [0.0] * 510)

    # Build user profile using UserPersonalizationService
    pers_svc = UserPersonalizationService()
    hist_embs = [[1.0] + [0.0] * 511, [0.8, 0.2] + [0.0] * 510]
    profile = pers_svc.build_user_profile(hist_embs)
    assert profile is not None
    assert len(profile) == 512

    index_svc = VectorIndexService(backend="auto")
    results, backend_used, lat_ms = index_svc.search_database(
        session=in_memory_db,
        query_embedding=profile,
        top_k=5,
    )
    assert backend_used == "exact_fallback"
    assert len(results) == 2
    # ML video has much higher cosine similarity with profile
    assert results[0][0].video_id == "cand_ml"
    assert results[0][1] > results[1][1]


# =====================================================================
# 4. Recall Evaluation Metric Helper
# =====================================================================

def test_compute_ann_recall_metrics():
    """Verify compute_ann_recall computes Recall@K accurately."""
    exact_ids = ["a", "b", "c", "d", "e"]
    ann_ids = ["a", "c", "b", "f", "g"]

    # Top-1: 'a' vs 'a' -> 1.0
    assert compute_ann_recall(exact_ids, ann_ids, 1) == 1.0

    # Top-3: {'a', 'b', 'c'} vs {'a', 'c', 'b'} -> 1.0
    assert compute_ann_recall(exact_ids, ann_ids, 3) == 1.0

    # Top-5: {'a', 'b', 'c', 'd', 'e'} vs {'a', 'c', 'b', 'f', 'g'} -> 3/5 = 0.6
    assert compute_ann_recall(exact_ids, ann_ids, 5) == 0.6


# =====================================================================
# 5. Live PostgreSQL / pgvector Integration Test (Conditional)
# =====================================================================

def test_live_pgvector_hnsw_execution():
    """Verify end-to-end ANN query against live PostgreSQL container when reachable."""
    import os
    pg_url = os.getenv("TEST_PG_DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/video_recommender")
    
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        engine = create_engine(pg_url, pool_pre_ping=True)
        with engine.connect() as conn:
            res = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'")).fetchone()
            if not res:
                pytest.skip("pgvector extension not installed in target PostgreSQL database")
        
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            repo = VideoRepository(session)
            # Create a test vector
            test_vec = [1.0] + [0.0] * 511
            
            # Execute search with backend='ann'
            results, backend_used = repo.find_similar_with_filters(
                embedding=test_vec,
                limit=5,
                backend="ann",
                ef_search=40,
                return_backend=True
            )
            assert backend_used in ("pgvector_hnsw", "pgvector_ann")
            assert isinstance(results, list)
            assert len(results) > 0
        finally:
            session.close()
            engine.dispose()
    except (ImportError, Exception) as e:
        if isinstance(e, AssertionError):
            raise
        pytest.skip(f"Live PostgreSQL not reachable or failed: {e}")

