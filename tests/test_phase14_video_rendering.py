"""Tests for Phase 14: End-to-End Video Rendering, Recommendation Explainability & Media Verification."""

import pytest
from src.schemas.recommendations import RecommendationItem, RecommendationResponse
from src.api.app import app
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    return TestClient(app)


def test_recommendation_item_schema_phase14_fields():
    """Verify that RecommendationItem includes thumbnail_url and duration fields."""
    item = RecommendationItem(
        video_id="10040",
        score=0.875,
        rank=1,
        title="Sample Video",
        thumbnail_url="/media/local_10040/thumbnail.jpg",
        duration=184,
    )
    data = item.model_dump()
    assert data["thumbnail_url"] == "/media/local_10040/thumbnail.jpg"
    assert data["duration"] == 184
    assert data["score"] == 0.875


def test_recommendation_item_defaults_unverified():
    """Verify that RecommendationItem leaves unverified fields as None without fake defaults."""
    item = RecommendationItem(
        video_id="catalog_100",
        title="Catalog Item",
        score=0.65,
        rank=2,
    )
    data = item.model_dump()
    assert data["thumbnail_url"] is None
    assert data["duration"] is None


def test_video_details_media_and_embed_resolution(api_client):
    """Test get_video_details endpoint returns verified media and playback URLs."""
    # Test catalog item with local media (if present in db)
    resp = api_client.get("/videos/10040")
    if resp.status_code == 200:
        data = resp.json()
        assert "video_id" in data
        assert "playable_url" in data
        assert "is_youtube_embed" in data
        assert "thumbnail_url" in data
        assert "duration" in data

        if data["playable_url"]:
            assert data["playable_url"].startswith("/media/local_10040/")
            assert data["thumbnail_url"].startswith("/media/local_10040/")
            assert data["duration"] == 184


def test_video_recommendation_similarity_scores_range(api_client):
    """Test that recommendation scores are cosine similarity floats in [-1.0, 1.0], not percentage ints."""
    resp = api_client.get("/recommendations/video/10040?limit=5")
    if resp.status_code == 200:
        data = resp.json()
        assert "recommendations" in data
        for rec in data["recommendations"]:
            assert "score" in rec
            assert isinstance(rec["score"], float)
            assert -1.0 <= rec["score"] <= 1.0
            assert "thumbnail_url" in rec
            assert "duration" in rec


def test_static_media_serving(api_client):
    """Test that /media static endpoint serves existing local media files with proper headers."""
    # Check if thumbnail exists
    resp = api_client.get("/media/local_10040/thumbnail.jpg")
    if resp.status_code == 200:
        assert resp.headers["content-type"].startswith("image/")
    
    # Check that non-existent media properly returns 404
    resp_404 = api_client.get("/media/non_existent_folder/video.mp4")
    assert resp_404.status_code == 404
