from unittest.mock import MagicMock
from api.app import state
from db.repository import VideoRepository


def test_recommend_validation_missing_video(client):
    """Test recommendation returns 404 if video is not in index."""
    state.model = MagicMock()
    state.is_trained = True

    response = client.post("/recommend", json={"video_id": "non_existent_vid", "top_k": 5})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_recommend_validation_invalid_top_k(client):
    """Test recommendation rejects invalid top_k outside range [1, 100]."""
    state.model = MagicMock()

    # top_k = 0 should fail validation (ge=1)
    res_zero = client.post("/recommend", json={"video_id": "test", "top_k": 0})
    assert res_zero.status_code == 422

    # top_k = 200 should fail validation (le=100)
    res_large = client.post("/recommend", json={"video_id": "test", "top_k": 200})
    assert res_large.status_code == 422


def test_recommend_insufficient_index(client, in_memory_db):
    """Test recommendation returns 400 when less than 2 videos exist in index."""
    state.model = MagicMock()
    state.is_trained = True

    repo = VideoRepository(in_memory_db)
    repo.create(
        video_id="only_one_vid",
        title="Sole Video",
        embedding=[0.1] * 512
    )

    response = client.post("/recommend", json={"video_id": "only_one_vid", "top_k": 5})
    assert response.status_code == 400
    assert "Need at least 2 videos" in response.json()["detail"]


def test_recommend_success_flow(client, in_memory_db):
    """Test recommendation returns similar videos successfully."""
    state.model = MagicMock()
    state.is_trained = True

    repo = VideoRepository(in_memory_db)
    v1 = repo.create(
        video_id="query_vid",
        title="Query Video",
        embedding=[1.0] + [0.0] * 511
    )
    v2 = repo.create(
        video_id="candidate_1",
        title="Candidate Video 1",
        embedding=[0.9] + [0.1] * 511
    )
    v3 = repo.create(
        video_id="candidate_2",
        title="Candidate Video 2",
        embedding=[0.8] + [0.2] * 511
    )

    response = client.post("/recommend", json={"video_id": "query_vid", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["query_video_id"] == "query_vid"
    assert len(data["recommendations"]) == 2
    # Verify recommendations contain video_id and score
    rec_ids = [r["video_id"] for r in data["recommendations"]]
    assert "candidate_1" in rec_ids
    assert "candidate_2" in rec_ids
