from api.app import state


def test_health_endpoint(client):
    """Test health endpoint returns expected schema and status ok."""
    state.device = "cpu"
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "model_trained" in data
    assert "environment" in data
    assert "index_size" in data
    assert isinstance(data["index_size"], int)
    assert data["storage"] == "postgresql"


def test_root_endpoint(client):
    """Test root endpoint returns version and name."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
