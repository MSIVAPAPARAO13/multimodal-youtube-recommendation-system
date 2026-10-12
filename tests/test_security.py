import pytest
from fastapi import HTTPException
from api.app import validate_media_path, state


def test_validate_media_path_traversal():
    """Test validate_media_path rejects path traversal attempts."""
    traversal_paths = [
        "../../etc/passwd",
        "../../../windows/system32/calc.exe",
        "/etc/shadow",
        "/root/secret.txt",
        "C:\\Windows\\cmd.exe"
    ]

    for bad_path in traversal_paths:
        with pytest.raises(HTTPException) as exc_info:
            validate_media_path(bad_path)
        assert exc_info.value.status_code == 400
        assert "Security violation" in exc_info.value.detail


def test_validate_media_path_none_or_empty():
    """Test validate_media_path safely returns None for empty input."""
    assert validate_media_path(None) is None
    assert validate_media_path("") is None


def test_index_endpoint_rejects_invalid_path(client):
    """Test POST /index rejects path traversal requests."""
    state.model = True  # Mock model loaded

    payload = {
        "video_id": "attack_01",
        "video_path": "../../etc/shadow",
        "title": "Hacked Video",
        "description": "Exploit test"
    }

    response = client.post("/index", json=payload)
    assert response.status_code == 400
    assert "Security violation" in response.json()["detail"]
