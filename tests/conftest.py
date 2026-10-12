import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure src is in sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from db.models import Base, Video, VideoIngestion, VideoAsset, VideoFeature, UserInteraction
from api.app import app, get_db, state


@pytest.fixture(autouse=True)
def mock_test_model():
    """Mock heavy model loading and initialization for all tests."""
    mock_model = MagicMock()
    mock_model.encode_video.return_value = torch.ones(1, 512)
    mock_model.get_similarity.return_value = torch.tensor([0.95])

    mock_service = MagicMock()
    mock_service.model = mock_model
    mock_service.is_checkpoint_loaded = True
    result = MagicMock()
    result.embedding = torch.ones(1, 512)
    result.embedding_list = [1.0] * 512
    result.video_embedding = torch.ones(1, 512)
    result.audio_embedding = torch.ones(1, 512)
    result.text_embedding = torch.ones(1, 512)
    result.modalities_present = ["video", "audio", "text"]
    result.device = "cpu"
    mock_service.encode_multimodal.return_value = result

    state.model = mock_model
    state.inference_service = mock_service
    state.device = torch.device("cpu")
    state.is_trained = True

    with patch("models.two_tower.TwoTowerModel", return_value=mock_model), \
         patch("src.models.two_tower.TwoTowerModel", return_value=mock_model), \
         patch("src.services.multimodal_inference.MultimodalInferenceService", return_value=mock_service), \
         patch("api.app.init_database"):
        yield mock_model


@pytest.fixture(scope="session")
def test_engine():
    """Shared in-memory SQLite engine with StaticPool for thread-safety."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def in_memory_db(test_engine):
    """Database session fixture with table isolation between tests."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    session.query(UserInteraction).delete()
    session.query(VideoFeature).delete()
    session.query(VideoAsset).delete()
    session.query(VideoIngestion).delete()
    session.query(Video).delete()
    session.commit()
    try:
        yield session
    finally:
        session.query(UserInteraction).delete()
        session.query(VideoFeature).delete()
        session.query(VideoAsset).delete()
        session.query(VideoIngestion).delete()
        session.query(Video).delete()
        session.commit()
        session.close()


@pytest.fixture
def client(test_engine):
    """FastAPI TestClient with overridden database dependency."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client
    app.dependency_overrides.clear()
