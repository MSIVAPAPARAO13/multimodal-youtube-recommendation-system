from unittest.mock import MagicMock, patch
import torch
import torch.nn.functional as F
from models.two_tower import TwoTowerModel, VideoTower


def test_two_tower_similarity_computation():
    """Test get_similarity and find_similar methods."""
    model = TwoTowerModel.__new__(TwoTowerModel)
    model.temperature = 0.07

    # Identical normalized vectors should have similarity 1.0
    v1 = F.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
    sim = model.get_similarity(v1, v1)
    assert torch.isclose(sim, torch.tensor([1.0]), atol=1e-5)

    # Orthogonal vectors should have similarity 0.0
    v2 = F.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=-1)
    sim_ortho = model.get_similarity(v1, v2)
    assert torch.isclose(sim_ortho, torch.tensor([0.0]), atol=1e-5)

    # Opposite vectors should have similarity -1.0
    v3 = F.normalize(torch.tensor([[-1.0, 0.0, 0.0]]), dim=-1)
    sim_neg = model.get_similarity(v1, v3)
    assert torch.isclose(sim_neg, torch.tensor([-1.0]), atol=1e-5)


def test_two_tower_find_similar():
    """Test top-k nearest neighbor search."""
    model = TwoTowerModel.__new__(TwoTowerModel)
    query = F.normalize(torch.randn(512), dim=0)
    candidates = F.normalize(torch.randn(50, 512), dim=1)

    # Force candidate index 7 to be identical to query
    candidates[7] = query.clone()

    indices, scores = model.find_similar(query, candidates, top_k=5)
    assert len(indices) == 5
    assert len(scores) == 5
    # The top candidate should be index 7 with score ~1.0
    assert indices[0].item() == 7
    assert torch.isclose(scores[0], torch.tensor(1.0), atol=1e-5)


def test_video_tower_forward_shape():
    """Test VideoTower forward pass with mocked modality encoders."""
    with patch("models.two_tower.VideoEncoder") as mock_ve, \
         patch("models.two_tower.AudioEncoder") as mock_ae, \
         patch("models.two_tower.TextEncoder") as mock_te:

        mock_video_instance = MagicMock()
        mock_video_instance.return_value = torch.randn(2, 512)
        mock_ve.return_value = mock_video_instance

        mock_audio_instance = MagicMock()
        mock_audio_instance.return_value = torch.randn(2, 512)
        mock_ae.return_value = mock_audio_instance

        mock_text_instance = MagicMock()
        mock_text_instance.return_value = torch.randn(2, 512)
        mock_te.return_value = mock_text_instance

        tower = VideoTower(feature_dim=512, freeze_encoders=True, fusion_type="gated", pretrained=False)

        dummy_frames = torch.randn(2, 8, 3, 224, 224)
        dummy_audio = torch.randn(2, 80000)
        dummy_titles = ["video 1", "video 2"]

        embedding = tower(dummy_frames, dummy_audio, dummy_titles)

        assert embedding.shape == (2, 512)
        # Output should be L2-normalized
        norms = torch.norm(embedding, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)
