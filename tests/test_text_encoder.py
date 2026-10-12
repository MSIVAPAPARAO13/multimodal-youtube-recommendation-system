from unittest.mock import MagicMock, patch
import torch
from features.text_encoder import TextEncoder, format_text_input


def test_format_text_input_abstraction():
    """Test text formatting abstraction for string and structured inputs."""
    # Plain string (backward compatibility)
    assert format_text_input("How to cook pasta") == "How to cook pasta"

    # Structured metadata
    meta = {
        "title": "Machine Learning Tutorial",
        "description": "Learn deep learning step by step",
        "tags": ["AI", "Python", "DeepLearning"],
        "transcript": "Welcome to this tutorial on neural networks"
    }
    formatted = format_text_input(meta)
    assert "Machine Learning Tutorial" in formatted
    assert "Learn deep learning step by step" in formatted
    assert "Tags: AI, Python, DeepLearning" in formatted
    assert "Transcript: Welcome to this tutorial on neural networks" in formatted


def test_text_encoder_shape():
    """Test TextEncoder forward output shape with mocked BERT backbone."""
    with patch("transformers.BertModel.from_pretrained") as mock_bert, \
         patch("transformers.BertTokenizer.from_pretrained") as mock_tok:

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(2, 16, dtype=torch.long),
            "attention_mask": torch.ones(2, 16, dtype=torch.long)
        }
        mock_tok.return_value = mock_tokenizer

        # Mock BERT model output
        mock_model = MagicMock()
        # [CLS] token shape (batch_size=2, seq_len=16, hidden_dim=768)
        dummy_hidden_state = torch.randn(2, 16, 768)
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = dummy_hidden_state
        mock_model.return_value = mock_outputs
        mock_model.device = torch.device("cpu")
        mock_bert.return_value = mock_model

        encoder = TextEncoder(
            model_name="bert-base-uncased",
            feature_dim=512,
            aggregation="cls",
            freeze_backbone=True
        )

        texts = ["First test video", "Second test video"]
        output = encoder(texts=texts)

        # Output shape must be (batch_size=2, feature_dim=512)
        assert output.shape == (2, 512)
        assert isinstance(output, torch.Tensor)
