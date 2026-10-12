"""
Tests for Transcript Text Processing and Embedding (TASK 12: Transcript Text Tests).
"""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from src.features.text_processor import clean_transcript, YouTubeTextInput, TextProcessor
from src.features.text_encoder import TextEncoder, create_text_encoder
from src.services.multimodal_inference import TextEmbeddingService, TextEmbeddingResult


def test_missing_transcript():
    """Verify missing transcript (None) produces empty string without failing."""
    cleaned = clean_transcript(None)
    assert cleaned == ""

    item = YouTubeTextInput(title="Video Without Transcript", transcript=None)
    assert item.normalized_transcript == ""


def test_empty_transcript():
    """Verify empty or purely whitespace transcript produces empty string."""
    assert clean_transcript("") == ""
    assert clean_transcript("   \n\t  ") == ""


def test_repeated_captions_cleanup():
    """Verify consecutive duplicate lines and rolling subtitle prefixes are removed."""
    raw_subtitles = (
        "Welcome back to the channel.\n"
        "Welcome back to the channel.\n"  # Exact duplicate
        "Today we are going to learn\n"
        "Today we are going to learn Python programming\n"  # Rolling prefix
        "Don't forget to like and subscribe.\n"
        "Don't forget to like and subscribe."  # Exact duplicate
    )
    cleaned = clean_transcript(raw_subtitles)

    # Should not contain repetitive phrasing
    assert cleaned.count("Welcome back to the channel.") == 1
    assert "Today we are going to learn Python programming" in cleaned
    assert cleaned.count("Today we are going to learn") == 1
    assert cleaned.count("Don't forget to like and subscribe.") == 1


def test_timestamp_removal_srt_vtt_and_brackets():
    """Verify SRT/VTT timing cues and bracketed timestamps are stripped."""
    raw = (
        "00:00:01.000 --> 00:00:04.500\n"
        "[01:23] In this chapter we introduce neural networks.\n"
        "(02:45) Skip connections solve vanishing gradients.\n"
        "00:03:10.250 --> 00:03:15.000\n"
        "Thank you for watching."
    )
    cleaned = clean_transcript(raw)

    assert "-->" not in cleaned
    assert "[01:23]" not in cleaned
    assert "(02:45)" not in cleaned
    assert "00:00:01" not in cleaned
    assert "In this chapter we introduce neural networks." in cleaned
    assert "Skip connections solve vanishing gradients." in cleaned
    assert "Thank you for watching." in cleaned


def test_sound_effects_removal():
    """Verify audio cues like [Music], [Applause], (laughter) are stripped."""
    raw = (
        "[Music] Welcome to the conference. [Applause]\n"
        "We are excited to be here. (Laughter)\n"
        "[Cheering] Let's begin the keynote."
    )
    cleaned = clean_transcript(raw)

    assert "[Music]" not in cleaned
    assert "[Applause]" not in cleaned
    assert "(Laughter)" not in cleaned
    assert "[Cheering]" not in cleaned
    assert "Welcome to the conference." in cleaned
    assert "We are excited to be here." in cleaned
    assert "Let's begin the keynote." in cleaned


def test_long_transcript_truncation_at_word_boundary():
    """Verify extremely long transcript is truncated cleanly at word boundary."""
    long_raw = "word " * 5000  # 25,000 chars
    cleaned = clean_transcript(long_raw, max_chars=1000)

    assert len(cleaned) <= 1000
    assert not cleaned.endswith("wo")  # Should not cut word in half
    assert cleaned.endswith("word")


def test_bert_embedding_shape_and_determinism_mocked():
    """Verify TextEncoder produces (1, 512) tensor and is deterministic across runs."""
    with patch("transformers.BertModel.from_pretrained") as mock_bert, \
         patch("transformers.BertTokenizer.from_pretrained") as mock_tok:

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.cls_token_id = 101
        mock_tokenizer.sep_token_id = 102
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 16, dtype=torch.long),
            "attention_mask": torch.ones(1, 16, dtype=torch.long)
        }
        mock_tok.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        dummy_state = torch.ones(1, 16, 768) * 0.5
        mock_out = MagicMock()
        mock_out.last_hidden_state = dummy_state
        mock_model.return_value = mock_out
        mock_bert.return_value = mock_model

        encoder = TextEncoder(
            model_name="bert-base-uncased",
            feature_dim=512,
            freeze_backbone=True,
            pretrained=True
        )
        encoder.eval()

        meta = {
            "title": "Deterministic Test Video",
            "description": "Testing reproducibility of text pipeline.",
            "tags": ["test", "deterministic"],
            "transcript": "Checking numeric consistency."
        }

        with torch.no_grad():
            emb1 = encoder(texts=[meta])
            emb2 = encoder(texts=[meta])

        assert emb1.shape == (1, 512)
        assert emb2.shape == (1, 512)
        assert torch.allclose(emb1, emb2, atol=1e-6)
        assert not torch.isnan(emb1).any()
        assert not torch.isinf(emb1).any()


def test_nan_inf_protection_in_text_service():
    """Verify TextEmbeddingService sanitizes any NaNs/Infs produced during inference."""
    class FaultyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dim = 512
        def forward(self, texts=None, input_ids=None, attention_mask=None):
            t = torch.randn(1, 512)
            t[0, 0] = float("nan")
            t[0, 1] = float("inf")
            return t

    faulty_enc = FaultyEncoder()
    svc = TextEmbeddingService(text_encoder=faulty_enc)

    res = svc.encode("Sample Title")
    emb = res.embedding

    assert emb.shape == (1, 512)
    assert not torch.isnan(emb).any()
    assert not torch.isinf(emb).any()
    assert emb[0, 0].item() == 0.0
    assert emb[0, 1].item() == 0.0


def test_transcript_diagnostics_in_embedding_service():
    """Verify TextEmbeddingService tracks diagnostics correctly and unpacks as tuple."""
    class MockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dim = 512
        def forward(self, texts=None, input_ids=None, attention_mask=None):
            return torch.randn(len(texts) if texts else 1, 512)

    svc = TextEmbeddingService(text_encoder=MockEncoder())

    payload = {
        "title": "Machine Learning with Transcripts",
        "description": "Explaining deep learning concepts.",
        "tags": ["ml", "ai"],
        "transcript": "Hello and welcome to this lecture."
    }

    # Test object & dict-like access
    result = svc.encode(payload)
    assert isinstance(result, TextEmbeddingResult)
    assert result.embedding.shape == (1, 512)
    assert result.transcript_used is True
    assert result.transcript_available is True
    assert result.description_available is True
    assert result.tags_available is True
    assert result["transcript_used"] is True
    assert result["chunk_count"] >= 1

    # Test backward-compatible tuple unpacking: emb, ok = svc.encode(...)
    emb, ok = svc.encode(payload)
    assert ok is True
    assert emb.shape == (1, 512)


def test_experimental_encoder_factory():
    """Verify create_text_encoder instantiates BERT and SentenceTransformer without downloading."""
    # Test BERT factory
    with patch("transformers.BertModel.from_pretrained"), \
         patch("transformers.BertTokenizer.from_pretrained"):
        bert_enc = create_text_encoder("bert", pretrained=True)
        assert isinstance(bert_enc, TextEncoder)

    # Test sentence-transformers factory with pretrained=False (no download)
    st_enc = create_text_encoder("sentence-transformers", pretrained=False)
    assert st_enc.feature_dim == 512
    assert st_enc._is_loaded is False

    # Calling unloaded experimental encoder raises descriptive RuntimeError
    with pytest.raises(RuntimeError, match="has not been loaded"):
        st_enc(["Sample text"])
