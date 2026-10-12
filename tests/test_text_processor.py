"""
Tests for YouTubeTextInput and TextProcessor (TASK 12: Text Processor Tests).
"""

import pytest
from src.features.text_processor import (
    YouTubeTextInput,
    TextProcessor,
    normalize_unicode,
    normalize_whitespace,
    normalize_tags,
    clean_transcript,
    TextQualityDiagnostics,
    DEFAULT_MAX_TITLE_CHARS,
    DEFAULT_MAX_DESCRIPTION_CHARS,
    DEFAULT_MAX_TAGS_CHARS,
)


def test_title_only_input():
    """Verify title-only input is properly normalized and formatted."""
    processor = TextProcessor()
    raw = "   Building a Multimodal Recommendation System   "
    structured, diag, canonical = processor.process(raw)

    assert canonical.normalized_title == "Building a Multimodal Recommendation System"
    assert canonical.normalized_description == ""
    assert canonical.normalized_tags == []
    assert canonical.normalized_transcript == ""

    assert "TITLE:\nBuilding a Multimodal Recommendation System" in structured
    assert "DESCRIPTION:" not in structured
    assert "TAGS:" not in structured
    assert "TRANSCRIPT:" not in structured

    assert diag.transcript_available is False
    assert diag.description_available is False
    assert diag.tags_available is False
    assert diag.title_length > 0


def test_title_and_description():
    """Verify title + description structured section markers and length limits."""
    processor = TextProcessor()
    data = {
        "title": "PyTorch Tutorial",
        "description": "Learn tensors and autograd in PyTorch with detailed code examples."
    }
    structured, diag, canonical = processor.process(data)

    assert "TITLE:\nPyTorch Tutorial" in structured
    assert "DESCRIPTION:\nLearn tensors and autograd in PyTorch with detailed code examples." in structured
    assert "TAGS:" not in structured
    assert "TRANSCRIPT:" not in structured

    assert diag.description_available is True
    assert diag.tags_available is False
    assert diag.transcript_available is False


def test_title_and_tags():
    """Verify title + tags as list and comma-separated string."""
    processor = TextProcessor()

    # List of tags
    data_list = {
        "title": "Machine Learning Overview",
        "tags": ["AI", " Deep Learning ", "Neural Networks", "AI"]  # includes duplicates and spaces
    }
    structured_list, diag_list, canonical_list = processor.process(data_list)
    assert "TITLE:\nMachine Learning Overview" in structured_list
    assert "TAGS:\nAI, Deep Learning, Neural Networks" in structured_list
    assert canonical_list.normalized_tags == ["AI", "Deep Learning", "Neural Networks"]
    assert diag_list.tags_available is True
    assert diag_list.tags_count == 3

    # String of tags
    data_str = {
        "title": "Machine Learning Overview",
        "tags": "python, fast-api, postgresql"
    }
    structured_str, _, canonical_str = processor.process(data_str)
    assert "TAGS:\npython, fast-api, postgresql" in structured_str
    assert canonical_str.normalized_tags == ["python", "fast-api", "postgresql"]


def test_full_multimodal_text_input():
    """Verify full text input with title, description, tags, and transcript."""
    processor = TextProcessor()
    full_data = {
        "title": "End-to-End Multimodal Video AI",
        "description": "Comprehensive pipeline from frames and audio to vector search.",
        "tags": ["video", "audio", "transformers", "pgvector"],
        "transcript": "Hello and welcome to this multimodal AI architectural deep dive."
    }
    structured, diag, canonical = processor.process(full_data)

    assert "TITLE:\nEnd-to-End Multimodal Video AI" in structured
    assert "DESCRIPTION:\nComprehensive pipeline from frames and audio to vector search." in structured
    assert "TAGS:\nvideo, audio, transformers, pgvector" in structured
    assert "TRANSCRIPT:\nHello and welcome to this multimodal AI architectural deep dive." in structured

    assert diag.transcript_available is True
    assert diag.description_available is True
    assert diag.tags_available is True
    assert diag.transcript_used is True
    assert diag.transcript_length > 0


def test_whitespace_normalization():
    """Verify excessive spaces, tabs, and newlines are normalized while preserving punctuation."""
    raw = "Title with\t\ttabs   and    spaces.\n\n\n\nNew paragraph with ? and ! preserved."
    normalized = normalize_whitespace(raw)
    assert "\t" not in normalized
    assert "   " not in normalized
    assert "\n\n\n" not in normalized
    assert normalized == "Title with tabs and spaces.\n\nNew paragraph with ? and ! preserved."


def test_unicode_handling():
    """Verify NFKC normalization unifies composite characters and cleans malformed Unicode."""
    # Composite character vs canonical form
    composite_text = "Café \u00e9clair \u2122 and \uff21\uff22\uff23"
    cleaned = normalize_unicode(composite_text)
    assert "Café" in cleaned
    assert "ABC" in cleaned  # Fullwidth characters normalized to standard ASCII by NFKC


def test_youtube_text_input_from_any():
    """Verify YouTubeTextInput construction from string, dict, or instance."""
    # From string
    ti_str = YouTubeTextInput.from_any("A Test Title")
    assert ti_str.normalized_title == "A Test Title"

    # From dict
    ti_dict = YouTubeTextInput.from_any({"title": "Dict Title", "description": "Some description"})
    assert ti_dict.normalized_title == "Dict Title"
    assert ti_dict.normalized_description == "Some description"

    # From instance
    ti_copy = YouTubeTextInput.from_any(ti_dict)
    assert ti_copy is ti_dict

    # Conversion to dict
    as_dict = ti_dict.to_dict()
    assert as_dict["title"] == "Dict Title"
    assert as_dict["description"] == "Some description"
