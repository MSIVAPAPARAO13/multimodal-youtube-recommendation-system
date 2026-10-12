"""Automated regression tests for Streamlit application and cloud deployment readiness."""
import os
import re
from pathlib import Path
import pytest

import streamlit_app as sapp


def test_youtube_id_validation():
    """Verify strict 11-char YouTube ID validation to prevent broken watch links."""
    assert sapp.is_valid_youtube_id("dQw4w9WgXcQ") is True
    assert sapp.is_valid_youtube_id("10040") is False  # MicroLens local ID
    assert sapp.is_valid_youtube_id("local_10040") is False
    assert sapp.is_valid_youtube_id("") is False
    assert sapp.is_valid_youtube_id(None) is False
    assert sapp.is_valid_youtube_id("too_long_to_be_a_youtube_id_12345") is False


def test_video_watch_url():
    """Verify only valid YouTube IDs produce external watch links."""
    yt_vid = {"video_id": "dQw4w9WgXcQ", "youtube_video_id": "dQw4w9WgXcQ"}
    assert sapp.get_video_watch_url(yt_vid) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    microlens_vid = {"video_id": "10040", "youtube_video_id": None}
    assert sapp.get_video_watch_url(microlens_vid) is None


def test_load_catalog_fixtures():
    """Verify catalog fixtures load reliably in standalone demo mode."""
    items = sapp.load_catalog_fixtures()
    assert isinstance(items, list)
    assert len(items) > 0
    first = items[0]
    assert "video_id" in first
    assert "title" in first


def test_no_noisy_emojis_or_external_logos():
    """Verify streamlit_app.py has no tacky emoji clutter or external image hotlinks."""
    app_text = Path("streamlit_app.py").read_text(encoding="utf-8")

    # Disallowed emoji symbols
    noisy_emojis = ["🚀", "🎯", "🧠", "⚡", "🔬", "💡", "▶️", "📺", "🏷️", "⏱️", "🆔", "🔍", "👤", "📹", "📊", "🟢", "ℹ️"]
    for emoji in noisy_emojis:
        assert emoji not in app_text, f"Unwanted emoji '{emoji}' found in streamlit_app.py"

    # Disallowed external logo host
    assert "upload.wikimedia.org" not in app_text, "External Wikimedia logo found in streamlit_app.py"


def test_deployment_config_files():
    """Verify Streamlit configuration files exist and have correct settings."""
    config_file = Path(".streamlit/config.toml")
    assert config_file.exists(), ".streamlit/config.toml must exist"
    content = config_file.read_text(encoding="utf-8")
    assert "maxUploadSize = 100" in content

    packages_file = Path("packages.txt")
    assert packages_file.exists(), "packages.txt must exist for Streamlit Cloud OpenCV"
    pkgs = packages_file.read_text(encoding="utf-8")
    assert "libgl1" in pkgs
    assert "libglib2.0-0" in pkgs
