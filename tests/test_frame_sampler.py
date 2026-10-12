"""
Tests for FrameSampler (Phase 6):
- Deterministic frame sampling
- Correct requested number of frames
- Short-video handling
- Missing-video handling
- Corrupt-video handling
- Partial frame decode recovery
- Diagnostics verification
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

from src.features.frame_sampler import FrameSampler, FrameSamplingResult


def create_synthetic_video(file_path: Path, num_frames: int = 16, width: int = 64, height: int = 64, fps: float = 30.0) -> Path:
    """Helper to generate a lightweight synthetic MP4 video file with distinct deterministic frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        # Create distinct solid color frames
        val = int((i * 255) / max(1, num_frames - 1))
        frame = np.full((height, width, 3), val, dtype=np.uint8)
        out.write(frame)
    out.release()
    return file_path


@pytest.fixture
def synthetic_video(tmp_path):
    """Fixture providing a 16-frame synthetic video."""
    video_path = tmp_path / "test_synthetic.mp4"
    return create_synthetic_video(video_path, num_frames=16)


@pytest.fixture
def short_video(tmp_path):
    """Fixture providing a 3-frame synthetic video."""
    video_path = tmp_path / "test_short.mp4"
    return create_synthetic_video(video_path, num_frames=3)


def test_deterministic_frame_sampling(synthetic_video):
    """Sampling same video twice produces identical frame indices and identical pixel values."""
    sampler = FrameSampler(default_num_frames=8)

    res1 = sampler.sample_frames(synthetic_video, num_frames=8)
    res2 = sampler.sample_frames(synthetic_video, num_frames=8)

    assert res1.success is True
    assert res2.success is True
    assert res1.frame_indices == res2.frame_indices
    assert len(res1.frames) == 8
    assert len(res2.frames) == 8

    for f1, f2 in zip(res1.frames, res2.frames):
        np.testing.assert_array_equal(f1, f2)


def test_correct_requested_frame_count(synthetic_video):
    """Sampler respects requested frame counts (e.g., 4, 8, 12)."""
    sampler = FrameSampler()

    for count in [4, 8, 12]:
        res = sampler.sample_frames(synthetic_video, num_frames=count)
        assert res.success is True
        assert res.requested_frames == count
        assert len(res.frames) == count
        assert res.decoded_frames == count


def test_short_video_handling(short_video):
    """Videos with fewer frames than requested (3 frames vs 8 requested) don't crash."""
    sampler = FrameSampler(default_num_frames=8)
    res = sampler.sample_frames(short_video, num_frames=8)

    assert res.success is True
    assert res.requested_frames == 8
    assert res.decoded_frames == 3
    assert res.total_frames == 3
    # Result frames padded to 8 for downstream tensor compatibility
    assert len(res.frames) == 8


def test_missing_video_handling(tmp_path):
    """Missing video file returns clean failed diagnostics without crashing."""
    sampler = FrameSampler(default_num_frames=8)
    missing_path = tmp_path / "non_existent_video.mp4"

    res = sampler.sample_frames(missing_path, num_frames=8)
    assert res.success is False
    assert len(res.frames) == 0
    assert "not found" in (res.error_message or "").lower()


def test_corrupt_video_handling(tmp_path):
    """Corrupted / non-video file returns clean failure without unhandled crash."""
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"This is not a valid mp4 video header or stream.")

    sampler = FrameSampler(default_num_frames=8)
    res = sampler.sample_frames(corrupt_file, num_frames=8)

    assert res.success is False
    assert len(res.frames) == 0
    assert res.error_message is not None


def test_empty_video_file_handling(tmp_path):
    """Empty 0-byte file returns clean failure."""
    empty_file = tmp_path / "empty.mp4"
    empty_file.touch()

    sampler = FrameSampler(default_num_frames=8)
    res = sampler.sample_frames(empty_file, num_frames=8)

    assert res.success is False
    assert "empty" in (res.error_message or "").lower()


def test_partial_frame_decode_recovery(synthetic_video):
    """If an intermediate frame read fails, sampler recovers and reports valid diagnostics."""
    sampler = FrameSampler(default_num_frames=8)

    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 16,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 64,
            cv2.CAP_PROP_FRAME_WIDTH: 64,
        }.get(prop, 0)

        # Let frame 3 fail to decode (ret=False)
        frame_mock = np.zeros((64, 64, 3), dtype=np.uint8)
        read_returns = [(True, frame_mock)] * 2 + [(False, None)] + [(True, frame_mock)] * 5
        mock_cap.read.side_effect = read_returns

        res = sampler.sample_frames(synthetic_video, num_frames=8)

        assert res.success is True
        assert len(res.frames) == 8
        # Decoded frames accurately records successful decodes
        assert res.decoded_frames == 7


def test_frame_sampler_diagnostics(synthetic_video):
    """Diagnostics dict contains all expected fields with proper types."""
    sampler = FrameSampler(default_num_frames=8)
    res = sampler.sample_frames(synthetic_video, num_frames=8)
    diag = res.to_dict()

    assert "requested_frames" in diag
    assert "decoded_frames" in diag
    assert "frame_indices" in diag
    assert "total_frames" in diag
    assert "fps" in diag
    assert "duration_seconds" in diag
    assert "video_path" in diag
    assert "success" in diag
    assert diag["success"] is True
