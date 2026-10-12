"""
Media validator for YouTube media acquisition.
Verifies file existence, size, container format, duration, and decodability.
"""

import logging
from pathlib import Path
from typing import Optional

import hashlib
import wave
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

from .exceptions import (
    MediaValidationError,
    MediaSizeExceededError,
    MediaDurationExceededError,
    MediaCorruptError,
)

logger = logging.getLogger(__name__)


class MediaValidator:
    """
    Validates downloaded video and audio files against system constraints.
    """

    def __init__(
        self,
        max_video_size_mb: int = 100,
        max_audio_size_mb: int = 50,
        max_duration_seconds: int = 600,
    ):
        self.max_video_size_mb = max_video_size_mb
        self.max_audio_size_mb = max_audio_size_mb
        self.max_duration_seconds = max_duration_seconds

    def validate_video_file(self, video_path: Path | str, video_id: Optional[str] = None) -> dict:
        """
        Validate downloaded video file: existence, size, duration, container, and decodability.
        Returns video metadata dictionary (duration, fps, frame_count, width, height, size_bytes).
        """
        path = Path(video_path)
        if not path.exists() or not path.is_file():
            raise MediaValidationError(f"Video file does not exist: {path}", video_id=video_id)

        file_size_bytes = path.stat().st_size
        if file_size_bytes == 0:
            raise MediaCorruptError(f"Video file is empty (0 bytes): {path}", video_id=video_id)

        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > self.max_video_size_mb:
            raise MediaSizeExceededError(
                f"Video size ({file_size_mb:.2f} MB) exceeds maximum allowed ({self.max_video_size_mb} MB)",
                video_id=video_id,
            )

        if not HAS_OPENCV or cv2 is None:
            logger.warning("OpenCV is not available; basic file size validation performed.")
            return {"size_bytes": file_size_bytes, "size_mb": file_size_mb}

        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise MediaCorruptError(f"Cannot open video file (corrupt or unsupported format): {path}", video_id=video_id)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if frame_count <= 0 or width <= 0 or height <= 0:
                raise MediaCorruptError(f"Video has invalid dimensions or zero frames: {path}", video_id=video_id)

            duration = (frame_count / fps) if fps > 0 else 0.0
            if duration > self.max_duration_seconds:
                raise MediaDurationExceededError(
                    f"Video duration ({duration:.1f}s) exceeds maximum allowed ({self.max_duration_seconds}s)",
                    video_id=video_id,
                )

            # Test read of first frame to ensure frame decoding works
            ret, frame = cap.read()
            if not ret or frame is None:
                raise MediaCorruptError(f"Failed to decode initial frame from video: {path}", video_id=video_id)

            return {
                "size_bytes": file_size_bytes,
                "size_mb": round(file_size_mb, 2),
                "duration_seconds": round(duration, 2),
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "width": width,
                "height": height,
            }
        finally:
            cap.release()

    @staticmethod
    def compute_sha256(file_path: Path | str) -> str:
        """Compute SHA-256 checksum of a local file safely in 64KB chunks."""
        p = Path(file_path)
        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def validate_audio_file(
        self,
        audio_path: Path | str,
        video_id: Optional[str] = None,
        expected_sample_rate: Optional[int] = 16000,
        expected_channels: Optional[int] = 1,
        check_silence: bool = False,
    ) -> dict:
        """
        Validate audio file existence, size, WAV header, sample rate, channels,
        duration, and optional nonempty signal.
        """
        path = Path(audio_path)
        if not path.exists() or not path.is_file():
            raise MediaValidationError(f"Audio file does not exist: {path}", video_id=video_id)

        file_size_bytes = path.stat().st_size
        if file_size_bytes == 0:
            raise MediaCorruptError(f"Audio file is empty (0 bytes): {path}", video_id=video_id)

        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > self.max_audio_size_mb:
            raise MediaSizeExceededError(
                f"Audio size ({file_size_mb:.2f} MB) exceeds maximum allowed ({self.max_audio_size_mb} MB)",
                video_id=video_id,
            )

        metadata = {
            "size_bytes": file_size_bytes,
            "size_mb": round(file_size_mb, 2),
        }

        # Validate WAV structure if file has .wav extension
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as wf:
                    channels = wf.getnchannels()
                    sample_rate = wf.getframerate()
                    n_frames = wf.getnframes()
                    duration = n_frames / sample_rate if sample_rate > 0 else 0.0

                    if n_frames == 0:
                        raise MediaCorruptError(f"Audio file contains zero audio frames: {path}", video_id=video_id)

                    if expected_channels is not None and channels != expected_channels:
                        raise MediaCorruptError(
                            f"Audio channels mismatch: got {channels}, expected {expected_channels}",
                            video_id=video_id,
                        )

                    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
                        raise MediaCorruptError(
                            f"Audio sample rate mismatch: got {sample_rate} Hz, expected {expected_sample_rate} Hz",
                            video_id=video_id,
                        )

                    if duration > self.max_duration_seconds:
                        raise MediaDurationExceededError(
                            f"Audio duration ({duration:.1f}s) exceeds maximum allowed ({self.max_duration_seconds}s)",
                            video_id=video_id,
                        )

                    if check_silence and n_frames > 0:
                        raw_bytes = wf.readframes(n_frames)
                        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        if not np.all(np.isfinite(samples)):
                            raise MediaCorruptError(f"Audio file contains non-finite samples: {path}", video_id=video_id)
                        rms = float(np.sqrt(np.mean(samples ** 2)))
                        if rms < 1e-5:
                            raise MediaCorruptError(
                                f"Audio file is silent (RMS energy={rms:.6f}): {path}",
                                video_id=video_id,
                            )
                        metadata["rms_energy"] = round(rms, 6)

                    metadata["channels"] = channels
                    metadata["sample_rate"] = sample_rate
                    metadata["duration_seconds"] = round(duration, 2)
                    metadata["frames"] = n_frames
            except wave.Error as e:
                raise MediaCorruptError(f"Corrupted or invalid WAV file: {e}", video_id=video_id) from e

        return metadata


    def validate_image_file(
        self,
        image_path: Path | str,
        min_width: int = 16,
        min_height: int = 16,
        video_id: Optional[str] = None,
    ) -> dict:
        """
        Validate image existence, size, and decodability.
        """
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise MediaValidationError(f"Image file does not exist: {path}", video_id=video_id)

        file_size_bytes = path.stat().st_size
        if file_size_bytes == 0:
            raise MediaCorruptError(f"Image file is empty (0 bytes): {path}", video_id=video_id)

        if not HAS_OPENCV or cv2 is None:
            return {"size_bytes": file_size_bytes}

        img = cv2.imread(str(path))
        if img is None:
            raise MediaCorruptError(f"Failed to decode image file (corrupted or unsupported format): {path}", video_id=video_id)

        h, w = img.shape[:2]
        if w < min_width or h < min_height:
            raise MediaCorruptError(f"Image dimensions ({w}x{h}) smaller than minimum ({min_width}x{min_height}): {path}", video_id=video_id)

        return {
            "size_bytes": file_size_bytes,
            "width": w,
            "height": h,
        }
