"""
FrameSampler: Deterministic Video Frame Extraction.

Extracts evenly-spaced frames from local MP4 video files with:
- Deterministic frame selection
- Safe sequential/index-based seeking (avoids loading entire video into memory)
- Graceful handling of short, corrupted, or missing videos
- Resilient per-frame decode failure recovery
- Detailed diagnostics
"""

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

logger = logging.getLogger(__name__)


@dataclass
class FrameSamplingResult:
    """Diagnostic and payload result from FrameSampler."""
    frames: list[np.ndarray] = field(default_factory=list)
    requested_frames: int = 0
    decoded_frames: int = 0
    frame_indices: list[int] = field(default_factory=list)
    total_frames: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0
    video_path: str = ""
    success: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostic metadata to dict without raw numpy frames."""
        return {
            "requested_frames": self.requested_frames,
            "decoded_frames": self.decoded_frames,
            "frame_indices": self.frame_indices,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "video_path": self.video_path,
            "success": self.success,
            "error_message": self.error_message,
        }


class FrameSampler:
    """
    Deterministically samples video frames from local media files.
    """

    def __init__(self, default_num_frames: int = 8):
        self.default_num_frames = max(1, int(default_num_frames))

    def sample_frames(
        self,
        video_path: Union[str, Path],
        num_frames: Optional[int] = None,
    ) -> FrameSamplingResult:
        """
        Deterministically sample frames from video_path.

        Args:
            video_path: Absolute or relative local path to video file
            num_frames: Target frame count (defaults to self.default_num_frames)

        Returns:
            FrameSamplingResult with decoded RGB numpy arrays and diagnostics.
        """
        target_frames = num_frames if num_frames is not None and num_frames > 0 else self.default_num_frames
        path_str = str(video_path)

        if not HAS_OPENCV:
            logger.error("OpenCV (cv2) is not installed; cannot sample frames.")
            return FrameSamplingResult(
                requested_frames=target_frames,
                video_path=path_str,
                success=False,
                error_message="OpenCV is required for frame sampling.",
            )

        video_path_obj = Path(video_path)
        if not video_path_obj.exists() or not video_path_obj.is_file():
            logger.warning("Video file not found or is not a file: %s", path_str)
            return FrameSamplingResult(
                requested_frames=target_frames,
                video_path=path_str,
                success=False,
                error_message=f"Video file not found: {path_str}",
            )

        if video_path_obj.stat().st_size == 0:
            logger.warning("Video file is empty (0 bytes): %s", path_str)
            return FrameSamplingResult(
                requested_frames=target_frames,
                video_path=path_str,
                success=False,
                error_message=f"Video file is empty (0 bytes): {path_str}",
            )

        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            logger.warning("Failed to open video with cv2: %s", path_str)
            return FrameSamplingResult(
                requested_frames=target_frames,
                video_path=path_str,
                success=False,
                error_message=f"Cannot open video file: {path_str}",
            )

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0 or np.isnan(fps):
                fps = 30.0

            # Fallback if CAP_PROP_FRAME_COUNT returned 0 or negative
            if total_frames <= 0:
                logger.warning("Video reported <= 0 frame count: %s. Attempting probe.", path_str)
                # Quick probe of first frame
                ret, probe_frame = cap.read()
                if not ret or probe_frame is None:
                    return FrameSamplingResult(
                        requested_frames=target_frames,
                        total_frames=0,
                        fps=fps,
                        video_path=path_str,
                        success=False,
                        error_message="Video has no readable frames.",
                    )
                # At least 1 frame readable
                total_frames = 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            duration_seconds = float(total_frames / fps) if fps > 0 else 0.0

            # Compute deterministic sampling indices
            if total_frames >= target_frames:
                frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int).tolist()
            else:
                frame_indices = list(range(total_frames))

            decoded_frames: list[np.ndarray] = []
            successful_decodes = 0
            last_valid_frame: Optional[np.ndarray] = None

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()

                if ret and frame is not None:
                    # OpenCV loads as BGR; convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    decoded_frames.append(frame_rgb)
                    last_valid_frame = frame_rgb
                    successful_decodes += 1
                else:
                    logger.warning("Failed to decode frame index %d in video %s; using recovery strategy.", idx, path_str)
                    if last_valid_frame is not None:
                        decoded_frames.append(last_valid_frame.copy())
                    else:
                        # Fallback default black frame (will be sized by processor or default 224x224)
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 224
                        black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                        decoded_frames.append(black_frame)

            if successful_decodes == 0:
                return FrameSamplingResult(
                    frames=[],
                    requested_frames=target_frames,
                    decoded_frames=0,
                    frame_indices=frame_indices,
                    total_frames=total_frames,
                    fps=fps,
                    duration_seconds=duration_seconds,
                    video_path=path_str,
                    success=False,
                    error_message="Zero frames could be successfully decoded from video.",
                )

            # Pad short videos to target_frames deterministically
            if len(decoded_frames) < target_frames and len(decoded_frames) > 0:
                last_frame = decoded_frames[-1]
                while len(decoded_frames) < target_frames:
                    decoded_frames.append(last_frame.copy())

            return FrameSamplingResult(
                frames=decoded_frames,
                requested_frames=target_frames,
                decoded_frames=successful_decodes,
                frame_indices=frame_indices,
                total_frames=total_frames,
                fps=fps,
                duration_seconds=duration_seconds,
                video_path=path_str,
                success=True,
                error_message=None,
            )

        except Exception as e:
            logger.error("Exception occurred while sampling frames from %s: %s", path_str, e)
            return FrameSamplingResult(
                requested_frames=target_frames,
                video_path=path_str,
                success=False,
                error_message=str(e),
            )
        finally:
            cap.release()

    def extract_and_save_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_frames: Optional[int] = None,
        video_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Deterministically extract frames from video_path and save them as JPEGs.
        Computes SHA-256 checksums, timestamps, and dimensions for each frame.

        Args:
            video_path: Path to local video file.
            output_dir: Directory where frame files will be saved.
            num_frames: Number of frames to extract (default self.default_num_frames).
            video_id: Optional video identifier for logging.

        Returns:
            List of frame metadata dictionaries.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sampling_result = self.sample_frames(video_path, num_frames=num_frames)
        if not sampling_result.success:
            logger.warning(
                "Frame extraction failed for video %s: %s",
                video_id or str(video_path),
                sampling_result.error_message
            )
            return []

        if not HAS_OPENCV or cv2 is None:
            return []

        frame_metadata_list: list[dict[str, Any]] = []

        for i, frame_rgb in enumerate(sampling_result.frames):
            frame_filename = f"frame_{i:03d}.jpg"
            frame_file = out_dir / frame_filename

            # OpenCV expects BGR for imwrite
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(frame_file), frame_bgr)
            if not success or not frame_file.exists():
                logger.warning("Failed to save frame image to %s", frame_file)
                continue

            file_size = frame_file.stat().st_size
            sha256 = hashlib.sha256()
            with open(frame_file, "rb") as f:
                while chunk := f.read(65536):
                    sha256.update(chunk)
            checksum = sha256.hexdigest()

            h, w = frame_rgb.shape[:2]
            frame_num = sampling_result.frame_indices[i] if i < len(sampling_result.frame_indices) else i
            timestamp = round(frame_num / sampling_result.fps, 3) if sampling_result.fps > 0 else 0.0

            frame_metadata_list.append({
                "frame_index": i,
                "frame_number": frame_num,
                "timestamp_seconds": timestamp,
                "filename": frame_filename,
                "storage_path": str(frame_file),
                "width": w,
                "height": h,
                "file_size_bytes": file_size,
                "checksum_sha256": checksum,
            })

        return frame_metadata_list
