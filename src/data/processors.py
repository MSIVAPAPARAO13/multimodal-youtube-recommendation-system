"""
Media Processors: Extract frames and audio from video files

These utilities handle the raw video → tensor conversion needed for our encoders.
"""

import logging
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    librosa = None
    HAS_LIBROSA = False

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VideoProcessor:
    """
    Extracts frames from video files with standard ImageNet preprocessing.

    Args:
        num_frames: How many frames to sample from each video
        frame_size: Resize frames to (height, width)
        normalize: Whether to normalize pixel values to ImageNet distribution
    """

    def __init__(
        self,
        num_frames: int = 8,
        frame_size: tuple[int, int] = (224, 224),
        normalize: bool = True
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.normalize = normalize
        if self.normalize:
            self.transform = transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        else:
            self.transform = None

    def extract_frames(self, video_path: str) -> torch.Tensor:
        """
        Extract evenly-spaced frames from a video.

        Args:
            video_path: Path to video file (mp4, avi, etc.)

        Returns:
            frames: (num_frames, 3, H, W) tensor
        """
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV (cv2) is required for video extraction. Please install opencv-python.")

        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_path_str = str(video_path)

        # Open video file
        cap = cv2.VideoCapture(video_path_str)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path_str}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no readable frames: {video_path_str}")

        # Calculate which frames to sample (evenly spaced)
        if total_frames >= self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Video shorter than num_frames: take all and pad
            frame_indices = np.arange(total_frames)

        frames = []

        for idx in frame_indices:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

            # Read frame
            ret, frame = cap.read()

            if not ret or frame is None:
                # If frame read fails, use black frame
                frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            else:
                # BGR -> RGB (OpenCV loads as BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to target size
            frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
            frames.append(frame)

        cap.release()

        # Pad if video was too short
        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())

        # Stack into numpy array: (num_frames, H, W, 3)
        frames = np.stack(frames)

        # Convert to tensor and rearrange: (num_frames, 3, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

        # Normalize with ImageNet mean and std
        if self.normalize:
            frames = frames / 255.0
            frames = self.transform(frames)

        return frames


class AudioProcessor:
    """
    Extracts audio waveform from video files.

    Args:
        sample_rate: Target sample rate (16000 for Wav2Vec2)
        duration: How many seconds of audio to extract
        mono: Convert to mono (single channel)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 5.0,
        mono: bool = True
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
        self.target_length = int(sample_rate * duration)
        self.last_extraction_success = False

    def extract_audio(self, video_path: str) -> torch.Tensor:
        """
        Extract audio waveform from a video file.

        Args:
            video_path: Path to video file

        Returns:
            waveform: (num_samples,) tensor, e.g., (80000,) for 5 seconds.
                      Returns an explicit zero tensor and flags last_extraction_success=False
                      if extraction fails or is unavailable.
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            logger.warning("[AUDIO_EXTRACTION_UNAVAILABLE] Video file not found: %s", video_path)
            self.last_extraction_success = False
            return torch.zeros(self.target_length, dtype=torch.float32)

        if not HAS_LIBROSA:
            logger.warning("[AUDIO_EXTRACTION_UNAVAILABLE] librosa is not available. Returning silent placeholder.")
            self.last_extraction_success = False
            return torch.zeros(self.target_length, dtype=torch.float32)

        video_path_str = str(video_path)

        try:
            waveform, _ = librosa.load(
                video_path_str,
                sr=self.sample_rate,
                mono=self.mono,
                duration=self.duration
            )
            # Replace NaNs or Infs if any
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
            # Check if extracted waveform is purely silent
            if np.max(np.abs(waveform)) < 1e-5:
                logger.info("[AUDIO_EXTRACTION_SILENT] Extracted waveform from %s is purely silent.", video_path_str)
                self.last_extraction_success = False
            else:
                self.last_extraction_success = True
        except Exception as e:
            logger.warning("[AUDIO_EXTRACTION_UNAVAILABLE] Could not extract audio from %s: %s", video_path_str, e)
            waveform = np.zeros(self.target_length, dtype=np.float32)
            self.last_extraction_success = False

        # Ensure correct length (pad or truncate)
        if len(waveform) < self.target_length:
            padding = self.target_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode="constant")
        elif len(waveform) > self.target_length:
            waveform = waveform[:self.target_length]

        waveform = torch.from_numpy(waveform).float()
        return waveform


# Test code
if __name__ == "__main__":
    print("Testing processors...")

    video_proc = VideoProcessor(num_frames=8, frame_size=(224, 224))
    audio_proc = AudioProcessor(sample_rate=16000, duration=5.0)

    print(f"VideoProcessor: extracts {video_proc.num_frames} frames at {video_proc.frame_size}")
    print(f"AudioProcessor: extracts {audio_proc.duration}s at {audio_proc.sample_rate}Hz")

    # To test with a real video:
    # frames = video_proc.extract_frames("path/to/video.mp4")
    # print(f"Frames shape: {frames.shape}")  # (8, 3, 224, 224)
    #
    # audio = audio_proc.extract_audio("path/to/video.mp4")
    # print(f"Audio shape: {audio.shape}")  # (80000,)

    print("[OK] Processors ready!")
