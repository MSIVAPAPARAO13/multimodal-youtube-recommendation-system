"""
AudioProcessor: Reusable Audio Preprocessing & Acoustic Intelligence.

Responsibilities:
- Input validation (file existence, non-empty, supported formats)
- Stereo-to-mono deterministic conversion
- Strict polyphase resampling to 16,000 Hz for Wav2Vec2
- Floating-point normalization [-1.0, 1.0]
- Defensive NaN/Inf cleansing
- Silence and clipping detection
- Duration limit enforcement
- Rich acoustic diagnostics
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import torch

logger = logging.getLogger(__name__)


@dataclass
class AudioProcessingResult:
    """Structured result and diagnostics from AudioProcessor."""
    waveform: torch.Tensor
    sample_rate: int = 16000
    duration_seconds: float = 0.0
    original_sample_rate: int = 16000
    original_channels: int = 1
    is_valid: bool = False
    is_silent: bool = False
    is_clipped: bool = False
    is_truncated: bool = False
    rms_energy: float = 0.0
    silence_ratio: float = 0.0
    error_message: Optional[str] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_waveform: bool = False) -> dict[str, Any]:
        """Convert diagnostic metadata to dict."""
        res = {
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 2),
            "original_sample_rate": self.original_sample_rate,
            "original_channels": self.original_channels,
            "is_valid": self.is_valid,
            "is_silent": self.is_silent,
            "is_clipped": self.is_clipped,
            "is_truncated": self.is_truncated,
            "rms_energy": round(self.rms_energy, 6),
            "silence_ratio": round(self.silence_ratio, 4),
            "error_message": self.error_message,
            "diagnostics": self.diagnostics,
        }
        if include_waveform:
            res["waveform"] = self.waveform
        return res


class AudioProcessor:
    """
    Validates, normalizes, resamples, and computes acoustic signal features for audio waveforms.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        sample_rate: Optional[int] = None,
        max_duration_seconds: float = 600.0,
        silence_threshold: float = 1e-4,
        clipping_threshold: float = 0.99,
    ):
        sr = sample_rate if sample_rate is not None else target_sample_rate
        self.target_sample_rate = max(1, int(sr))
        self.max_duration_seconds = max(1.0, float(max_duration_seconds))
        self.silence_threshold = float(silence_threshold)
        self.clipping_threshold = float(clipping_threshold)
        self.max_samples = int(self.max_duration_seconds * self.target_sample_rate)

    def _load_audio_file(self, file_path: Path) -> Tuple[np.ndarray, int, int]:
        """
        Safely load WAV file into (waveform_array, sample_rate, num_channels).
        """
        path_str = str(file_path)
        try:
            sr, data = wavfile.read(path_str)
            if data.ndim == 1:
                return data, sr, 1
            elif data.ndim == 2:
                # Shape could be (samples, channels) or (channels, samples)
                if data.shape[0] < data.shape[1] and data.shape[0] <= 8:
                    data = data.T
                return data, sr, data.shape[1]
            else:
                raise ValueError(f"Unexpected audio data dimensions: {data.ndim}")
        except Exception as e_wav:
            # Fallback to standard library wave module for PCM WAV
            import wave
            try:
                with wave.open(path_str, "rb") as wf:
                    sr = wf.getframerate()
                    channels = wf.getnchannels()
                    n_frames = wf.getnframes()
                    raw_bytes = wf.readframes(n_frames)
                    sampwidth = wf.getsampwidth()

                    if sampwidth == 2:
                        dtype = np.int16
                    elif sampwidth == 4:
                        dtype = np.int32
                    elif sampwidth == 1:
                        dtype = np.uint8
                    else:
                        raise ValueError(f"Unsupported sample width: {sampwidth}")

                    data = np.frombuffer(raw_bytes, dtype=dtype)
                    if channels > 1:
                        data = data.reshape(-1, channels)
                    return data, sr, channels
            except Exception as e_std:
                logger.warning("Could not read WAV file %s via wave or wavfile: %s, %s", path_str, e_wav, e_std)
                raise ValueError(f"Failed to decode audio file: {e_wav}") from e_wav

    def process_audio(
        self,
        audio_input: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        sample_rate: Optional[int] = None,
    ) -> AudioProcessingResult:
        """
        Execute full validation, mono conversion, resampling, and acoustic feature computation.

        Args:
            audio_input: Path to local audio file, numpy array, or torch tensor.
            sample_rate: Sampling rate of input if passing raw array/tensor (defaults to target).

        Returns:
            AudioProcessingResult with normalized 16 kHz mono waveform and acoustic metrics.
        """
        if audio_input is None:
            return AudioProcessingResult(
                waveform=torch.zeros(0, dtype=torch.float32),
                sample_rate=self.target_sample_rate,
                is_valid=False,
                is_silent=True,
                error_message="No audio input provided.",
            )

        orig_channels = 1
        orig_sr = sample_rate or self.target_sample_rate
        raw_data: np.ndarray

        try:
            # 1. Handle file path
            if isinstance(audio_input, (str, Path)):
                file_path = Path(audio_input)
                if not file_path.exists() or not file_path.is_file():
                    return AudioProcessingResult(
                        waveform=torch.zeros(0, dtype=torch.float32),
                        sample_rate=self.target_sample_rate,
                        is_valid=False,
                        error_message=f"Audio file not found: {audio_input}",
                    )

                if file_path.stat().st_size == 0:
                    return AudioProcessingResult(
                        waveform=torch.zeros(0, dtype=torch.float32),
                        sample_rate=self.target_sample_rate,
                        is_valid=False,
                        is_silent=True,
                        error_message=f"Audio file is empty (0 bytes): {audio_input}",
                    )

                raw_data, orig_sr, orig_channels = self._load_audio_file(file_path)

            elif isinstance(audio_input, torch.Tensor):
                raw_data = audio_input.detach().cpu().numpy()
            elif isinstance(audio_input, np.ndarray):
                raw_data = audio_input
            else:
                return AudioProcessingResult(
                    waveform=torch.zeros(0, dtype=torch.float32),
                    sample_rate=self.target_sample_rate,
                    is_valid=False,
                    error_message=f"Unsupported audio input type: {type(audio_input)}",
                )

            # 2. Channel normalization (Stereo/Multi-channel -> Mono)
            if raw_data.ndim == 2:
                orig_channels = raw_data.shape[1] if raw_data.shape[0] > raw_data.shape[1] else raw_data.shape[0]
                if raw_data.shape[0] < raw_data.shape[1] and raw_data.shape[0] <= 8:
                    raw_data = raw_data.T
                # Deterministic mono average: (L + R) / 2
                mono_data = np.mean(raw_data, axis=-1)
            elif raw_data.ndim == 1:
                mono_data = raw_data
                orig_channels = 1
            else:
                return AudioProcessingResult(
                    waveform=torch.zeros(0, dtype=torch.float32),
                    sample_rate=self.target_sample_rate,
                    is_valid=False,
                    error_message=f"Unexpected waveform dimensions: {raw_data.ndim}",
                )

            # 3. Numeric type conversion to float32 [-1.0, 1.0]
            if np.issubdtype(mono_data.dtype, np.integer):
                if mono_data.dtype == np.int16:
                    waveform_float = mono_data.astype(np.float32) / 32768.0
                elif mono_data.dtype == np.int32:
                    waveform_float = mono_data.astype(np.float32) / 2147483648.0
                elif mono_data.dtype == np.uint8:
                    waveform_float = (mono_data.astype(np.float32) - 128.0) / 128.0
                else:
                    waveform_float = mono_data.astype(np.float32) / float(np.iinfo(mono_data.dtype).max)
            else:
                waveform_float = mono_data.astype(np.float32)

            # 4. Clean non-finite numbers and clip
            waveform_float = np.nan_to_num(waveform_float, nan=0.0, posinf=0.0, neginf=0.0)
            waveform_float = np.clip(waveform_float, -1.0, 1.0)

            if len(waveform_float) == 0:
                return AudioProcessingResult(
                    waveform=torch.zeros(0, dtype=torch.float32),
                    sample_rate=self.target_sample_rate,
                    is_valid=False,
                    is_silent=True,
                    error_message="Decoded audio waveform is empty (0 samples).",
                )

            # 5. Resampling to target_sample_rate if needed
            if orig_sr != self.target_sample_rate:
                waveform_float = self.resample(waveform_float, orig_sr, self.target_sample_rate)

            # 6. Enforce duration bound
            is_truncated = False
            if len(waveform_float) > self.max_samples:
                waveform_float = waveform_float[: self.max_samples]
                is_truncated = True

            # 7. Acoustic feature intelligence computation
            num_samples = len(waveform_float)
            duration_seconds = float(num_samples / self.target_sample_rate)
            rms = float(np.sqrt(np.mean(waveform_float ** 2))) if num_samples > 0 else 0.0
            silence_samples = np.sum(np.abs(waveform_float) < self.silence_threshold)
            silence_ratio = float(silence_samples / num_samples) if num_samples > 0 else 1.0
            is_silent = bool(rms < 1e-5 or np.all(waveform_float == 0))
            is_clipped = bool(np.mean(np.abs(waveform_float) >= self.clipping_threshold) > 0.005)

            diagnostics = {
                "num_samples": num_samples,
                "duration_seconds": round(duration_seconds, 2),
                "original_sample_rate": orig_sr,
                "target_sample_rate": self.target_sample_rate,
                "original_channels": orig_channels,
                "rms_energy": round(rms, 6),
                "silence_ratio": round(silence_ratio, 4),
                "is_silent": is_silent,
                "is_clipped": is_clipped,
                "is_truncated": is_truncated,
            }

            final_tensor = torch.from_numpy(waveform_float).float()

            return AudioProcessingResult(
                waveform=final_tensor,
                sample_rate=self.target_sample_rate,
                duration_seconds=duration_seconds,
                original_sample_rate=orig_sr,
                original_channels=orig_channels,
                is_valid=not is_silent,
                is_silent=is_silent,
                is_clipped=is_clipped,
                is_truncated=is_truncated,
                rms_energy=rms,
                silence_ratio=silence_ratio,
                error_message="Audio is purely silent." if is_silent else None,
                diagnostics=diagnostics,
            )

        except Exception as e:
            logger.error("Audio processing exception: %s", e)
            return AudioProcessingResult(
                waveform=torch.zeros(0, dtype=torch.float32),
                sample_rate=self.target_sample_rate,
                is_valid=False,
                error_message=str(e),
            )

    def process_file(self, file_path: Union[str, Path]) -> AudioProcessingResult:
        """Convenience method for processing audio from file path."""
        return self.process_audio(file_path)

    def process_waveform(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> AudioProcessingResult:
        """Convenience method for processing in-memory waveform tensor/array."""
        return self.process_audio(waveform, sample_rate=sample_rate)

    def extract_audio(self, video_or_audio_path: str) -> torch.Tensor:
        """
        Backward-compatible helper returning a 1D waveform tensor.
        """
        res = self.process_audio(video_or_audio_path)
        if res.is_valid and res.waveform.numel() > 0:
            return res.waveform
        return torch.zeros(int(self.target_sample_rate * 5.0), dtype=torch.float32)

    def resample(
        self,
        waveform: np.ndarray,
        orig_sr: int,
        target_sr: Optional[int] = None,
    ) -> np.ndarray:
        """
        Resample waveform to target sampling rate using polyphase filtering.
        """
        target = target_sr or self.target_sample_rate
        if orig_sr == target:
            return waveform.astype(np.float32)

        from math import gcd
        common = gcd(target, orig_sr)
        up = target // common
        down = orig_sr // common
        try:
            resampled = signal.resample_poly(waveform, up, down)
        except Exception as e_poly:
            logger.warning("Polyphase resampling failed (%s); falling back to Fourier resample.", e_poly)
            num_target_samples = int(round(len(waveform) * target / orig_sr))
            resampled = signal.resample(waveform, num_target_samples)

        waveform_float = resampled.astype(np.float32)
        waveform_float = np.nan_to_num(waveform_float, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(waveform_float, -1.0, 1.0)
