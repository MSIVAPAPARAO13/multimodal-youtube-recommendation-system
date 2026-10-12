"""
SpeechRecognizer: Optional speech recognition provider abstraction.

Responsibilities:
- Optional and independently configurable (AUDIO_SPEECH_RECOGNITION_ENABLED)
- Does not force heavyweight model downloads by default
- Decoupled from audio embedding computation (audio embeddings are purely acoustic)
- Protects against overwriting existing valid captions/transcripts
- Gracefully handles absent audio, unsupported languages, or missing models
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SpeechRecognitionResult:
    """Outcome of an optional speech-to-text recognition attempt."""
    text: Optional[str] = None
    is_available: bool = False
    language: Optional[str] = None
    provider: str = "none"
    confidence: Optional[float] = None
    error_message: Optional[str] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Diagnostic metadata dictionary."""
        return {
            "text": self.text,
            "is_available": self.is_available,
            "language": self.language,
            "provider": self.provider,
            "confidence": self.confidence,
            "error_message": self.error_message,
            "diagnostics": self.diagnostics,
        }


class BaseSpeechRecognizer(ABC):
    """Abstract interface for speech recognition providers."""

    @abstractmethod
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> SpeechRecognitionResult:
        """Transcribe an audio waveform or audio file to text."""
        pass


class OptionalSpeechRecognizer(BaseSpeechRecognizer):
    """
    Optional, opt-in speech recognition provider.

    By default, speech recognition is disabled to avoid heavy external downloads
    and long inference latency. Audio embeddings remain purely acoustic.
    """

    def __init__(
        self,
        enabled: bool = False,
        provider_name: str = "optional_asr",
        transcribe_fn: Optional[Callable[..., SpeechRecognitionResult]] = None,
    ):
        self.enabled = bool(enabled)
        self.provider_name = provider_name
        self._transcribe_fn = transcribe_fn

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> SpeechRecognitionResult:
        """
        Transcribe audio if enabled. Otherwise returns a graceful disabled result.
        """
        if not self.enabled:
            return SpeechRecognitionResult(
                text=None,
                is_available=False,
                language=language,
                provider=self.provider_name,
                error_message="Speech recognition is disabled by configuration.",
                diagnostics={"status": "disabled"},
            )

        if self._transcribe_fn is not None:
            try:
                return self._transcribe_fn(audio, sample_rate=sample_rate, language=language)
            except Exception as e:
                logger.warning(f"Custom ASR transcription failed: {e}")
                return SpeechRecognitionResult(
                    text=None,
                    is_available=False,
                    language=language,
                    provider=self.provider_name,
                    error_message=f"Transcription error: {str(e)}",
                    diagnostics={"status": "error"},
                )

        # Opt-in interface without third-party heavy backend configured
        return SpeechRecognitionResult(
            text=None,
            is_available=False,
            language=language,
            provider=self.provider_name,
            error_message="No speech recognition engine configured. Set transcribe_fn or backend provider.",
            diagnostics={"status": "unconfigured_backend"},
        )

    @staticmethod
    def should_transcribe(existing_transcript: Optional[str], force: bool = False) -> bool:
        """
        Determine whether speech recognition should be performed.
        Never overwrite an existing valid transcript unless explicitly forced.
        """
        if force:
            return True
        if existing_transcript and len(existing_transcript.strip()) > 0:
            return False
        return True
