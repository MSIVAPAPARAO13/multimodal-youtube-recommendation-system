"""
Multimodal Inference Service

Coordinates video, audio, and text embedding extraction, fusion, and normalization
into a deterministic 512-dimensional vector for video recommendation.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from src.data.processors import VideoProcessor, AudioProcessor
from src.features.audio_processor import AudioProcessor as FeatureAudioProcessor, AudioProcessingResult
from src.features.audio_chunker import AudioChunker, AudioChunkResult
from src.features.speech_recognizer import OptionalSpeechRecognizer, SpeechRecognitionResult
from src.features.video_encoder import VideoEncoder
from src.features.audio_encoder import AudioEncoder
from src.features.text_encoder import TextEncoder, format_text_input
from src.features.frame_sampler import FrameSampler, FrameSamplingResult
from src.features.visual_processor import VisualProcessor
from src.features.visual_encoder import BaseVisualEncoder, ResNet50VisualEncoder, get_visual_encoder
from src.features.thumbnail_encoder import ThumbnailEncoder, ThumbnailEncodingResult
from src.models.fusion import GatedFusion
from src.models.two_tower import TwoTowerModel, VideoTower
from src.db.config import get_settings, Settings

logger = logging.getLogger(__name__)


@dataclass
class VisualEmbeddingResult:
    """
    Structured result returned by VisualEmbeddingService / VideoEmbeddingService.

    Supports attribute access, dict conversion, and backwards-compatible tuple unpacking:
    `emb, ok = svc.encode(...)`.
    """
    embedding: torch.Tensor
    is_available: bool
    embedding_dimension: int = 512
    video_frames_requested: int = 8
    video_frames_used: int = 0
    thumbnail_used: bool = False
    thumbnail_available: bool = False
    pooling_method: str = "mean"
    encoder_name: str = "resnet50"
    processing_time_ms: float = 0.0
    modality_status: str = "missing"
    video_embedding: Optional[torch.Tensor] = None
    thumbnail_embedding: Optional[torch.Tensor] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_tensor: bool = False) -> dict[str, Any]:
        """Convert visual metadata to dict."""
        res = {
            "is_available": self.is_available,
            "embedding_dimension": self.embedding_dimension,
            "video_frames_requested": self.video_frames_requested,
            "video_frames_used": self.video_frames_used,
            "thumbnail_used": self.thumbnail_used,
            "thumbnail_available": self.thumbnail_available,
            "pooling_method": self.pooling_method,
            "encoder_name": self.encoder_name,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "modality_status": self.modality_status,
            "diagnostics": self.diagnostics,
        }
        if include_tensor:
            res["embedding"] = self.embedding
        else:
            res["embedding"] = self.embedding[0].cpu().tolist() if self.embedding.numel() > 0 else []
        return res

    def __iter__(self):
        """Enable backward-compatible unpacking: emb, ok = video_service.encode(...)"""
        yield self.embedding
        yield self.is_available

    def __getitem__(self, key: Union[str, int]) -> Any:
        if key == 0 or key == "embedding":
            return self.embedding
        if key == 1 or key == "is_available":
            return self.is_available
        if key == "video_frames_requested":
            return self.video_frames_requested
        if key == "video_frames_used":
            return self.video_frames_used
        if key == "thumbnail_used":
            return self.thumbnail_used
        if key == "thumbnail_available":
            return self.thumbnail_available
        if key == "pooling_method":
            return self.pooling_method
        if key == "encoder_name":
            return self.encoder_name
        if key == "processing_time_ms":
            return self.processing_time_ms
        if key == "modality_status":
            return self.modality_status
        if key == "diagnostics":
            return self.diagnostics
        raise KeyError(key)


@dataclass
class MultimodalInferenceResult:
    """Structured result returned by MultimodalInferenceService."""
    embedding: torch.Tensor
    embedding_list: list[float]
    video_embedding: torch.Tensor
    audio_embedding: torch.Tensor
    text_embedding: torch.Tensor
    modalities_present: list[str] = field(default_factory=list)
    is_checkpoint_loaded: bool = False
    device: str = "cpu"
    text_diagnostics: Optional[dict[str, Any]] = None
    visual_diagnostics: Optional[dict[str, Any]] = None
    audio_diagnostics: Optional[dict[str, Any]] = None


class VideoEmbeddingService:
    """
    Extracts and aggregates visual features using ResNet-50 and YouTube thumbnail intelligence.

    Expected Output Tensor Shape:
        - Single video: (1, 512)
        - Batch of N videos: (N, 512)
    """

    def __init__(
        self,
        video_encoder: Optional[Union[VideoEncoder, BaseVisualEncoder, nn.Module]] = None,
        processor: Optional[Union[VideoProcessor, VisualProcessor]] = None,
        frame_sampler: Optional[FrameSampler] = None,
        thumbnail_encoder: Optional[ThumbnailEncoder] = None,
        device: Optional[torch.device] = None,
        feature_dim: int = 512,
        settings: Optional[Settings] = None,
        video_weight: Optional[float] = None,
        thumbnail_weight: Optional[float] = None,
        pooling_method: Optional[str] = None,
        num_frames: Optional[int] = None,
    ):
        self.settings = settings or get_settings()
        self.device = device or torch.device("cpu")
        self.feature_dim = feature_dim
        self.num_frames = num_frames or getattr(self.settings, "VISUAL_FRAME_COUNT", 8)
        self.pooling_method = (pooling_method or getattr(self.settings, "VISUAL_FRAME_POOLING", "mean")).lower()
        self.video_weight = video_weight if video_weight is not None else getattr(self.settings, "VISUAL_VIDEO_WEIGHT", 0.7)
        self.thumbnail_weight = thumbnail_weight if thumbnail_weight is not None else getattr(self.settings, "VISUAL_THUMBNAIL_WEIGHT", 0.3)

        self.visual_processor = VisualProcessor(target_size=(224, 224), normalize=True)
        self.processor = processor or VideoProcessor(num_frames=self.num_frames, frame_size=(224, 224), normalize=True)
        self.frame_sampler = frame_sampler or FrameSampler(default_num_frames=self.num_frames)

        self.encoder = video_encoder or VideoEncoder(
            backbone=getattr(self.settings, "RESNET_BACKBONE", "resnet50"),
            feature_dim=feature_dim,
            pretrained=getattr(self.settings, "PRETRAINED_FOUNDATION", True),
            freeze_backbone=True
        )
        self.encoder.to(self.device)
        self.encoder.eval()

        self.thumbnail_encoder = thumbnail_encoder or ThumbnailEncoder(
            encoder=self.encoder if isinstance(self.encoder, BaseVisualEncoder) else None,
            processor=self.visual_processor,
            device=self.device,
            feature_dim=self.feature_dim,
        )

    def embed_video(
        self,
        video_path: Optional[Union[str, Path]] = None,
        thumbnail_path: Optional[Union[str, Path, bool]] = None,
        frames_tensor: Optional[torch.Tensor] = None,
        auto_discover_thumbnail: bool = True,
    ) -> VisualEmbeddingResult:
        """
        Extract video and thumbnail embeddings, apply visual fusion, and return structured result.

        Args:
            video_path: Optional local path to video file
            thumbnail_path: Optional local path to thumbnail image (pass False to disable thumbnail)
            frames_tensor: Optional pre-extracted frames tensor (T, 3, H, W) or (B, T, 3, H, W)
            auto_discover_thumbnail: Whether to auto-detect sibling thumbnail.jpg if thumbnail_path is None

        Returns:
            VisualEmbeddingResult with 512-D embedding and rich diagnostics.
        """
        t0 = time.perf_counter()
        encoder_name = getattr(self.settings, "VISUAL_ENCODER", "resnet50")
        video_frames_requested = self.num_frames
        video_frames_used = 0
        sampling_diag: dict[str, Any] = {}
        v_ok = False
        video_emb: Optional[torch.Tensor] = None

        with torch.no_grad():
            # 1. Process Video Stream
            if frames_tensor is not None:
                tensor = frames_tensor.float().to(self.device)
                if tensor.dim() == 4:
                    tensor = tensor.unsqueeze(0)

                if tensor.dim() == 5:
                    B, T = tensor.shape[:2]
                    video_frames_requested = T
                    video_frames_used = T
                    raw_v_emb = self.encoder(tensor)
                    raw_v_emb = torch.nan_to_num(raw_v_emb, nan=0.0, posinf=0.0, neginf=0.0)
                    v_ok = True
                    video_emb = F.normalize(raw_v_emb, p=2, dim=-1)

                    if B > 1:
                        # Batch tensor input
                        t_ms = (time.perf_counter() - t0) * 1000
                        return VisualEmbeddingResult(
                            embedding=video_emb,
                            is_available=True,
                            embedding_dimension=self.feature_dim,
                            video_frames_requested=video_frames_requested,
                            video_frames_used=video_frames_used,
                            thumbnail_used=False,
                            thumbnail_available=False,
                            pooling_method=self.pooling_method,
                            encoder_name=encoder_name,
                            processing_time_ms=t_ms,
                            modality_status="video_only",
                            video_embedding=video_emb,
                            thumbnail_embedding=None,
                            diagnostics={"input_type": "batch_tensor", "batch_size": B},
                        )
                else:
                    logger.warning("Unexpected frames_tensor shape: %s", frames_tensor.shape)

            elif video_path is not None:
                v_path_obj = Path(video_path)
                if v_path_obj.exists() and v_path_obj.is_file():
                    sample_res = self.frame_sampler.sample_frames(v_path_obj, num_frames=self.num_frames)
                    sampling_diag = sample_res.to_dict()
                    video_frames_requested = sample_res.requested_frames
                    video_frames_used = sample_res.decoded_frames

                    if sample_res.success and len(sample_res.frames) > 0:
                        try:
                            frames_t = self.visual_processor.preprocess_frames(sample_res.frames)
                            frames_t = frames_t.unsqueeze(0).to(self.device)  # (1, T, 3, 224, 224)
                            raw_v_emb = self.encoder(frames_t)
                            raw_v_emb = torch.nan_to_num(raw_v_emb, nan=0.0, posinf=0.0, neginf=0.0)
                            video_emb = F.normalize(raw_v_emb, p=2, dim=-1)
                            v_ok = True
                        except Exception as e:
                            logger.warning("Frame preprocessing/encoding failed for %s: %s", video_path, e)
                            sampling_diag["error_processing"] = str(e)
                    else:
                        logger.warning("Frame sampling returned 0 valid frames for %s", video_path)
                else:
                    logger.warning("Video file does not exist: %s", video_path)
                    sampling_diag["error"] = f"Video file not found: {video_path}"

            if video_emb is None:
                video_emb = torch.zeros(1, self.feature_dim, device=self.device)

            # 2. Process Thumbnail Stream
            candidate_thumb: Optional[Path] = None
            if thumbnail_path is False:
                candidate_thumb = None
            elif thumbnail_path is not None:
                candidate_thumb = Path(thumbnail_path)
            elif auto_discover_thumbnail and video_path is not None and isinstance(video_path, (str, Path)):
                # Auto-discover local thumbnail.jpg stored alongside video.mp4
                sibling_thumb = Path(video_path).parent / "thumbnail.jpg"
                if sibling_thumb.exists() and sibling_thumb.is_file():
                    candidate_thumb = sibling_thumb

            t_ok = False
            thumb_emb: Optional[torch.Tensor] = None
            thumb_diag: dict[str, Any] = {}

            if candidate_thumb is not None:
                thumb_res = self.thumbnail_encoder.encode_thumbnail(candidate_thumb)
                thumb_diag = thumb_res.to_dict()
                if thumb_res.is_available:
                    raw_t_emb = thumb_res.embedding.to(self.device)
                    thumb_emb = F.normalize(raw_t_emb, p=2, dim=-1)
                    t_ok = True
                else:
                    thumb_emb = torch.zeros(1, self.feature_dim, device=self.device)
            else:
                thumb_emb = torch.zeros(1, self.feature_dim, device=self.device)

            # 3. Deterministic Visual Fusion
            if v_ok and t_ok:
                fused = (self.video_weight * video_emb) + (self.thumbnail_weight * thumb_emb)
                final_emb = F.normalize(fused, p=2, dim=-1)
                modality_status = "video+thumbnail"
                is_available = True
            elif v_ok:
                final_emb = video_emb
                modality_status = "video_only"
                is_available = True
            elif t_ok:
                final_emb = thumb_emb
                modality_status = "thumbnail_only"
                is_available = True
            else:
                final_emb = torch.zeros(1, self.feature_dim, device=self.device)
                modality_status = "missing"
                is_available = False

            final_emb = torch.nan_to_num(final_emb, nan=0.0, posinf=0.0, neginf=0.0)
            t_ms = (time.perf_counter() - t0) * 1000

            diagnostics = {
                "sampling": sampling_diag,
                "thumbnail": thumb_diag,
                "video_weight": self.video_weight,
                "thumbnail_weight": self.thumbnail_weight,
            }

            return VisualEmbeddingResult(
                embedding=final_emb,
                is_available=is_available,
                embedding_dimension=self.feature_dim,
                video_frames_requested=video_frames_requested,
                video_frames_used=video_frames_used,
                thumbnail_used=t_ok,
                thumbnail_available=(candidate_thumb is not None and candidate_thumb.exists()),
                pooling_method=self.pooling_method,
                encoder_name=encoder_name,
                processing_time_ms=t_ms,
                modality_status=modality_status,
                video_embedding=video_emb if v_ok else None,
                thumbnail_embedding=thumb_emb if t_ok else None,
                diagnostics=diagnostics,
            )

    def encode(
        self,
        video_input: Optional[Union[str, Path, torch.Tensor, np.ndarray]] = None,
        thumbnail_input: Optional[Union[str, Path, torch.Tensor]] = None,
    ) -> VisualEmbeddingResult:
        """
        Encode video and optional thumbnail into a 512-dimensional embedding.
        Maintains complete backward compatibility with `emb, ok = svc.encode(...)`.
        """
        if video_input is None and thumbnail_input is None:
            return VisualEmbeddingResult(
                embedding=torch.zeros(1, self.feature_dim, device=self.device),
                is_available=False,
                embedding_dimension=self.feature_dim,
                video_frames_requested=self.num_frames,
                video_frames_used=0,
                thumbnail_used=False,
                thumbnail_available=False,
                pooling_method=self.pooling_method,
                encoder_name=getattr(self.settings, "VISUAL_ENCODER", "resnet50"),
                processing_time_ms=0.0,
                modality_status="missing",
                diagnostics={"error": "No visual input provided"},
            )

        if isinstance(video_input, np.ndarray):
            frames_t = torch.from_numpy(video_input).float()
            return self.embed_video(frames_tensor=frames_t, thumbnail_path=thumbnail_input)

        if isinstance(video_input, torch.Tensor):
            return self.embed_video(frames_tensor=video_input, thumbnail_path=thumbnail_input)

        if isinstance(video_input, (str, Path)):
            return self.embed_video(video_path=video_input, thumbnail_path=thumbnail_input)

        if thumbnail_input is not None:
            return self.embed_video(video_path=None, thumbnail_path=thumbnail_input)

        return VisualEmbeddingResult(
            embedding=torch.zeros(1, self.feature_dim, device=self.device),
            is_available=False,
            embedding_dimension=self.feature_dim,
            modality_status="missing",
        )


# Clean alias for Phase 6 VisualEmbeddingService API
VisualEmbeddingService = VideoEmbeddingService


@dataclass
class AudioEmbeddingResult:
    """
    Structured result returned by AudioEmbeddingService.

    Supports attribute access, dict conversion, and backwards-compatible tuple unpacking:
    `emb, ok = audio_service.encode(...)`.
    """
    embedding: torch.Tensor
    is_available: bool
    embedding_dimension: int = 512
    sample_rate: int = 16000
    duration_seconds: float = 0.0
    original_channel_count: int = 1
    num_chunks: int = 1
    chunk_aggregation_method: str = "mean"
    processing_time_ms: float = 0.0
    model_name: str = "facebook/wav2vec2-base-960h"
    is_checkpoint_loaded: bool = False
    is_truncated: bool = False
    failure_reason: Optional[str] = None
    acoustic_diagnostics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_tensor: bool = False) -> dict[str, Any]:
        """Convert audio metadata to dict without exposing raw local paths."""
        res = {
            "is_available": self.is_available,
            "embedding_dimension": self.embedding_dimension,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 2),
            "original_channel_count": self.original_channel_count,
            "num_chunks": self.num_chunks,
            "chunk_aggregation_method": self.chunk_aggregation_method,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model_name": self.model_name,
            "is_checkpoint_loaded": self.is_checkpoint_loaded,
            "is_truncated": self.is_truncated,
            "failure_reason": self.failure_reason,
            "acoustic_diagnostics": self.acoustic_diagnostics,
            "diagnostics": self.diagnostics,
        }
        if include_tensor:
            res["embedding"] = self.embedding
        else:
            res["embedding"] = self.embedding[0].cpu().tolist() if self.embedding.numel() > 0 else []
        return res

    def __iter__(self):
        """Enable backward-compatible unpacking: emb, ok = audio_service.encode(...)"""
        yield self.embedding
        yield self.is_available

    def __getitem__(self, key: Union[str, int]) -> Any:
        if key == 0 or key == "embedding":
            return self.embedding
        if key == 1 or key == "is_available":
            return self.is_available
        if key == "embedding_dimension":
            return self.embedding_dimension
        if key == "sample_rate":
            return self.sample_rate
        if key == "duration_seconds":
            return self.duration_seconds
        if key == "original_channel_count":
            return self.original_channel_count
        if key == "num_chunks":
            return self.num_chunks
        if key == "chunk_aggregation_method":
            return self.chunk_aggregation_method
        if key == "processing_time_ms":
            return self.processing_time_ms
        if key == "model_name":
            return self.model_name
        if key == "is_checkpoint_loaded":
            return self.is_checkpoint_loaded
        if key == "is_truncated":
            return self.is_truncated
        if key == "failure_reason":
            return self.failure_reason
        if key == "acoustic_diagnostics":
            return self.acoustic_diagnostics
        if key == "diagnostics":
            return self.diagnostics
        raise KeyError(key)


class AudioEmbeddingService:
    """
    Extracts and aggregates audio features using Wav2Vec2 and Acoustic Intelligence.

    Expected Output Tensor Shape:
        - Single audio track: (1, 512)
        - Batch of N audios: (N, 512)
    """

    def __init__(
        self,
        audio_encoder: Optional[AudioEncoder] = None,
        processor: Optional[Union[AudioProcessor, FeatureAudioProcessor]] = None,
        chunker: Optional[AudioChunker] = None,
        speech_recognizer: Optional[OptionalSpeechRecognizer] = None,
        device: Optional[torch.device] = None,
        feature_dim: int = 512,
        settings: Optional[Settings] = None,
        model_name: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.device = device or torch.device("cpu")
        self.feature_dim = feature_dim
        self.model_name = model_name or getattr(self.settings, "WAV2VEC2_MODEL", "facebook/wav2vec2-base-960h")
        self.sample_rate = getattr(self.settings, "AUDIO_SAMPLE_RATE", 16000)

        # Reusable AudioProcessor
        if processor is not None and isinstance(processor, FeatureAudioProcessor):
            self.processor = processor
        else:
            self.processor = FeatureAudioProcessor(
                sample_rate=self.sample_rate,
                max_duration_seconds=getattr(self.settings, "AUDIO_MAX_DURATION_SECONDS", 600.0),
            )

        # Reusable AudioChunker
        self.chunker = chunker or AudioChunker(
            chunk_duration_seconds=getattr(self.settings, "AUDIO_CHUNK_DURATION_SECONDS", 20.0),
            overlap_seconds=getattr(self.settings, "AUDIO_CHUNK_OVERLAP_SECONDS", 2.0),
            max_chunks=getattr(self.settings, "AUDIO_MAX_CHUNKS", 30),
            sample_rate=self.sample_rate,
        )

        # Optional SpeechRecognizer
        self.speech_recognizer = speech_recognizer or OptionalSpeechRecognizer(
            enabled=getattr(self.settings, "AUDIO_SPEECH_RECOGNITION_ENABLED", False),
        )

        # Audio encoder
        self.encoder = audio_encoder or AudioEncoder(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            freeze_backbone=True,
            pretrained=getattr(self.settings, "PRETRAINED_FOUNDATION", True),
        )
        self.encoder.to(self.device)
        self.encoder.eval()

    def encode(
        self,
        audio_input: Optional[Union[str, Path, torch.Tensor, np.ndarray]] = None
    ) -> AudioEmbeddingResult:
        """
        Encode audio waveform into a 512-dimensional embedding with acoustic diagnostics.

        Args:
            audio_input: Video/audio file path, raw waveform tensor (N,) or (B, N)

        Returns:
            AudioEmbeddingResult with embedding tensor, diagnostics, and tuple unpackability.
        """
        t0 = time.perf_counter()
        if audio_input is None:
            return AudioEmbeddingResult(
                embedding=torch.zeros(1, self.feature_dim, device=self.device),
                is_available=False,
                embedding_dimension=self.feature_dim,
                sample_rate=self.sample_rate,
                model_name=self.model_name,
                failure_reason="Audio input is None",
                processing_time_ms=0.0,
            )

        with torch.no_grad():
            # Check for batch tensor (B, N) where B > 1
            if isinstance(audio_input, torch.Tensor) and audio_input.dim() == 2 and audio_input.shape[0] > 1:
                batch = audio_input.float().to(self.device)
                emb = self.encoder(batch)
                emb = F.normalize(emb, p=2, dim=-1)
                emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return AudioEmbeddingResult(
                    embedding=emb,
                    is_available=True,
                    embedding_dimension=self.feature_dim,
                    sample_rate=self.sample_rate,
                    duration_seconds=audio_input.shape[1] / float(self.sample_rate),
                    original_channel_count=1,
                    num_chunks=1,
                    model_name=self.model_name,
                    processing_time_ms=elapsed_ms,
                    is_checkpoint_loaded=getattr(self.encoder, "pretrained", False),
                )

            # File path input
            if isinstance(audio_input, (str, Path)):
                audio_path_obj = Path(audio_input)
                if not audio_path_obj.exists():
                    logger.warning("Media file not found for audio extraction: %s", audio_input)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return AudioEmbeddingResult(
                        embedding=torch.zeros(1, self.feature_dim, device=self.device),
                        is_available=False,
                        embedding_dimension=self.feature_dim,
                        sample_rate=self.sample_rate,
                        model_name=self.model_name,
                        failure_reason=f"Media file not found: {audio_path_obj.name}",
                        processing_time_ms=elapsed_ms,
                    )
                proc_res = self.processor.process_file(audio_path_obj)
            elif isinstance(audio_input, (np.ndarray, torch.Tensor)):
                proc_res = self.processor.process_waveform(audio_input, sample_rate=self.sample_rate)
            else:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return AudioEmbeddingResult(
                    embedding=torch.zeros(1, self.feature_dim, device=self.device),
                    is_available=False,
                    embedding_dimension=self.feature_dim,
                    sample_rate=self.sample_rate,
                    model_name=self.model_name,
                    failure_reason=f"Unsupported audio input type: {type(audio_input)}",
                    processing_time_ms=elapsed_ms,
                )

            if not proc_res.is_valid:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return AudioEmbeddingResult(
                    embedding=torch.zeros(1, self.feature_dim, device=self.device),
                    is_available=False,
                    embedding_dimension=self.feature_dim,
                    sample_rate=proc_res.sample_rate,
                    duration_seconds=proc_res.duration_seconds,
                    original_channel_count=proc_res.original_channels,
                    model_name=self.model_name,
                    failure_reason=proc_res.error_message or "Invalid audio",
                    acoustic_diagnostics=proc_res.to_dict(),
                    processing_time_ms=elapsed_ms,
                )

            waveform = proc_res.waveform  # 1D tensor at 16 kHz
            chunk_res = self.chunker.chunk_waveform(waveform)
            num_chunks = chunk_res.num_chunks

            if num_chunks == 1 and not chunk_res.is_truncated:
                chunk_waveform = chunk_res.chunks.to(self.device)
                emb = self.encoder(chunk_waveform)
            else:
                chunks = chunk_res.chunks.to(self.device)
                attention_mask = chunk_res.attention_mask.to(self.device)
                if hasattr(self.encoder, "encode_chunks"):
                    emb = self.encoder.encode_chunks(
                        chunks,
                        attention_mask=attention_mask,
                        aggregation=getattr(self.settings, "AUDIO_POOLING", "mean"),
                    )
                else:
                    chunk_embs = [self.encoder(c.unsqueeze(0)) for c in chunks]
                    emb = torch.stack(chunk_embs, dim=0).mean(dim=0)

            # L2 Normalization
            emb = F.normalize(emb, p=2, dim=-1)
            emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            return AudioEmbeddingResult(
                embedding=emb,
                is_available=True,
                embedding_dimension=self.feature_dim,
                sample_rate=proc_res.sample_rate,
                duration_seconds=proc_res.duration_seconds,
                original_channel_count=proc_res.original_channels,
                num_chunks=num_chunks,
                chunk_aggregation_method=getattr(self.settings, "AUDIO_POOLING", "mean"),
                processing_time_ms=elapsed_ms,
                model_name=self.model_name,
                is_checkpoint_loaded=getattr(self.encoder, "pretrained", False),
                is_truncated=chunk_res.is_truncated,
                acoustic_diagnostics=proc_res.to_dict(),
                diagnostics={
                    "chunker": chunk_res.to_dict(),
                },
            )


from src.features.text_processor import (
    YouTubeTextInput,
    TextProcessor,
    TextQualityDiagnostics,
    clean_transcript,
)


@dataclass
class TextEmbeddingResult:
    """
    Structured result returned by TextEmbeddingService.

    Supports both object attribute access, dictionary key access,
    and backwards-compatible tuple unpacking: `emb, ok = svc.encode(...)`.
    """
    embedding: torch.Tensor
    token_count: int
    chunk_count: int
    transcript_used: bool
    transcript_available: bool = False
    description_available: bool = False
    tags_available: bool = False
    is_available: bool = True

    def to_dict(self, include_tensor: bool = False) -> dict[str, Any]:
        """Convert to dict without exposing raw internal tensor by default."""
        res = {
            "token_count": self.token_count,
            "chunk_count": self.chunk_count,
            "transcript_used": self.transcript_used,
            "transcript_available": self.transcript_available,
            "description_available": self.description_available,
            "tags_available": self.tags_available,
            "is_available": self.is_available,
        }
        if include_tensor:
            res["embedding"] = self.embedding
        else:
            res["embedding"] = self.embedding[0].cpu().tolist() if self.embedding.numel() > 0 else []
        return res

    def __iter__(self):
        """Enable backward-compatible unpacking: emb, ok = text_service.encode(...)"""
        yield self.embedding
        yield self.is_available

    def __getitem__(self, key: Union[str, int]) -> Any:
        """Enable dict-like and index-based access."""
        if key == "embedding" or key == 0:
            return self.embedding
        if key == "token_count":
            return self.token_count
        if key == "chunk_count":
            return self.chunk_count
        if key == "transcript_used":
            return self.transcript_used
        if key == 1 or key == "is_available":
            return self.is_available
        if key == "transcript_available":
            return self.transcript_available
        if key == "description_available":
            return self.description_available
        if key == "tags_available":
            return self.tags_available
        raise KeyError(key)


class TextEmbeddingService:
    """
    Extracts text features using BERT baseline with chunking and YouTube metadata modeling.

    Supports:
        - Plain string: "Video Title"
        - Canonical YouTubeTextInput
        - Structured metadata dictionary:
          {
              "title": "...",
              "description": "...",
              "tags": [...],
              "transcript": "..."
          }

    Expected Output Tensor Shape:
        - Single text: (1, 512)
        - Batch of N texts: (N, 512)
    """

    def __init__(
        self,
        text_encoder: Optional[TextEncoder] = None,
        processor: Optional[TextProcessor] = None,
        device: Optional[torch.device] = None,
        feature_dim: int = 512
    ):
        self.device = device or torch.device("cpu")
        self.feature_dim = feature_dim
        self.processor = processor or TextProcessor()
        self.encoder = text_encoder or TextEncoder(
            model_name="bert-base-uncased",
            feature_dim=feature_dim,
            freeze_backbone=True,
            pretrained=True
        )
        self.encoder.to(self.device)
        self.encoder.eval()

    def encode(
        self,
        text_input: Optional[Union[str, dict[str, Any], YouTubeTextInput, list[Any]]] = None
    ) -> TextEmbeddingResult:
        """
        Encode text into a 512-dimensional embedding with chunking and quality diagnostics.

        Args:
            text_input: String, dictionary with title/description/tags/transcript,
                        YouTubeTextInput, or list of such items.

        Returns:
            TextEmbeddingResult with embedding Tensor, token_count, chunk_count, and diagnostics.
        """
        if text_input is None:
            return TextEmbeddingResult(
                embedding=torch.zeros(1, self.feature_dim, device=self.device),
                token_count=0,
                chunk_count=0,
                transcript_used=False,
                is_available=False
            )

        with torch.no_grad():
            # Batch input handling
            if isinstance(text_input, list):
                if len(text_input) == 0:
                    return TextEmbeddingResult(
                        embedding=torch.zeros(0, self.feature_dim, device=self.device),
                        token_count=0,
                        chunk_count=0,
                        transcript_used=False,
                        is_available=False
                    )

                results = [self.encode(item) for item in text_input]
                all_embeddings = torch.cat([r.embedding for r in results], dim=0)
                any_available = any(r.is_available for r in results)
                total_tokens = sum(r.token_count for r in results)
                total_chunks = sum(r.chunk_count for r in results)
                any_transcript = any(r.transcript_used for r in results)

                return TextEmbeddingResult(
                    embedding=all_embeddings,
                    token_count=total_tokens,
                    chunk_count=total_chunks,
                    transcript_used=any_transcript,
                    transcript_available=any_transcript,
                    description_available=any(r.description_available for r in results),
                    tags_available=any(r.tags_available for r in results),
                    is_available=any_available
                )

            # Single item processing
            structured_text, diag, _ = self.processor.process(text_input)

            if not structured_text.strip():
                return TextEmbeddingResult(
                    embedding=torch.zeros(1, self.feature_dim, device=self.device),
                    token_count=0,
                    chunk_count=0,
                    transcript_used=False,
                    is_available=False
                )

            # Check if encoder supports chunked long-text encoding
            if hasattr(self.encoder, "encode_single_long_text"):
                emb, num_chunks, total_tokens = self.encoder.encode_single_long_text(structured_text)
            else:
                # Fallback for standard or dummy text encoders
                emb = self.encoder(texts=[structured_text])
                num_chunks = 1
                total_tokens = len(structured_text.split())

            emb = emb.to(self.device)
            emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

            return TextEmbeddingResult(
                embedding=emb,
                token_count=total_tokens,
                chunk_count=num_chunks,
                transcript_used=diag.transcript_available,
                transcript_available=diag.transcript_available,
                description_available=diag.description_available,
                tags_available=diag.tags_available,
                is_available=True
            )


class MultimodalInferenceService:
    """
    Unified multimodal inference pipeline.

    Coordinates:
        - Video frames -> ResNet-50 -> (B, 512)
        - Audio waveform -> Wav2Vec2 -> (B, 512)
        - Text metadata -> BERT -> (B, 512)
        - GatedFusion -> (B, 512)
        - L2 Normalization -> Unit sphere for Cosine Similarity search.
    """

    def __init__(
        self,
        model: Optional[TwoTowerModel] = None,
        settings: Optional[Settings] = None,
        device: Optional[torch.device] = None
    ):
        self.settings = settings or get_settings()
        self.device = device or torch.device(
            "cuda" if (torch.cuda.is_available() and self.settings.DEVICE in ("auto", "cuda")) else "cpu"
        )
        self.feature_dim = self.settings.FEATURE_DIM
        self.is_checkpoint_loaded = False

        # Configure deterministic settings if requested
        if self.settings.DETERMINISTIC_INFERENCE:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if model is not None:
            self.model = model
            self.model.to(self.device)
            self.model.eval()
            self._init_subservices_from_model()
        else:
            self._init_pipeline()

    def _init_subservices_from_model(self):
        """Bind subservices from existing TwoTowerModel."""
        video_tower = getattr(self.model, "video_tower", self.model)
        self.video_service = VideoEmbeddingService(
            video_encoder=getattr(video_tower, "video_encoder", None),
            device=self.device,
            feature_dim=self.feature_dim
        )
        self.audio_service = AudioEmbeddingService(
            audio_encoder=getattr(video_tower, "audio_encoder", None),
            device=self.device,
            feature_dim=self.feature_dim
        )
        self.text_service = TextEmbeddingService(
            text_encoder=getattr(video_tower, "text_encoder", None),
            device=self.device,
            feature_dim=self.feature_dim
        )
        self.fusion = getattr(video_tower, "fusion", GatedFusion(self.feature_dim, self.feature_dim))
        self.fusion.to(self.device)
        self.fusion.eval()

    def _init_pipeline(self):
        """
        Initialize foundation models and evaluate checkpoint policy.
        Distinguishes between:
            A. Pretrained foundation weights (ResNet-50, Wav2Vec2, BERT)
            B. Project-specific fusion/projection checkpoint
        """
        checkpoint_path = Path(self.settings.MODEL_PATH)
        if not checkpoint_path.exists() and self.settings.APP_ENV == "production" and not self.settings.ALLOW_UNINITIALIZED_MODEL:
            raise RuntimeError(
                f"Production environment requires a valid checkpoint at '{checkpoint_path}'. "
                "No checkpoint found. Terminating startup."
            )

        pretrained_foundation = self.settings.PRETRAINED_FOUNDATION

        self.model = TwoTowerModel(
            feature_dim=self.feature_dim,
            share_towers=True,
            freeze_encoders=True,
            fusion_type=self.settings.FUSION_TYPE,
            pretrained=pretrained_foundation,
            video_backbone=self.settings.RESNET_BACKBONE,
            audio_model=self.settings.WAV2VEC2_MODEL,
            text_model=self.settings.BERT_MODEL
        )
        self.model.to(self.device)
        self.model.eval()

        if checkpoint_path.exists():
            logger.info("Loading project checkpoint from %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.is_checkpoint_loaded = True
            logger.info("Project checkpoint successfully loaded.")
        else:
            if self.settings.APP_ENV == "production" and not self.settings.ALLOW_UNINITIALIZED_MODEL:
                raise RuntimeError(
                    f"Production environment requires a valid checkpoint at '{checkpoint_path}'. "
                    "No checkpoint found. Terminating startup."
                )
            logger.warning(
                "[DEVELOPMENT MODE] No project checkpoint found at '%s'. "
                "Running with pretrained foundation encoders and initialized projection/fusion layers.",
                checkpoint_path
            )
            self.is_checkpoint_loaded = False

        self._init_subservices_from_model()

    def fuse(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse individual modality embeddings using GatedFusion and apply L2 normalization.

        Normalization Strategy:
            1. Modality encoders generate 512-d representations.
            2. GatedFusion computes dynamic gating weights with zero-modality masking.
            3. Fused vector is projected to 512 dimensions.
            4. Result is L2-normalized: F.normalize(fused, p=2, dim=-1).
            5. Final vector is cleaned via torch.nan_to_num.

        Returns:
            Normalized 512-dimensional tensor.
        """
        with torch.no_grad():
            fused = self.fusion(video_emb, audio_emb, text_emb)
            # L2 normalize so dot product equals cosine similarity
            normalized = F.normalize(fused, p=2, dim=-1)
            cleaned = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            return cleaned

    def encode_multimodal(
        self,
        video_path: Optional[Union[str, Path]] = None,
        text: Optional[Union[str, dict[str, Any], list[Any]]] = None,
        frames: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        thumbnail_path: Optional[Union[str, Path]] = None,
    ) -> MultimodalInferenceResult:
        """
        Execute deterministic end-to-end multimodal inference.

        Args:
            video_path: Optional path to video file (used for frames and audio)
            text: Video metadata (dict or string)
            frames: Pre-extracted video frames tensor
            audio: Pre-extracted audio waveform tensor
            thumbnail_path: Optional path to local thumbnail file

        Returns:
            MultimodalInferenceResult with normalized embedding and individual modality features.
        """
        with torch.no_grad():
            modalities_present = []

            # 1. Visual stream (Video + Thumbnail Intelligence)
            if hasattr(self.video_service, "embed_video"):
                if frames is not None:
                    visual_res = self.video_service.embed_video(frames_tensor=frames, thumbnail_path=thumbnail_path)
                else:
                    visual_res = self.video_service.embed_video(video_path=video_path, thumbnail_path=thumbnail_path)
                video_emb = visual_res.embedding
                v_ok = visual_res.is_available
                visual_diag = visual_res.to_dict()
            else:
                if frames is not None:
                    video_emb, v_ok = self.video_service.encode(frames)
                else:
                    video_emb, v_ok = self.video_service.encode(video_path)
                visual_diag = {"is_available": v_ok}

            if v_ok:
                modalities_present.append("video")

            # 2. Audio stream
            audio_res = self.audio_service.encode(audio if audio is not None else video_path)
            if isinstance(audio_res, AudioEmbeddingResult):
                audio_emb = audio_res.embedding
                a_ok = audio_res.is_available
                audio_diag = audio_res.to_dict()
            else:
                audio_emb, a_ok = audio_res
                audio_diag = {"is_available": a_ok}

            if a_ok:
                modalities_present.append("audio")

            # 3. Text stream
            text_res = self.text_service.encode(text)
            text_emb = text_res.embedding
            if text_res.is_available:
                modalities_present.append("text")

            # 4. Fusion and L2 Normalization
            fused_emb = self.fuse(video_emb, audio_emb, text_emb)

            return MultimodalInferenceResult(
                embedding=fused_emb,
                embedding_list=fused_emb[0].cpu().tolist(),
                video_embedding=video_emb,
                audio_embedding=audio_emb,
                text_embedding=text_emb,
                modalities_present=modalities_present,
                is_checkpoint_loaded=self.is_checkpoint_loaded,
                device=str(self.device),
                text_diagnostics=text_res.to_dict(),
                visual_diagnostics=visual_diag,
                audio_diagnostics=audio_diag,
            )
