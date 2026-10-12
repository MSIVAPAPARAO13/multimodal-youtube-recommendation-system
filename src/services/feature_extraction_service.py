"""
Feature Extraction Service (Phase 12B).

Extracts, validates, normalizes, and persists real visual (ResNet-50 frame & thumbnail)
and audio (Wav2Vec2) feature embeddings for catalog and YouTube video assets.
Associates all generated features with provenance metadata, source checksums,
model identity, dimensions, and validation states.
"""

import hashlib
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sqlalchemy.orm import Session

from src.db.config import get_settings, Settings
from src.db.models import (
    Video,
    VideoAsset,
    VideoFeature,
    AssetType,
    AssetStatus,
    FeatureModality,
)
from src.db.repository import VideoAssetRepository, VideoFeatureRepository
from src.features.visual_processor import VisualProcessor
from src.features.visual_encoder import ResNet50VisualEncoder, CLIPExperimentalEncoder
from src.features.thumbnail_encoder import ThumbnailEncoder
from src.features.audio_processor import AudioProcessor
from src.features.audio_encoder import AudioEncoder
from src.integrations.youtube.media.validator import MediaValidator
from src.storage.local import LocalStorage

logger = logging.getLogger(__name__)


class FeatureExtractionService:
    """
    Multimodal Feature Extraction Service.
    Coordinates model loading, asset checksum verification, frame and waveform processing,
    and persistent storage of embeddings in video_features table.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        storage: Optional[LocalStorage] = None,
        device: Optional[torch.device] = None,
        enable_clip: bool = True,
    ):
        self.settings = settings or get_settings()
        self.storage = storage or LocalStorage(root_dir="data/raw/youtube")
        self.device = device or torch.device("cpu")
        self.validator = MediaValidator()

        logger.info("Initializing FeatureExtractionService on device: %s", self.device)

        # 1. Visual Processor and ResNet-50 Backbone
        self.visual_processor = VisualProcessor(target_size=(224, 224), normalize=True)
        self.resnet_encoder = ResNet50VisualEncoder(
            feature_dim=512,
            pretrained=True,
            freeze_backbone=True,
            aggregation="mean",
        )
        self.resnet_encoder.to(self.device)
        self.resnet_encoder.eval()

        # 2. Thumbnail Encoder
        self.thumbnail_encoder = ThumbnailEncoder(
            encoder=self.resnet_encoder,
            processor=self.visual_processor,
            device=self.device,
            feature_dim=512,
        )

        # 3. Audio Processor & Wav2Vec2 Encoder
        self.audio_processor = AudioProcessor(
            target_sample_rate=16000,
            max_duration_seconds=5.0,
        )
        self.audio_encoder = AudioEncoder(
            feature_dim=512,
            aggregation="mean",
            freeze_backbone=True,
            pretrained=False,  # Deterministic offline architecture inference
        )
        self.audio_encoder.to(self.device)
        self.audio_encoder.eval()

        # 4. Experimental CLIP Encoder (Optional)
        self.enable_clip = enable_clip
        if enable_clip:
            self.clip_encoder = CLIPExperimentalEncoder(
                feature_dim=512,
                pretrained=False,
                freeze_backbone=True,
                aggregation="mean",
            )
            self.clip_encoder.to(self.device)
            self.clip_encoder.eval()
        else:
            self.clip_encoder = None

    def _compute_composite_checksum(self, checksums: List[str]) -> str:
        """Combine multiple source checksums into a deterministic single hash."""
        concat_str = "".join(sorted(checksums))
        return hashlib.sha256(concat_str.encode("utf-8")).hexdigest()

    def extract_visual_features_for_video(
        self,
        video_id: str,
        session: Session,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract 2048-D raw and 512-D projected ResNet-50 features from 8 video frames.
        """
        t0 = time.perf_counter()
        asset_repo = VideoAssetRepository(session)
        feat_repo = VideoFeatureRepository(session)

        # 1. Fetch FRAME assets
        assets = asset_repo.get_assets_for_video(video_id)
        frame_assets = [
            a for a in assets
            if a.asset_type == AssetType.FRAME.value and a.status == AssetStatus.READY.value
        ]
        # Sort by frame_index stored in extra_metadata or storage path
        frame_assets.sort(
            key=lambda a: (a.extra_metadata or {}).get("frame_index", a.storage_path)
        )

        if len(frame_assets) < 8:
            return {
                "success": False,
                "status": "INSUFFICIENT_FRAMES",
                "message": f"Expected at least 8 ready frames, found {len(frame_assets)}",
                "features_generated": [],
            }

        # Select first 8 deterministic frames
        selected_frames = frame_assets[:8]
        frame_paths: List[Path] = []
        frame_checksums: List[str] = []

        for fa in selected_frames:
            try:
                full_path = self.storage.get_path(fa.storage_path)
            except Exception as e:
                return {
                    "success": False,
                    "status": "STORAGE_PATH_ERROR",
                    "message": f"Storage path resolution failed for {fa.storage_path}: {e}",
                    "features_generated": [],
                }

            if not full_path.exists():
                return {
                    "success": False,
                    "status": "FILE_MISSING",
                    "message": f"Frame file not found on disk: {full_path}",
                    "features_generated": [],
                }

            # Checksum integrity
            current_chk = MediaValidator.compute_sha256(full_path)
            if fa.checksum_sha256 and current_chk != fa.checksum_sha256:
                return {
                    "success": False,
                    "status": "CHECKSUM_MISMATCH",
                    "message": f"Checksum mismatch for frame {fa.storage_path}: expected {fa.checksum_sha256}, got {current_chk}",
                    "features_generated": [],
                }

            frame_paths.append(full_path)
            frame_checksums.append(current_chk)

        composite_checksum = self._compute_composite_checksum(frame_checksums)

        # 2. Check for existing active feature
        if not force:
            existing_2048 = feat_repo.get_feature(video_id, FeatureModality.VISUAL_FRAMES, "resnet50_mean_2048")
            existing_512 = feat_repo.get_feature(video_id, FeatureModality.VISUAL_FRAMES, "resnet50_projected_512")
            if (
                existing_2048 and existing_2048.source_checksum == composite_checksum and
                existing_512 and existing_512.source_checksum == composite_checksum
            ):
                return {
                    "success": True,
                    "status": "SKIPPED_IDEMPOTENT",
                    "message": "Valid features with identical source checksum already exist",
                    "features_generated": [existing_2048.to_dict(), existing_512.to_dict()],
                    "duration_seconds": round(time.perf_counter() - t0, 4),
                }

        # Invalidate any stale features if source checksum changed
        feat_repo.invalidate_stale_features(video_id, FeatureModality.VISUAL_FRAMES, composite_checksum)

        # 3. Preprocess frames
        try:
            frames_tensor = self.visual_processor.preprocess_frames(frame_paths)  # (8, 3, 224, 224)
            frames_tensor = frames_tensor.to(self.device)
        except Exception as e:
            return {
                "success": False,
                "status": "PREPROCESSING_FAILED",
                "message": f"Image preprocessing failed: {e}",
                "features_generated": [],
            }

        # 4. Extract embeddings
        with torch.no_grad():
            # Raw ResNet-50 2048-D features
            raw_feats = self.resnet_encoder.extract_backbone_features(frames_tensor)  # (8, 2048)
            pooled_2048 = raw_feats.mean(dim=0, keepdim=True)  # (1, 2048)
            norm_2048 = F.normalize(pooled_2048, p=2, dim=-1)
            norm_2048 = torch.nan_to_num(norm_2048, nan=0.0)

            # Projected 512-D features
            proj_512 = self.resnet_encoder(frames_tensor.unsqueeze(0))  # (1, 512)
            norm_512 = F.normalize(proj_512, p=2, dim=-1)
            norm_512 = torch.nan_to_num(norm_512, nan=0.0)

        vec_2048 = norm_2048[0].cpu().tolist()
        vec_512 = norm_512[0].cpu().tolist()

        # 5. Persist into repository
        preprocessing_meta = {
            "target_size": [224, 224],
            "normalization": "imagenet",
            "frame_count": 8,
            "pooling": "mean",
        }

        yt_id = selected_frames[0].youtube_video_id

        feat_2048 = feat_repo.upsert_feature(
            video_id=video_id,
            modality=FeatureModality.VISUAL_FRAMES,
            feature_type="resnet50_mean_2048",
            encoder_name="torchvision_resnet50",
            encoder_version="IMAGENET1K_V1",
            dimension=2048,
            source_checksum=composite_checksum,
            embedding=vec_2048,
            youtube_video_id=yt_id,
            checkpoint_id="torch_hub_resnet50-0676ba61",
            preprocessing_config=preprocessing_meta,
            is_trained=True,  # ImageNet-1K pretrained weights
            validation_status="VALID",
            pipeline_version="12B",
        )

        feat_512 = feat_repo.upsert_feature(
            video_id=video_id,
            modality=FeatureModality.VISUAL_FRAMES,
            feature_type="resnet50_projected_512",
            encoder_name="torchvision_resnet50_projected",
            encoder_version="1.0.0",
            dimension=512,
            source_checksum=composite_checksum,
            embedding=vec_512,
            youtube_video_id=yt_id,
            checkpoint_id="untrained_projection_head",
            preprocessing_config=preprocessing_meta,
            is_trained=False,  # Untrained projection head
            validation_status="VALID",
            pipeline_version="12B",
        )

        return {
            "success": True,
            "status": "EXTRACTED",
            "message": "Successfully extracted ResNet-50 2048-D and 512-D visual features",
            "features_generated": [feat_2048.to_dict(), feat_512.to_dict()],
            "duration_seconds": round(time.perf_counter() - t0, 4),
        }

    def extract_thumbnail_features_for_video(
        self,
        video_id: str,
        session: Session,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract ResNet-50 thumbnail features (2048-D raw and 512-D projected).
        """
        t0 = time.perf_counter()
        asset_repo = VideoAssetRepository(session)
        feat_repo = VideoFeatureRepository(session)

        thumb_asset = asset_repo.get_asset_by_type(video_id, AssetType.THUMBNAIL)
        if not thumb_asset or thumb_asset.status != AssetStatus.READY.value:
            return {
                "success": False,
                "status": "THUMBNAIL_NOT_READY",
                "message": f"Ready thumbnail asset not found for video {video_id}",
                "features_generated": [],
            }

        try:
            full_path = self.storage.get_path(thumb_asset.storage_path)
        except Exception as e:
            return {
                "success": False,
                "status": "STORAGE_PATH_ERROR",
                "message": str(e),
                "features_generated": [],
            }

        if not full_path.exists():
            return {
                "success": False,
                "status": "FILE_MISSING",
                "message": f"Thumbnail file not found on disk: {full_path}",
                "features_generated": [],
            }

        checksum = MediaValidator.compute_sha256(full_path)
        if thumb_asset.checksum_sha256 and checksum != thumb_asset.checksum_sha256:
            return {
                "success": False,
                "status": "CHECKSUM_MISMATCH",
                "message": f"Thumbnail checksum mismatch: expected {thumb_asset.checksum_sha256}, got {checksum}",
                "features_generated": [],
            }

        # Idempotency check
        if not force:
            existing_2048 = feat_repo.get_feature(video_id, FeatureModality.VISUAL_THUMBNAIL, "resnet50_thumbnail_2048")
            existing_512 = feat_repo.get_feature(video_id, FeatureModality.VISUAL_THUMBNAIL, "resnet50_thumbnail_512")
            if (
                existing_2048 and existing_2048.source_checksum == checksum and
                existing_512 and existing_512.source_checksum == checksum
            ):
                return {
                    "success": True,
                    "status": "SKIPPED_IDEMPOTENT",
                    "message": "Valid thumbnail features with identical checksum already exist",
                    "features_generated": [existing_2048.to_dict(), existing_512.to_dict()],
                    "duration_seconds": round(time.perf_counter() - t0, 4),
                }

        feat_repo.invalidate_stale_features(video_id, FeatureModality.VISUAL_THUMBNAIL, checksum)

        try:
            img_tensor = self.visual_processor.preprocess_thumbnail(full_path).to(self.device)
            with torch.no_grad():
                raw_2048 = self.resnet_encoder.extract_backbone_features(img_tensor)
                norm_2048 = F.normalize(raw_2048, p=2, dim=-1)
                norm_2048 = torch.nan_to_num(norm_2048, nan=0.0)

                proj_512 = self.resnet_encoder.encode_single_image(img_tensor)
                norm_512 = F.normalize(proj_512, p=2, dim=-1)
                norm_512 = torch.nan_to_num(norm_512, nan=0.0)

            vec_2048 = norm_2048[0].cpu().tolist()
            vec_512 = norm_512[0].cpu().tolist()

            preprocessing_meta = {
                "target_size": [224, 224],
                "normalization": "imagenet",
            }

            f_2048 = feat_repo.upsert_feature(
                video_id=video_id,
                modality=FeatureModality.VISUAL_THUMBNAIL,
                feature_type="resnet50_thumbnail_2048",
                encoder_name="torchvision_resnet50",
                encoder_version="IMAGENET1K_V1",
                dimension=2048,
                source_checksum=checksum,
                embedding=vec_2048,
                youtube_video_id=thumb_asset.youtube_video_id,
                asset_id=thumb_asset.id,
                checkpoint_id="torch_hub_resnet50-0676ba61",
                preprocessing_config=preprocessing_meta,
                is_trained=True,
                validation_status="VALID",
                pipeline_version="12B",
            )

            f_512 = feat_repo.upsert_feature(
                video_id=video_id,
                modality=FeatureModality.VISUAL_THUMBNAIL,
                feature_type="resnet50_thumbnail_512",
                encoder_name="torchvision_resnet50_projected",
                encoder_version="1.0.0",
                dimension=512,
                source_checksum=checksum,
                embedding=vec_512,
                youtube_video_id=thumb_asset.youtube_video_id,
                asset_id=thumb_asset.id,
                checkpoint_id="untrained_projection_head",
                preprocessing_config=preprocessing_meta,
                is_trained=False,
                validation_status="VALID",
                pipeline_version="12B",
            )

            # Optional CLIP Thumbnail extraction
            f_clip = None
            if self.enable_clip and self.clip_encoder is not None:
                with torch.no_grad():
                    clip_emb = self.clip_encoder.encode_single_image(img_tensor)
                    clip_norm = F.normalize(clip_emb, p=2, dim=-1)
                    clip_norm = torch.nan_to_num(clip_norm, nan=0.0)
                    vec_clip = clip_norm[0].cpu().tolist()

                f_clip = feat_repo.upsert_feature(
                    video_id=video_id,
                    modality=FeatureModality.CLIP_THUMBNAIL,
                    feature_type="clip_thumbnail_512",
                    encoder_name="clip_vision_base",
                    encoder_version="1.0.0",
                    dimension=512,
                    source_checksum=checksum,
                    embedding=vec_clip,
                    youtube_video_id=thumb_asset.youtube_video_id,
                    asset_id=thumb_asset.id,
                    checkpoint_id="offline_clip_architecture",
                    preprocessing_config=preprocessing_meta,
                    is_trained=False,
                    validation_status="VALID",
                    pipeline_version="12B",
                )

            gen = [f_2048.to_dict(), f_512.to_dict()]
            if f_clip:
                gen.append(f_clip.to_dict())

            return {
                "success": True,
                "status": "EXTRACTED",
                "message": "Successfully extracted thumbnail visual features",
                "features_generated": gen,
                "duration_seconds": round(time.perf_counter() - t0, 4),
            }

        except Exception as e:
            return {
                "success": False,
                "status": "EXTRACTION_FAILED",
                "message": f"Thumbnail extraction failed: {e}",
                "features_generated": [],
            }

    def extract_audio_features_for_video(
        self,
        video_id: str,
        session: Session,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract audio features from 16 kHz mono WAV asset using Wav2Vec2.
        Enforces non-empty signal and explicitly rejects silent audio files
        per Phase 12B non-negotiable rules.
        """
        t0 = time.perf_counter()
        asset_repo = VideoAssetRepository(session)
        feat_repo = VideoFeatureRepository(session)

        audio_asset = asset_repo.get_asset_by_type(video_id, AssetType.AUDIO)
        if not audio_asset or audio_asset.status != AssetStatus.READY.value:
            return {
                "success": False,
                "status": "AUDIO_ASSET_NOT_READY",
                "message": f"Ready audio asset not found for video {video_id}",
                "features_generated": [],
            }

        try:
            full_path = self.storage.get_path(audio_asset.storage_path)
        except Exception as e:
            return {
                "success": False,
                "status": "STORAGE_PATH_ERROR",
                "message": str(e),
                "features_generated": [],
            }

        if not full_path.exists():
            return {
                "success": False,
                "status": "FILE_MISSING",
                "message": f"Audio file not found on disk: {full_path}",
                "features_generated": [],
            }

        checksum = MediaValidator.compute_sha256(full_path)
        if audio_asset.checksum_sha256 and checksum != audio_asset.checksum_sha256:
            return {
                "success": False,
                "status": "CHECKSUM_MISMATCH",
                "message": f"Audio checksum mismatch: expected {audio_asset.checksum_sha256}, got {checksum}",
                "features_generated": [],
            }

        # Idempotency check
        if not force:
            existing_audio = feat_repo.get_feature(video_id, FeatureModality.AUDIO, "wav2vec2_mean_512")
            if existing_audio and existing_audio.source_checksum == checksum:
                return {
                    "success": True,
                    "status": "SKIPPED_IDEMPOTENT",
                    "message": "Valid audio feature with identical checksum already exists",
                    "features_generated": [existing_audio.to_dict()],
                    "duration_seconds": round(time.perf_counter() - t0, 4),
                }

        # Validate waveform and check for nonempty signal
        proc_result = self.audio_processor.process_audio(full_path)

        # Explicit silence and empty signal rejection
        if (
            proc_result.is_silent
            or proc_result.rms_energy == 0.0
            or proc_result.silence_ratio >= 0.99
            or (proc_result.error_message and "silent" in proc_result.error_message.lower())
        ):
            return {
                "success": False,
                "status": "SILENT_AUDIO_REJECTED",
                "message": f"Audio asset rejected: silent waveform ({proc_result.error_message or f'RMS energy: {proc_result.rms_energy}'}). Fake audio is strictly forbidden.",
                "features_generated": [],
            }

        if not proc_result.is_valid:
            return {
                "success": False,
                "status": "INVALID_WAVEFORM",
                "message": f"Invalid audio waveform: {proc_result.error_message}",
                "features_generated": [],
            }

        feat_repo.invalidate_stale_features(video_id, FeatureModality.AUDIO, checksum)

        try:
            waveform_tensor = proc_result.waveform.unsqueeze(0).to(self.device)  # (1, num_samples)
            with torch.no_grad():
                audio_emb = self.audio_encoder(waveform_tensor)  # (1, 512)
                norm_emb = F.normalize(audio_emb, p=2, dim=-1)
                norm_emb = torch.nan_to_num(norm_emb, nan=0.0)

            vec_512 = norm_emb[0].cpu().tolist()

            feat_audio = feat_repo.upsert_feature(
                video_id=video_id,
                modality=FeatureModality.AUDIO,
                feature_type="wav2vec2_mean_512",
                encoder_name="facebook_wav2vec2_base",
                encoder_version="1.0.0",
                dimension=512,
                source_checksum=checksum,
                embedding=vec_512,
                youtube_video_id=audio_asset.youtube_video_id,
                asset_id=audio_asset.id,
                checkpoint_id="wav2vec2_architecture_eval",
                preprocessing_config={
                    "sample_rate": 16000,
                    "mono": True,
                    "rms_energy": proc_result.rms_energy,
                    "duration_seconds": proc_result.duration_seconds,
                },
                is_trained=False,  # Untrained projection head
                validation_status="VALID",
                pipeline_version="12B",
            )

            return {
                "success": True,
                "status": "EXTRACTED",
                "message": "Successfully extracted Wav2Vec2 512-D audio features",
                "features_generated": [feat_audio.to_dict()],
                "duration_seconds": round(time.perf_counter() - t0, 4),
            }

        except Exception as e:
            return {
                "success": False,
                "status": "EXTRACTION_FAILED",
                "message": f"Audio feature extraction failed: {e}",
                "features_generated": [],
            }

    def extract_features_for_video(
        self,
        video_id: str,
        session: Session,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract all available modalities for a single video.
        """
        vis_res = self.extract_visual_features_for_video(video_id, session, force=force)
        thumb_res = self.extract_thumbnail_features_for_video(video_id, session, force=force)
        aud_res = self.extract_audio_features_for_video(video_id, session, force=force)

        all_features = (
            vis_res.get("features_generated", []) +
            thumb_res.get("features_generated", []) +
            aud_res.get("features_generated", [])
        )

        return {
            "video_id": video_id,
            "visual_frames": vis_res,
            "visual_thumbnail": thumb_res,
            "audio": aud_res,
            "total_features_generated": len(all_features),
        }

    def batch_extract_features(
        self,
        video_ids: List[str],
        session: Session,
        batch_size: int = 10,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Bounded batch feature extraction with per-video failure isolation.
        """
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        counts = {
            "total": len(video_ids),
            "visual_frames_success": 0,
            "visual_thumbnail_success": 0,
            "audio_success": 0,
            "audio_rejected_silent": 0,
            "failed_videos": 0,
        }

        # Process in chunks of batch_size to bound memory
        for i in range(0, len(video_ids), batch_size):
            chunk = video_ids[i:i + batch_size]
            for vid in chunk:
                try:
                    res = self.extract_features_for_video(vid, session, force=force)
                    results.append(res)

                    if res["visual_frames"]["success"]:
                        counts["visual_frames_success"] += 1
                    if res["visual_thumbnail"]["success"]:
                        counts["visual_thumbnail_success"] += 1
                    if res["audio"]["success"]:
                        counts["audio_success"] += 1
                    elif res["audio"]["status"] == "SILENT_AUDIO_REJECTED":
                        counts["audio_rejected_silent"] += 1

                except Exception as e:
                    logger.error("Error extracting features for video %s: %s", vid, e)
                    counts["failed_videos"] += 1
                    results.append({
                        "video_id": vid,
                        "error": str(e),
                        "total_features_generated": 0,
                    })

        counts["duration_seconds"] = round(time.perf_counter() - t0, 4)
        return {
            "counts": counts,
            "results": results,
        }
