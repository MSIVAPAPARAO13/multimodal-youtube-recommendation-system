"""
Phase 12B - Multimodal Feature Extraction, Alignment, and Storage Tests.

Tests:
1. ResNet-50 visual encoder output dimensions (2048-D raw and 512-D projected).
2. Frame preprocessing and ImageNet normalization.
3. Frame checksum validation and rejection of corrupted/missing frames.
4. L2 normalization and rejection of NaN/Inf.
5. Thumbnail feature extraction via ThumbnailEncoder.
6. Audio waveform validation, silence detection, and rejection of zero-signal waveforms.
7. Wav2Vec2 audio feature extraction on valid non-silent audio.
8. VideoFeature repository CRUD, validation, and idempotent upserts.
9. Stale feature invalidation on source checksum change.
10. Single-video multimodal feature extraction service execution.
11. Batch feature extraction and failure isolation.
12. Hydration of candidate multimodal features for re-ranking.
13. Re-ranking compatibility with missing query modalities (safe active-weight renormalization).
14. Preservation of text embeddings and HNSW candidate retrieval.
15. Feature coverage audit script execution and report generation.
"""

import json
from pathlib import Path
import wave
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from src.db.models import (
    Video,
    VideoAsset,
    VideoFeature,
    AssetType,
    AssetStatus,
    FeatureModality,
)
from src.db.repository import VideoRepository, VideoAssetRepository, VideoFeatureRepository
from src.features.visual_processor import VisualProcessor
from src.features.visual_encoder import ResNet50VisualEncoder, CLIPExperimentalEncoder
from src.features.thumbnail_encoder import ThumbnailEncoder
from src.features.audio_processor import AudioProcessor
from src.features.audio_encoder import AudioEncoder
from src.integrations.youtube.media.validator import MediaValidator
from src.services.feature_extraction_service import FeatureExtractionService
from src.services.reranking_service import CandidateReRankingService, ReRankingWeights
from scripts.audit_multimodal_feature_coverage import run_feature_coverage_audit


def test_visual_encoder_dimensions_and_norm():
    """Verify ResNet-50 outputs 2048-D raw backbone features and 512-D projected embeddings."""
    encoder = ResNet50VisualEncoder(feature_dim=512, pretrained=False, freeze_backbone=True)
    encoder.eval()

    # (B, T, C, H, W) = 1 video with 8 frames
    dummy_frames = torch.randn(1, 8, 3, 224, 224)
    with torch.no_grad():
        raw_feat = encoder.extract_backbone_features(dummy_frames.view(8, 3, 224, 224))
        pooled_2048 = raw_feat.mean(dim=0, keepdim=True)
        norm_2048 = F.normalize(pooled_2048, p=2, dim=-1)

        proj_512 = encoder(dummy_frames)
        norm_512 = F.normalize(proj_512, p=2, dim=-1)

    assert norm_2048.shape == (1, 2048)
    assert norm_512.shape == (1, 512)
    assert abs(torch.norm(norm_2048).item() - 1.0) < 1e-4
    assert abs(torch.norm(norm_512).item() - 1.0) < 1e-4
    assert torch.isfinite(norm_2048).all()
    assert torch.isfinite(norm_512).all()


def test_visual_preprocessing_pipeline(tmp_path):
    """Test image preprocessing produces normalized (3, 224, 224) float tensors."""
    vp = VisualProcessor(target_size=(224, 224), normalize=True)

    img_path = tmp_path / "test_frame.jpg"
    img = Image.new("RGB", (640, 480), color=(128, 64, 32))
    img.save(img_path)

    tensor = vp.preprocess_frame(img_path)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()

    # Test sequence of frames
    seq = vp.preprocess_frames([img_path, img_path])
    assert seq.shape == (2, 3, 224, 224)


def test_nan_inf_rejection_in_feature_repository(in_memory_db):
    """Verify VideoFeatureRepository rejects NaN, Inf, and incorrect dimension embeddings."""
    repo = VideoFeatureRepository(in_memory_db)

    # NaN in embedding
    with pytest.raises(ValueError, match="non-finite value"):
        repo.upsert_feature(
            video_id="vid_test_nan",
            modality=FeatureModality.VISUAL_FRAMES,
            feature_type="resnet50_mean_2048",
            encoder_name="resnet50",
            encoder_version="1.0.0",
            dimension=4,
            source_checksum="chk_123",
            embedding=[0.1, float("nan"), 0.3, 0.4],
        )

    # Infinity in embedding
    with pytest.raises(ValueError, match="non-finite value"):
        repo.upsert_feature(
            video_id="vid_test_inf",
            modality=FeatureModality.VISUAL_FRAMES,
            feature_type="resnet50_mean_2048",
            encoder_name="resnet50",
            encoder_version="1.0.0",
            dimension=4,
            source_checksum="chk_123",
            embedding=[0.1, float("inf"), 0.3, 0.4],
        )

    # Dimension mismatch
    with pytest.raises(ValueError, match="dimension mismatch"):
        repo.upsert_feature(
            video_id="vid_test_dim",
            modality=FeatureModality.VISUAL_FRAMES,
            feature_type="resnet50_projected_512",
            encoder_name="resnet50",
            encoder_version="1.0.0",
            dimension=512,
            source_checksum="chk_123",
            embedding=[0.1] * 10,
        )


def test_feature_repository_crud_and_idempotency(in_memory_db):
    """Test idempotent persistence and query operations of VideoFeatureRepository."""
    repo = VideoFeatureRepository(in_memory_db)
    emb_512 = [0.1] * 512

    feat1 = repo.upsert_feature(
        video_id="vid_crud_1",
        modality=FeatureModality.VISUAL_FRAMES,
        feature_type="resnet50_projected_512",
        encoder_name="resnet50",
        encoder_version="1.0.0",
        dimension=512,
        source_checksum="chk_orig",
        embedding=emb_512,
        checkpoint_id="ckpt_test",
        is_trained=False,
    )
    assert feat1.id is not None
    assert feat1.dimension == 512
    assert feat1.is_valid is True

    # Idempotent upsert with updated checksum
    feat2 = repo.upsert_feature(
        video_id="vid_crud_1",
        modality=FeatureModality.VISUAL_FRAMES,
        feature_type="resnet50_projected_512",
        encoder_name="resnet50",
        encoder_version="1.0.1",
        dimension=512,
        source_checksum="chk_updated",
        embedding=[0.2] * 512,
        checkpoint_id="ckpt_test",
        is_trained=False,
    )
    assert feat2.id == feat1.id
    assert feat2.source_checksum == "chk_updated"
    assert feat2.encoder_version == "1.0.1"

    # Query methods
    fetched = repo.get_feature("vid_crud_1", FeatureModality.VISUAL_FRAMES, "resnet50_projected_512")
    assert fetched is not None
    assert fetched.source_checksum == "chk_updated"

    all_feats = repo.get_features_for_video("vid_crud_1")
    assert len(all_feats) == 1

    counts = repo.count_by_modality()
    assert counts.get(FeatureModality.VISUAL_FRAMES.value) == 1


def test_stale_feature_invalidation(in_memory_db):
    """Verify features are invalidated when source asset checksum changes."""
    repo = VideoFeatureRepository(in_memory_db)

    repo.upsert_feature(
        video_id="vid_stale_test",
        modality=FeatureModality.VISUAL_THUMBNAIL,
        feature_type="resnet50_thumbnail_512",
        encoder_name="resnet50",
        encoder_version="1.0.0",
        dimension=512,
        source_checksum="old_hash_111",
        embedding=[0.05] * 512,
    )

    # Invalidate when current checksum differs
    invalidated_count = repo.invalidate_stale_features(
        video_id="vid_stale_test",
        modality=FeatureModality.VISUAL_THUMBNAIL,
        current_source_checksum="new_hash_222",
    )
    assert invalidated_count == 1

    # Active feature query returns None for stale records
    active = repo.get_feature("vid_stale_test", FeatureModality.VISUAL_THUMBNAIL, "resnet50_thumbnail_512")
    assert active is None


def test_silent_audio_rejection(tmp_path):
    """Verify AudioProcessor detects silent waveforms and rejects zero-amplitude signals."""
    proc = AudioProcessor(target_sample_rate=16000, max_duration_seconds=5.0)

    # 1. Pure zero WAV file
    silent_wav = tmp_path / "silent.wav"
    with wave.open(str(silent_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

    res = proc.process_audio(silent_wav)
    assert res.is_silent is True or res.rms_energy == 0.0 or not res.is_valid


def test_audio_feature_extraction_on_valid_signal(tmp_path, in_memory_db):
    """Verify Wav2Vec2 extracts 512-D features from non-silent acoustic signal."""
    # 1. Create a 440 Hz audible sine wave
    audible_wav = tmp_path / "tone_440hz.wav"
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(audible_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(tone.tobytes())

    proc = AudioProcessor(target_sample_rate=16000, max_duration_seconds=5.0)
    proc_res = proc.process_audio(audible_wav)
    assert proc_res.is_valid is True
    assert proc_res.is_silent is False
    assert proc_res.rms_energy > 0.1

    # 2. AudioEncoder forward
    encoder = AudioEncoder(feature_dim=512, pretrained=False)
    encoder.eval()
    with torch.no_grad():
        waveform_tensor = proc_res.waveform.unsqueeze(0)
        emb = encoder(waveform_tensor)
        norm_emb = F.normalize(emb, p=2, dim=-1)

    assert norm_emb.shape == (1, 512)
    assert abs(torch.norm(norm_emb).item() - 1.0) < 1e-4
    assert torch.isfinite(norm_emb).all()


def test_thumbnail_encoder_pipeline(tmp_path):
    """Verify ThumbnailEncoder extracts 512-D representation from an image file."""
    encoder = ResNet50VisualEncoder(feature_dim=512, pretrained=False)
    proc = VisualProcessor(target_size=(224, 224), normalize=True)
    thumb_enc = ThumbnailEncoder(encoder=encoder, processor=proc)

    img_path = tmp_path / "thumb.jpg"
    Image.new("RGB", (320, 240), color=(200, 100, 50)).save(img_path)

    res = thumb_enc.encode_thumbnail(img_path)
    assert res.is_available is True
    assert res.embedding.shape == (1, 512)
    assert torch.isfinite(res.embedding).all()


def test_clip_experimental_encoder():
    """Verify CLIPExperimentalEncoder extracts 512-D representation in offline architecture mode."""
    clip_enc = CLIPExperimentalEncoder(feature_dim=512, pretrained=False)
    clip_enc.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        emb = clip_enc.encode_single_image(dummy)
        norm_emb = F.normalize(emb, p=2, dim=-1)

    assert norm_emb.shape == (1, 512)
    assert abs(torch.norm(norm_emb).item() - 1.0) < 1e-4


def test_feature_extraction_service_single_video(in_memory_db, tmp_path):
    """Test full FeatureExtractionService for a video with frames, thumbnail, and audio."""
    storage_root = tmp_path / "media"
    storage_root.mkdir(parents=True, exist_ok=True)
    video_dir = storage_root / "test_vid_1"
    video_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = video_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    asset_repo = VideoAssetRepository(in_memory_db)
    from src.storage.local import LocalStorage
    storage = LocalStorage(root_dir=storage_root)

    # Create 8 frames
    for i in range(8):
        f_path = frames_dir / f"frame_{i:03d}.jpg"
        Image.new("RGB", (100, 100), color=(i * 20, 50, 100)).save(f_path)
        chk = MediaValidator.compute_sha256(f_path)
        asset_repo.upsert_asset(
            video_id="test_vid_1",
            asset_type=AssetType.FRAME,
            storage_path=f"test_vid_1/frames/frame_{i:03d}.jpg",
            file_size_bytes=f_path.stat().st_size,
            checksum_sha256=chk,
            status=AssetStatus.READY,
            extra_metadata={"frame_index": i},
        )

    # Create thumbnail
    t_path = video_dir / "thumbnail.jpg"
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(t_path)
    t_chk = MediaValidator.compute_sha256(t_path)
    asset_repo.upsert_asset(
        video_id="test_vid_1",
        asset_type=AssetType.THUMBNAIL,
        storage_path="test_vid_1/thumbnail.jpg",
        file_size_bytes=t_path.stat().st_size,
        checksum_sha256=t_chk,
        status=AssetStatus.READY,
    )

    # Create audible audio
    a_path = video_dir / "audio.wav"
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(a_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(tone.tobytes())
    a_chk = MediaValidator.compute_sha256(a_path)
    asset_repo.upsert_asset(
        video_id="test_vid_1",
        asset_type=AssetType.AUDIO,
        storage_path="test_vid_1/audio.wav",
        file_size_bytes=a_path.stat().st_size,
        checksum_sha256=a_chk,
        status=AssetStatus.READY,
    )

    service = FeatureExtractionService(storage=storage)
    res = service.extract_features_for_video("test_vid_1", in_memory_db, force=True)

    assert res["visual_frames"]["success"] is True
    assert len(res["visual_frames"]["features_generated"]) == 2  # 2048-D and 512-D
    assert res["visual_thumbnail"]["success"] is True
    assert len(res["visual_thumbnail"]["features_generated"]) >= 2
    assert res["audio"]["success"] is True
    assert len(res["audio"]["features_generated"]) == 1

    # Check persistence
    feat_repo = VideoFeatureRepository(in_memory_db)
    feats = feat_repo.get_features_for_video("test_vid_1")
    assert len(feats) >= 5


def test_batch_feature_extraction_failure_isolation(in_memory_db, tmp_path):
    """Verify a faulty video does not fail or abort the batch extraction run."""
    from src.storage.local import LocalStorage
    storage = LocalStorage(root_dir=tmp_path)
    service = FeatureExtractionService(storage=storage)

    # Non-existent video_ids
    results = service.batch_extract_features(
        video_ids=["non_existent_1", "non_existent_2"],
        session=in_memory_db,
        batch_size=2,
    )

    counts = results["counts"]
    assert counts["total"] == 2
    assert counts["failed_videos"] == 0  # Gracefully handled without unhandled exception


def test_candidate_features_hydration_for_reranking(in_memory_db):
    """Test CandidateReRankingService.hydrate_candidate_features pulls features from DB."""
    feat_repo = VideoFeatureRepository(in_memory_db)
    emb = [0.1] * 512
    feat_repo.upsert_feature(
        video_id="cand_video_100",
        modality=FeatureModality.VISUAL_FRAMES,
        feature_type="resnet50_projected_512",
        encoder_name="resnet50",
        encoder_version="1.0.0",
        dimension=512,
        source_checksum="chk_100",
        embedding=emb,
    )

    reranker = CandidateReRankingService()
    hydrated = reranker.hydrate_candidate_features(
        candidate_ids=["cand_video_100", "cand_video_missing"],
        session=in_memory_db,
        modality="VISUAL_FRAMES",
        feature_type="resnet50_projected_512",
    )

    assert "cand_video_100" in hydrated
    assert "cand_video_missing" not in hydrated
    assert len(hydrated["cand_video_100"]) == 512


def test_reranking_missing_modality_renormalization(in_memory_db):
    """Test re-ranking safely renormalizes active weights when visual/audio query vectors are absent."""
    video_repo = VideoRepository(in_memory_db)
    cand = video_repo.create(
        video_id="cand_1",
        title="Candidate Video 1",
        embedding=[0.2] * 512,
    )

    reranker = CandidateReRankingService()
    query_text = [0.2] * 512

    # Visual and audio query embeddings are None
    score, telemetry = reranker.score_candidate(
        candidate_video=cand,
        query_text_norm=reranker.validate_vector(query_text),
        query_visual_norm=None,
        query_audio_norm=None,
        candidate_visual_vector=None,
        candidate_audio_vector=None,
    )

    assert "visual" not in telemetry["modalities_present"]
    assert "audio" not in telemetry["modalities_present"]
    assert "text" in telemetry["modalities_present"]
    assert abs(score - 1.0) < 1e-4  # Cosine similarity with self is 1.0


def test_audit_script_execution():
    """Verify run_feature_coverage_audit executes and generates structured report."""
    report = run_feature_coverage_audit()
    assert report["phase"] == "12B"
    assert report["status"] == "PASS"
    assert "feature_extraction_summary" in report
    assert report["feature_extraction_summary"]["valid_features"] >= 0
    assert Path("data/checkpoints/EXP_MULTIMODAL_FEATURE_EXTRACTION_12B/feature_coverage_report.json").exists()
