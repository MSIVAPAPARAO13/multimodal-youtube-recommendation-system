# Multimodal Feature Architecture and Data Contract

## 1. Overview and Purpose

Phase 12B establishes the multimodal feature extraction and storage layer for the Multimodal YouTube Recommendation System.
It extracts, validates, normalizes, and stores visual and acoustic features derived from genuine physical media assets ingested in Phase 12A.

Every feature vector persisted in PostgreSQL (`video_features` table) is accompanied by rigorous provenance metadata:
- Canonical internal video ID (`video_id`).
- External YouTube ID (`youtube_video_id`).
- Source asset reference and SHA-256 checksum.
- Modality and feature type.
- Model and encoder identity with checkpoint reference.
- Explicit dimension and normalization contract.
- Training provenance (`is_trained=True` for ImageNet-1K pretrained weights vs. `is_trained=False` for untrained projections).
- Validation status and invalidation flag (`is_stale`).

---

## 2. Model & Checkpoint Provenance

| Modality | Encoder Name | Version | Checkpoint ID | Raw Dim | Stored Dim | Training Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Visual Frames** | `torchvision_resnet50` | `IMAGENET1K_V1` | `torch_hub_resnet50-0676ba61` | 2048 | 2048 | **Pretrained** (ImageNet-1K) |
| **Visual Frames** | `torchvision_resnet50_projected` | `1.0.0` | `untrained_projection_head` | 2048 | 512 | **Untrained Projection** |
| **Visual Thumbnail** | `torchvision_resnet50` | `IMAGENET1K_V1` | `torch_hub_resnet50-0676ba61` | 2048 | 2048 | **Pretrained** (ImageNet-1K) |
| **Visual Thumbnail** | `torchvision_resnet50_projected` | `1.0.0` | `untrained_projection_head` | 2048 | 512 | **Untrained Projection** |
| **CLIP Thumbnail** | `clip_vision_base` | `1.0.0` | `offline_clip_architecture` | 768 | 512 | **Untrained Architecture** |
| **Audio Waveform** | `facebook_wav2vec2_base` | `1.0.0` | `wav2vec2_architecture_eval` | 768 | 512 | **Untrained Projection** |

> [!IMPORTANT]
> **Strict Space Separation:**  
> - ResNet-50 2048-D features reside in ImageNet semantic space.
> - ResNet-50 512-D features are generated via a non-linear projection (`2048 -> ReLU -> Dropout -> 512`) whose weights have **not** been contrastively fine-tuned for recommendation.
> - Text embeddings (512-D) reside in the Phase 8C co-engagement space (`a82037eb...`).
> - **These spaces are not directly comparable.** Cross-modal cosine similarities between raw text user profiles and visual embeddings are mathematically invalid and strictly disabled.

---

## 3. Preprocessing, Pooling, and Normalization Contracts

### Visual Frames Preprocessing (`VisualProcessor`)
1. **Input**: Exactly 8 deterministic JPEG video frames ($t_0 \dots t_7$) sampled uniformly across video duration ($t = \text{frame\_idx} / \text{fps}$).
2. **Resize**: Resized to $(224, 224)$ pixels in RGB color space.
3. **Normalization**: Canonical ImageNet mean and standard deviation:
   $$\text{mean} = [0.485, 0.456, 0.406], \quad \text{std} = [0.229, 0.224, 0.225]$$
4. **Backbone Inference**: Evaluation mode (`torch.no_grad()`), frozen backbone.
5. **Frame Aggregation**: Mean-pooling across the 8 frame representations:
   $$\mathbf{v}_{\text{raw}} = \frac{1}{8} \sum_{i=0}^7 \mathbf{f}_i \in \mathbb{R}^{2048}$$
6. **L2 Normalization**:
   $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

### Thumbnail Preprocessing (`ThumbnailEncoder`)
1. **Input**: Local `thumbnail.jpg` asset.
2. **Preprocessing**: Identical ImageNet RGB resize $(224, 224)$ and standardization.
3. **Features**: 2048-D backbone feature and 512-D projection, both unit L2-normalized.

### Audio Preprocessing (`AudioProcessor`)
1. **Input**: 16 kHz mono PCM WAV format.
2. **Signal Integrity**: Validates nonempty waveform, finiteness, duration, and computes RMS energy.
3. **Zero-Signal Rejection**: Waveforms with pure silence ($RMS = 0.0$ or silence ratio $\ge 0.99$) are **rejected without fabricating vectors**.

---

## 4. Storage Architecture (`video_features`)

The feature storage is decoupled from the `videos` table to preserve the existing Phase 9 HNSW index on `videos.embedding` without destructive schema migrations.

```sql
CREATE TABLE video_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id VARCHAR(255) NOT NULL,
    youtube_video_id VARCHAR(64),
    asset_id UUID,
    modality VARCHAR(32) NOT NULL,
    feature_type VARCHAR(64) NOT NULL,
    encoder_name VARCHAR(64) NOT NULL,
    encoder_version VARCHAR(64) NOT NULL,
    checkpoint_id VARCHAR(128),
    dimension INTEGER NOT NULL,
    preprocessing_config JSONB,
    source_checksum VARCHAR(64) NOT NULL,
    embedding JSONB NOT NULL,
    is_valid BOOLEAN NOT NULL DEFAULT TRUE,
    is_stale BOOLEAN NOT NULL DEFAULT FALSE,
    is_trained BOOLEAN NOT NULL DEFAULT FALSE,
    validation_status VARCHAR(32) NOT NULL DEFAULT 'VALID',
    pipeline_version VARCHAR(32) NOT NULL DEFAULT '12B',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX ix_video_feature_lookup ON video_features(video_id, modality, feature_type, encoder_name);
CREATE INDEX ix_video_features_source_checksum ON video_features(source_checksum);
CREATE INDEX ix_video_features_modality ON video_features(modality);
```

### Invalidation Policy:
If a video asset on disk is re-ingested or modified, its SHA-256 checksum changes.
`VideoFeatureRepository.invalidate_stale_features(video_id, modality, current_checksum)` detects discrepancies, marks stale features with `is_stale=True` and `validation_status="STALE"`, preventing stale features from being served.

---

## 5. Re-Ranking Integration Policy

1. **Text Baseline Remains Default**: Phase 11B candidate re-ranking uses `w_text=0.70`, `w_metadata=0.10` with missing visual/audio weights dynamically renormalized.
2. **Zero-Weight Safety**: Visual weights remain inactive (`w_visual=0.0`) in production until Phase 12C establishes a validated multimodal query representation.
3. **Candidate Feature Hydration**: Candidate visual representations are hydrated on-demand via `CandidateReRankingService.hydrate_candidate_features`.

---

## 6. Reproduction Commands

```bash
# 1. Run dry-run audit
python scripts/build_multimodal_features.py --dry-run

# 2. Extract features across all catalog videos
python scripts/build_multimodal_features.py --batch-size 10 --force

# 3. Generate coverage audit report
python scripts/audit_multimodal_feature_coverage.py

# 4. Run Phase 12B test suite
pytest tests/test_multimodal_features_12b.py -v
```
