# Cross-Modal Alignment & Visual Retrieval Specification

## Milestone: Phase 12C — Controlled Multimodal Alignment & Retrieval Engine
**Experiment ID:** `EXP_MULTIMODAL_ALIGNMENT_12C`  
**Status:** **PASS** (Infrastructure & Retrieval) / **PARTIAL_EXPLORATORY** (Alignment Generalization)  
**Trained Baseline Maintained:** Phase 8C Co-Engagement Checkpoint (`data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt`, SHA-256: `a82037eb790e2a23e133b8edc9973a0e6fdc3c466058abd61b68aed77d2d4996`)

---

## 1. Executive Summary & Scientific Baseline

Phase 12C constructs a scientifically defensible bridge between the existing 512-dimensional text-based recommendation space (Phase 8C–11B) and the visual feature spaces generated from verified media assets in Phase 12B.

### Key Findings
1. **Visual Video-to-Video Retrieval (Experiment A):** Unsupervised cosine similarity on 2048-dimensional ResNet-50 mean-pooled frame representations exhibits high intra-category semantic cohesion (mean cosine similarity: **0.9990**) versus inter-category similarity (**0.5403**), yielding a clean separation margin of **+0.4587** and 100% Top-1 intra-category retrieval.
2. **Cross-Modal Alignment (Experiment B):** Closed-form Ridge Regression ($2048 \to 512$) fitted on 11 video-disjoint training pairs achieves **0.7910** mean cosine similarity on held-out test videos, substantially exceeding random projection (**-0.0387**) and untrained projection heads (**0.0469**).
3. **Generalization Boundary ($N=17$):** Because only 17 physical media-backed videos exist in the catalog, this learned projection is classified strictly as an **exploratory proof-of-concept**. Overfitting cannot be statistically ruled out with $N_{\text{train}}=11$, and generalization to an arbitrary catalog cannot be claimed.
4. **Production Safeguards:** Cross-modal visual ranking weights ($w_{\text{visual}} = 0.0$) and audio weights ($w_{\text{audio}} = 0.0$) remain **disabled by default** in production recommendations. Text-based ANN candidate retrieval (Phase 9/11A) and recency-weighted user personalization (Phase 10) remain the active production drivers.

---

## 2. Feature Provenance & Modality Inventory

| Modality | Feature Type | Dimension | Encoder Identity | Checkpoint / Source | Persisted Count | Status |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **VISUAL_FRAMES** | `resnet50_mean_2048` | 2048 | ResNet-50 (ImageNet-1K) | `torch_hub_resnet50-0676ba61` | 17 | Verified |
| **VISUAL_FRAMES** | `resnet50_projected_512` | 512 | ResNet-50 + Untrained Head | Untrained Linear Head | 17 | Exploratory |
| **VISUAL_THUMBNAIL** | `resnet50_thumbnail_2048`| 2048 | ResNet-50 (ImageNet-1K) | `torch_hub_resnet50-0676ba61` | 17 | Verified |
| **VISUAL_THUMBNAIL** | `resnet50_thumbnail_512` | 512 | ResNet-50 + Untrained Head | Untrained Linear Head | 17 | Exploratory |
| **CLIP_THUMBNAIL** | `clip_thumbnail_512` | 512 | CLIP ViT-B/32 (Vision) | `openai/clip-vit-base-patch32` | 17 | Verified |
| **AUDIO** | `wav2vec2_mean_512` | 512 | Wav2Vec2-Base-960h | `facebook/wav2vec2-base-960h` | 0 | **Rejected (Silent)** |
| **TEXT** | `bert_coengagement_512`| 512 | BERT-base + Phase 8C Head | `EXP_MICROLENS_TEXT_COENGAGEMENT_V2` | 17 (paired) | Verified |

> [!NOTE]
> **Audio Modality Status:** The physical audio assets in the local catalog contain zero/silent signals due to environment-level absence of audio demuxers. Silent waveforms were explicitly rejected during Phase 12B without fabricating synthetic or fake embeddings. Audio re-ranking remains disabled ($w_{\text{audio}} = 0.0$).

---

## 3. Experiment A — Direct Visual Video-to-Video Retrieval

### Objective
Establish an unsupervised visual similarity baseline using compatible raw ResNet-50 2048-D features (`resnet50_mean_2048`) across the 17 catalog videos.

### Methodology
1. **Query & Candidate Space:** Both query and candidate vectors originate from the identical encoder (`torchvision.models.resnet50`) and feature type (`resnet50_mean_2048`).
2. **Pre-processing:** All vectors are validated for finiteness ($\text{NaN}/\text{Inf}$ checks) and strictly normalized to the unit hypersphere ($\|v\|_2 = 1.0$).
3. **Similarity Metric:** Cosine similarity bounded in $[-1.0, 1.0]$:
   $$\text{sim}(q, c) = \frac{q \cdot c}{\|q\|_2 \|c\|_2} = q \cdot c$$
4. **Query Exclusion:** The query video is explicitly filtered from its candidate list.
5. **Deterministic Tie-Breaking:** Candidate ranking sorts primarily by score descending, with deterministic secondary ordering on `video_id` ascending.

### Empirical Results (17 Videos, 5 Synthetic Categories + MicroLens Items)
- **Mean Intra-Category Cosine Similarity:** `0.9990`
- **Mean Inter-Category Cosine Similarity:** `0.5403`
- **Separation Margin:** `+0.4587`
- **Top-1 Intra-Category Retrieval Rate:** `1.000` (100% of synthetic category queries retrieve a fellow category video at rank 1)
- **MicroLens Items (`1002`, `10003`):** Demonstrate distinct geometric/visual characteristics with cross-category similarity averaging `0.4812`.

---

## 4. Experiment B — Cross-Modal Alignment

### Objective
Learn a regularized linear mapping $W \in \mathbb{R}^{2048 \times 512}$ that maps raw visual features into the Phase 8C text co-engagement embedding space.

### Dataset & Partition Integrity
The paired dataset links each video's ResNet-50 2048-D frame representation with its canonical Phase 8C text embedding:
- $N_{\text{total}} = 17$ verified pairs.
- **Strict Video-Level Partition:** Partitions are assigned by distinct video identity, guaranteeing **zero frame leakage** across splits.

```
+-----------------------------------------------------------------------------+
|                                17 Catalog Videos                            |
+--------------------------+-----------------------+--------------------------+
|  Train Split (N = 11)    |  Val Split (N = 3)    |  Held-out Test (N = 3)   |
|  - 2 per category (0, 1) |  - 1 per cat (idx 2)  |  - 1 per cat (idx 2)     |
|  - MicroLens '1002'      |    [prog, cyber, ML]  |    [web_dev, cloud]      |
|                          |                       |  - MicroLens '10003'     |
+--------------------------+-----------------------+--------------------------+
```

### Alignment Formulation: Closed-Form Ridge Projection
To prevent stochastic optimization artifacts on small samples, the projection is solved via closed-form Ridge Regression:
$$W^* = (X_{\text{train}}^T X_{\text{train}} + \alpha I)^{-1} X_{\text{train}}^T Y_{\text{train}}$$
$$\hat{y} = \frac{x W^*}{\|x W^*\|_2}$$

### Hyperparameter Tuning (Validation-Only)
Candidate regularization parameters $\alpha \in \{10^{-4}, 10^{-2}, 0.1, 1.0, 10.0, 100.0, 1000.0\}$ were evaluated solely against $X_{\text{val}}, Y_{\text{val}}$:
- **Optimal Regularizer:** $\alpha^* = 0.01$
- **Validation Cosine Similarity:** `0.9601`

### Held-Out Test Evaluation (Evaluated Once)
The fitted projector was evaluated once on the strictly held-out test split:

| Method / Model | Test Mean Cosine Sim | Test Top-1 Accuracy | Gain over Random |
| :--- | :---: | :---: | :---: |
| **Random Projection Baseline ($W \sim \mathcal{N}(0, 1)$)** | -0.0387 | 0.00% | — |
| **Untrained Head Baseline (`resnet50_projected_512`)** | 0.0469 | 33.33% | +0.0856 |
| **Learned Ridge Projector ($2048 \to 512$, $\alpha=0.01$)** | **0.7910** | **66.67%** | **+0.8298** |

> [!WARNING]
> **Exploratory Status:** While the aligned projection demonstrates strong in-sample and validation fidelity, $N=17$ is an exploratory dataset. Under Rule 10 & 12, this alignment model is classified as **PARTIAL_EXPLORATORY** and cannot be promoted to production ranking.

---

## 5. Experiment C — User Personalization Compatibility

Phase 10 user personalization constructs profile vectors $\mathbf{p}_u \in \mathbb{R}^{512}$ via recency-weighted decay over historical video interactions in the **Phase 8C text embedding space**.

### Mathematical Incompatibility Audit
1. **Raw Visual Vectors (2048-D):**
   $$\langle \mathbf{p}_u, \mathbf{v}_{\text{cand}} \rangle \implies \text{UNDEFINED (Dimension Mismatch: 512 vs 2048)}$$
2. **Pretrained CLIP Vectors (512-D):**
   $$\langle \mathbf{p}_u, \mathbf{v}_{\text{CLIP}} \rangle \implies \text{SEMANTIC INCOHERENCE}$$
   Equal dimensionality does not imply semantic alignment. CLIP visual embeddings inhabit OpenAI's multimodal space, while user profiles exist in Phase 8C's co-engagement projection. Unaligned cosine comparison yields arbitrary noise ($\approx 0.03$ mean similarity).

### Production Re-Ranking Safeguards
To safeguard recommendation quality, production candidate re-ranking applies **safe baseline weights**:
$$\mathbf{w}_{\text{prod}} = [w_{\text{text}}=0.90, w_{\text{visual}}=0.00, w_{\text{audio}}=0.00, w_{\text{metadata}}=0.10]$$
- If visual candidates or queries are provided experimentally, missing-modality dynamic weight renormalization guarantees that unpopulated visual signals collapse gracefully back to text and metadata.

---

## 6. Architecture & Implementation Optimizations

### 1. Batch Feature Hydration (N+1 Query Elimination)
Previous implementations loaded candidate visual features in an iterative loop. Phase 12C adds `VideoFeatureRepository.get_features_by_video_ids()` to hydrate candidate vectors in a single SQL query:
```python
stmt = select(VideoFeature).where(
    VideoFeature.video_id.in_(candidate_ids),
    VideoFeature.modality == modality,
    VideoFeature.feature_type == feature_type,
    VideoFeature.is_valid.is_(True),
    VideoFeature.is_stale.is_(False),
)
```

### 2. Dedicated Visual Retrieval Endpoint
A separate endpoint is exposed for exploratory visual nearest-neighbor search, preventing cross-contamination with production recommendation defaults:
- **Route:** `GET /api/recommendations/visual/{video_id}?limit=5&feature_type=resnet50_mean_2048`
- **Response Schema:** `VisualRetrievalResponse` with deterministic ranking and query exclusion.

---

## 7. Reproduction Commands

```bash
# 1. Execute Controlled Multimodal Evaluation Script
python scripts/evaluate_cross_modal_alignment.py

# 2. Run Phase 12C Targeted Test Suite
pytest tests/test_multimodal_alignment_12c.py -v

# 3. Run Full Test Suite
pytest tests/ -q
```
