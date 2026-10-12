# Multimodal Candidate Re-Ranking Architecture

**Phase:** Phase 11B  
**Experiment ID:** `EXP_MICROLENS_MULTIMODAL_RERANK_V1`  
**Retrieval Representation:** `TEXT_COENGAGEMENT_V2` (512-D, Phase 8C checkpoint)  
**Status:** Validated and Production-Ready

---

## 1. Overview & Two-Stage Architecture

Phase 11B introduces a **Two-Stage Retrieval and Re-Ranking Pipeline** to optimize recommendation accuracy while keeping query latencies sub-20 ms.

```
+-----------------------------------------------------------------------------------+
|                        Client Request (Video Seed or User ID)                     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               Stage 1: High-Recall Candidate Generation (Phase 11A)               |
|                                                                                   |
|  - Queries PostgreSQL 16 + pgvector HNSW index (`ix_video_embedding`)            |
|  - Candidate pool size: K_cand = 50 (bounded, O(log N) retrieval)                |
|  - Excludes seen items and applies metadata pre-filters                           |
|  - Candidate generation latency: ~16–17 ms                                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                  Stage 2: Multi-Signal Re-Ranking (Phase 11B)                     |
|                                                                                   |
|  - Batch-scores K_cand = 50 candidates across active, validated signals           |
|  - Dynamic active-weight renormalization for missing modalities                   |
|  - Configurable weights: w_text, w_visual, w_audio, w_metadata                   |
|  - Deterministic tie-breaking by video_id                                         |
|  - Re-ranking latency: ~1.4 ms                                                    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                  Top-K Recommended Items (default K = 10)                         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Feature Availability & Modality Coverage

An audit was performed across the 2,960-item catalog to establish genuine data availability before designing ranking scores:

| Modality / Feature | Catalog Availability | Coverage (%) | Feature Space / Source | Status in Re-Ranking Engine |
|---|---|---|---|---|
| **Text Semantics** | 2,960 / 2,960 items | **100.0%** | 512-D `TEXT_COENGAGEMENT_V2` (Phase 8C) | **ACTIVE** (Primary relevance signal) |
| **Session Context / Recency** | 2,960 / 2,960 items | **100.0%** | Most recent historical interaction embedding | **ACTIVE** (Short-term intent signal) |
| **Historical Popularity** | 2,960 / 2,960 items | **100.0%** | Interaction frequency strictly prior to $t_{\text{cutoff}}$ | **ACTIVE** (Prior popularity signal) |
| **Thumbnail / Visual** | 10 / 2,960 items | **0.34%** | ResNet-50 512-D embeddings | **CONDITIONAL / RENORMALIZED** (Active when visual embeddings exist) |
| **Audio Waveform** | 0 / 2,960 items | **0.0%** | Wav2Vec2 512-D embeddings | **UNAVAILABLE / DEACTIVATED** (Excluded from active weights) |
| **Category / Channel Metadata** | 0 / 2,960 items | **0.0%** | MicroLens-50K contains titles only | **CONDITIONAL** (Active for YouTube API ingested videos) |

---

## 3. Re-Ranking Scoring Formulation

### Multi-Signal Formulation
For a candidate item $c$ and query representation $\mathbf{q}$, the candidate score is computed as:

$$\text{Score}(c) = \frac{\sum_{m \in A_c} w_m \cdot S_m(c)}{\sum_{m \in A_c} w_m}$$

where:
- $A_c$ is the set of **active modalities** present for candidate $c$ and query $\mathbf{q}$.
- $w_m \ge 0$ is the configured weight for modality $m$.
- $S_m(c)$ is the normalized relevance score for modality $m$.

### Individual Signal Definitions
1. **Text Semantic Relevance ($S_{\text{text}}$):**
   $$S_{\text{text}}(c) = \cos(\mathbf{q}_{\text{text}}, \mathbf{e}_c) = \frac{\mathbf{q}_{\text{text}} \cdot \mathbf{e}_c}{\|\mathbf{q}_{\text{text}}\|_2 \|\mathbf{e}_c\|_2} \in [-1.0, 1.0]$$
2. **Visual Relevance ($S_{\text{visual}}$):**
   Active if and only if both query visual embedding $\mathbf{q}_{\text{vis}}$ and candidate visual embedding $\mathbf{c}_{\text{vis}}$ exist:
   $$S_{\text{visual}}(c) = \cos(\mathbf{q}_{\text{vis}}, \mathbf{c}_{\text{vis}}) \in [-1.0, 1.0]$$
3. **Audio Relevance ($S_{\text{audio}}$):**
   Active if and only if both query and candidate audio embeddings exist:
   $$S_{\text{audio}}(c) = \cos(\mathbf{q}_{\text{aud}}, \mathbf{c}_{\text{aud}}) \in [-1.0, 1.0]$$
4. **Metadata Alignment ($S_{\text{meta}}$):**
   Active if target category or channel is provided:
   $$S_{\text{meta}}(c) = 0.6 \cdot \mathbb{I}(\text{cat}_c = \text{cat}_q) + 0.4 \cdot \mathbb{I}(\text{chan}_c = \text{chan}_q) \in [0.0, 1.0]$$

### Dynamic Active-Weight Renormalization
When visual or audio data is missing for a candidate, the missing modalities are excluded from $A_c$. The active weights are dynamically renormalized:
$$W_{\text{active}} = \sum_{m \in A_c} w_m$$
This prevents missing modalities from penalizing candidates or distorting score distributions.

---

## 4. Leakage-Free Temporal Protocol

To prevent future-data contamination during evaluation and serving:
1. **Timestamp Cutoff:** For evaluation query at time $t_{\text{cutoff}}$, user interaction history $\mathcal{H}$ is strictly restricted to:
   $$\mathcal{H} = \{ (v_i, t_i) \mid t_i < t_{\text{cutoff}} \}$$
2. **Popularity Calibration:** Item interaction frequency priors are computed strictly using events occurring prior to $t_{\text{cutoff}}$.
3. **Validation / Test Split:** Candidate re-ranking weights were tuned strictly on a 200-user validation split and evaluated **once** on the 300-user held-out test split.

---

## 5. API Integration & Configuration

### Configuration Settings (`src/db/config.py`)
```bash
RERANKING_ENABLED=true                # Global feature flag
RERANKING_DEFAULT_POOL_SIZE=50        # Stage 1 candidate generation count
RERANKING_WEIGHT_TEXT=0.70            # Text similarity weight
RERANKING_WEIGHT_VISUAL=0.15          # Visual similarity weight
RERANKING_WEIGHT_AUDIO=0.05           # Audio similarity weight
RERANKING_WEIGHT_METADATA=0.10        # Metadata match weight
```

### Supported API Query Parameters
Endpoints support opt-in re-ranking without changing baseline defaults:
- `ranking_mode`: `"baseline"` (default, Stage 1 direct ranking) or `"rerank"` (Stage 1 + Stage 2 re-ranking).
- `candidate_pool_size`: Integer between 10 and 200 (default 50).

Example:
```http
GET /api/recommendations/user/user_1024?ranking_mode=rerank&candidate_pool_size=50
```
