# VECTOR RECOMMENDATION ENGINE — PHASE 9 SPECIFICATION

**Experiment ID:** `EXP_MICROLENS_TEXT_COENGAGEMENT_V2`  
**Retrieval Mode:** `TEXT_COENGAGEMENT_V2` (Text-Only)  
**Parent Phases:** Phase 8B / Phase 8C  
**Status:** PASS  

---

## 1. Executive Summary

Phase 9 implements a production-grade, vector-based candidate retrieval and ranking engine for the Multimodal YouTube Recommendation System.
The recommendation system uses the verified, trained text-only co-engagement representations from Phase 8C (`LinearProjectionModel(768, 512)` with frozen BERT-base backbone).

> [!IMPORTANT]
> **Text-Only Scope Notice:** The currently deployed recommendation model operates in `TEXT_COENGAGEMENT_V2` mode. Untrained audio and vision projection heads are NOT active in this retrieval path. Multimodal fusion remains blocked until aligned media or pre-extracted features are available.

---

## 2. Retrieval Architecture

The recommendation engine is organized into distinct, modular services:

```
[API Endpoints]
   ├── GET /api/recommendations/health
   ├── GET /api/recommendations/video/{video_id}?limit=10
   └── POST /api/recommendations/similar
            │
            ▼
[RecommendationService] (src/services/recommendation_service.py)
   ├── Embedding Generation: Frozen BERT [CLS] (768-D) + Phase 8C LinearProjection (512-D)
   ├── Query Validation & Normalization (L2 norm = 1.0)
   ├── Metadata Filtering: Pre-filtering & post-filtering (category, channel, seen videos)
   └── Candidate Hydration: Batch database resolution (no N+1 queries)
            │
            ▼
[VectorIndexService] (src/services/vector_index_service.py)
   ├── Exact In-Memory Baseline: NumPy / Tensor dot-product
   ├── pgvector Integration: Cosine distance (<=>) with HNSW indexing
   └── Automatic Fallback: Seamless exact cosine search when pgvector is unavailable
            │
            ▼
[VideoRepository] (src/db/repository.py)
   ├── `find_similar_with_filters()`
   ├── `get_by_video_ids()` (batch hydration)
   └── `upsert_embedding()` (idempotent persistence)
```

---

## 3. Phase 8C Checkpoint Integration

- **Checkpoint Path:** `data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt`
- **Verified SHA-256 Digest:** `a82037eb790e2a23e133b8edc9973a0e6fdc3c466058abd61b68aed77d2d4996`
- **Architecture:** `LinearProjectionModel(input_dim=768, output_dim=512, normalize=True)`
  - Weights: `linear.weight` [512, 768]
  - Bias: `linear.bias` [512]
- **Loading Function:** `src.models.text_projection_model.load_phase_8c_projection`
  - Validates file existence and state dict structure.
  - Freezes weights and enforces `eval()` mode.
  - Guaranteed offline execution without remote network calls.

---

## 4. Vector Storage & Indexing Strategy

1. **Storage Schema:**
   - Table: `videos`
   - Primary Key: `id` (UUID)
   - External Key: `video_id` (VARCHAR(255), unique, indexed)
   - Vector Column: `embedding` (`CompatibleEmbedding(512)`: native `Vector(512)` on PostgreSQL with pgvector, JSON / Float Array on SQLite / fallback)
2. **Indexing:**
   - PostgreSQL HNSW: `ix_video_embedding` on `Video.embedding` with `vector_cosine_ops` (`m=16, ef_construction=64`).
   - SQLite / Local Development Fallback: In-memory exact cosine similarity search over normalized embeddings.
3. **Exact Baseline Guarantee:**
   - Exact dot-product search provides the correctness baseline.
   - Self-match is strictly excluded from candidate pools (`video_id != query_video_id`).

---

## 5. API Endpoints

### A. Health & Subsystem Readiness
- **Route:** `GET /api/recommendations/health`
- **Response Model:** `RecommendationHealthResponse`
- **Output Sample:**
```json
{
  "status": "ok",
  "retrieval_mode": "TEXT_COENGAGEMENT_V2",
  "checkpoint_loaded": true,
  "checkpoint_sha256": "a82037eb790e2a23e133b8edc9973a0e6fdc3c466058abd61b68aed77d2d4996",
  "vector_backend": "exact_fallback",
  "pgvector_available": false,
  "total_indexed_videos": 2960
}
```

### B. Video Recommendations
- **Route:** `GET /api/recommendations/video/{video_id}?limit=10&category_id=10`
- **Response Model:** `RecommendationResponse`
- **Features:** Parameterized top-K (bounded 1 to 50), optional category and channel filters.

### C. Similar Videos with Exclusions
- **Route:** `POST /api/recommendations/similar`
- **Request Model:** `SimilarVideosRequest` (`video_id`, `top_k`, `exclude_video_ids`, `category_id`, `channel_id`)
- **Response Model:** `RecommendationResponse`

---

## 6. Backfill Procedure

Command-line backfill tool:
```bash
# Backfill from MicroLens dataset titles
python scripts/build_vector_index.py --from-csv data/raw/microlens/MicroLens-50k_titles.csv --batch-size 128

# Dry-run test mode
python scripts/build_vector_index.py --dry-run --limit 5 --from-csv data/raw/microlens/MicroLens-50k_titles.csv
```
Features:
- Idempotent upsert via `VideoRepository.upsert_embedding()`.
- Resume capability (`--skip-existing`).
- Safe batch processing preventing out-of-memory errors.

---

## 7. Retrieval Performance & Benchmarks

Evaluated on the held-out test split (188 valid test queries, 2,960 candidate gallery):

| Metric | Measured Value | Baseline (Raw BERT) | Delta |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **0.0213** | 0.0213 | 0.0% |
| **Recall@5** | **0.0479** | 0.0426 | **+12.4%** |
| **Recall@10** | **0.0585** | 0.0479 | **+22.1%** |
| **MRR** | **0.0308** | 0.0333 | -7.5% |
| **NDCG@10** | **0.0373** | 0.0345 | **+8.1%** |

### Latency Benchmark (2,960 Candidates)
- Mean Latency: **63.29 ms** / query
- p50 Latency: **62.51 ms** / query
- p95 Latency: **75.78 ms** / query
- p99 Latency: **82.14 ms** / query
- Embedding Coverage: **100.0%**
- Missing Metadata: **0**
- Invalid Embeddings: **0**

---

## 8. Limitations & Scope Boundary
1. **Text-Only Signal:** Recommendations are generated purely from video titles. Transcripts, audio, and visual features are not yet fused into this retrieval representation.
2. **Cold Start:** Videos without titles cannot generate embeddings.
3. **Scope Boundary:** Phase 9 implements candidate retrieval and filtering only. Personalized user session histories and persistent user profiles belong to Phase 10.
