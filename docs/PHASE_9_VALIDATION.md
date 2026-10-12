# PHASE 9 — VALIDATION AND AUDIT REPORT

**Experiment ID:** `EXP_MICROLENS_TEXT_COENGAGEMENT_V2`  
**Retrieval Mode:** `TEXT_COENGAGEMENT_V2`  
**Date:** 2026-10-09  
**Overall Phase Status:** **PASS**

---

## 1. Implementation Status Matrix

| Component | Status | Details |
| :--- | :---: | :--- |
| **Exact Retrieval Baseline** | **COMPLETE** | In-memory exact cosine similarity search in `VectorIndexService.search_exact_in_memory()` with strict self-match exclusion. |
| **Database-Backed Retrieval** | **COMPLETE** | Parameterized queries with metadata pre-filtering (`category_id`, `channel_id`, `exclude_video_ids`) in `VideoRepository.find_similar_with_filters()`. |
| **Approximate Index Architecture** | **COMPLETE** | PostgreSQL HNSW index `ix_video_embedding` configured in `src/db/models.py`; automated exact fallback active when pgvector is absent. |
| **API Integration** | **COMPLETE** | Dedicated endpoints `/api/recommendations/health`, `/api/recommendations/video/{video_id}`, and `/api/recommendations/similar` in `src/api/app.py`. |
| **Embedding Backfill Tool** | **COMPLETE** | Restartable CLI backfill script `scripts/build_vector_index.py` supporting bounded batches, resume capability, and dry-run mode. |
| **Retrieval Evaluation** | **COMPLETE** | Automated benchmark `scripts/evaluate_vector_retrieval.py` tracking Recall@K, Precision@K, MRR, NDCG@10, and query latency percentiles. |

---

## 2. Checkpoint and Weight Integrity

- **Phase 8B Checkpoint:** `data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V1/best.pt`  
  SHA256: `430dc7c44febbaa11fe8671515f96afddd7b2af191454da2af954167a2cd9bf0` (Verified immutable & unchanged)
- **Phase 8C Checkpoint:** `data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt`  
  SHA256: `a82037eb790e2a23e133b8edc9973a0e6fdc3c466058abd61b68aed77d2d4996` (Verified loaded into `LinearProjectionModel`)
- **Compatibility Test:** Verified that `load_phase_8c_projection()` recreates exact parameter dimensions:
  - `linear.weight`: `torch.Size([512, 768])`
  - `linear.bias`: `torch.Size([512])`
  - Outputs 512-D L2-normalized vectors on unit sphere.

---

## 3. Retrieval Evaluation Results

Evaluated on the held-out test split (188 valid test queries, 2,960 candidate gallery):

| Metric | Raw Pretrained BERT | Phase 8C Vector Engine | Relative Change |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 0.0213 | 0.0213 | 0.0% |
| **Recall@5** | 0.0426 | 0.0479 | **+12.4%** |
| **Recall@10** | 0.0479 | 0.0585 | **+22.1%** |
| **Precision@1** | 0.0213 | 0.0213 | 0.0% |
| **Precision@5** | 0.0085 | 0.0096 | **+12.9%** |
| **Precision@10** | 0.0048 | 0.0059 | **+22.9%** |
| **MRR** | 0.0333 | 0.0308 | -7.5% |
| **NDCG@10** | 0.0345 | 0.0373 | **+8.1%** |

### Latency Profile
- **Mean Search Latency:** 63.293 ms
- **p50 Search Latency:** 62.510 ms
- **p95 Search Latency:** 75.779 ms
- **p99 Search Latency:** 82.140 ms

---

## 4. Test Suite Verification

- **Previous Baseline:** 217 passing tests
- **Phase 9 Tests Added:** 15 tests in `tests/test_vector_recommendation.py`
- **Total Passing Tests:** 232 passed
- **Failed Tests:** 0
- **Regressions:** 0 regressions across all phases (Phase 1–8C)

---

## 5. Artifact & File Summary

### Files Created:
1. `src/schemas/recommendations.py`: Validated Pydantic models for recommendation retrieval.
2. `src/services/vector_index_service.py`: Exact cosine baseline and pgvector candidate retrieval abstraction.
3. `src/services/recommendation_service.py`: Core domain service for text encoding, ranking, hydration, and filtering.
4. `scripts/build_vector_index.py`: Restartable embedding backfill command.
5. `scripts/evaluate_vector_retrieval.py`: Retrieval benchmarking script.
6. `tests/test_vector_recommendation.py`: 15 unit and integration tests.
7. `docs/VECTOR_RECOMMENDATION.md`: System specification and architecture documentation.
8. `docs/PHASE_9_VALIDATION.md`: Validation and audit report.

### Files Modified:
1. `src/models/text_projection_model.py`: Added `LinearProjectionModel` and `load_phase_8c_projection()`.
2. `src/db/config.py`: Added Phase 9 configuration settings.
3. `src/db/repository.py`: Added `get_by_video_ids()`, `upsert_embedding()`, and `find_similar_with_filters()`.
4. `src/db/database.py`: Added graceful SQLite fallback when psycopg2 is not installed.
5. `src/api/schemas.py`: Re-exported recommendation schemas.
6. `src/api/app.py`: Registered `recommendation_service` and recommendation endpoints.
