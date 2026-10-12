# Phase 10 — User Personalization Engine Validation Report

**Experiment ID:** `EXP_MICROLENS_USER_PERSONALIZATION_V1`  
**Retrieval Representation:** `TEXT_COENGAGEMENT_V2` (512-D, Phase 8C checkpoint)  
**Date:** October 2026  
**Status:** **PASS** (All acceptance criteria met, 246/246 tests passing)

---

## 1. Executive Summary

Phase 10 implemented a leakage-free **User Personalization Engine** built directly on top of the Phase 9 Vector Recommendation Engine and Phase 8C's verified text co-engagement embedding space (`TEXT_COENGAGEMENT_V2`, 512 dimensions).

### Key Architectural Deliverables
1. **Recency-Weighted Profile Synthesizer:** Computes normalized 512-D interest profile vectors $\mathbf{p} = \text{norm}\left(\sum_k w_k \hat{\mathbf{e}}_k\right)$ using configurable exponential position decay ($\alpha = 0.85$) or wall-clock half-life decay ($T_{1/2} = 7$ days).
2. **Leakage Protection Protocol:** Strictly enforces $t < t_{\text{cutoff}}$ at the repository query boundary, ensuring zero future interactions ever influence profile construction or candidate ordering.
3. **Candidate Retrieval Reuse:** The user profile vector $\mathbf{p}$ queries the existing Phase 9 vector index without introducing redundant indexing layers.
4. **Cold-Start Fallback:** Automatically falls back to seed-based or catalog recommendations when user history is absent or un-embedded, returning HTTP 200 with explicit provenance metadata (`personalization_applied=False`, `fallback_reason="no_history"`).
5. **API Endpoint:** Added `GET /api/recommendations/user/{user_id}` supporting `limit`, `exclude_seen`, `as_of`, `seed_video_id`, `category_id`, and `channel_id`.
6. **Chronological Per-User Evaluation:** Benchmarked against 500 real users (16,246 eligible) and 2,960 catalog items from MicroLens-50K, comparing Cold-Start, Content-Only (Phase 9), and Personalized profiles.

---

## 2. Checkpoint and Dataset Audit

### Checkpoint Integrity Verification
| Checkpoint | Path | Expected SHA-256 | Verified SHA-256 | Status |
|---|---|---|---|---|
| **Phase 8B** | `data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V1/best.pt` | `430dc7c44febbaa11fe8671515f96afddd7b2af191454da2af954167a2cd9bf0` | `430dc7c44febbaa11fe8671515f96afddd7b2af191454da2af954167a2cd9bf0` | **MATCH** |
| **Phase 8C** | `data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt` | `a82037eb790e2a23e133b8edc9973a0e6fdc3c466058abd61b68aed77d2d4996` | `a82037eb790e2a23e133b8edc9973a0e6fdc3c466058abd61b68aed77d2d4996` | **MATCH** |

### Dataset Parameters
- **MicroLens-50K Pairs:** `data/raw/microlens/MicroLens-50k_pairs.csv` (359,708 interaction events across 50,000 users).
- **Item Catalog:** 2,960 unique items (matching Phase 9 indexed catalog).
- **Eligible Users ($\ge 3$ interactions in catalog):** 16,246 users.

---

## 3. Evaluation Protocol and Results

### Chronological Evaluation Setup
- **Evaluation Script:** `scripts/evaluate_user_personalization.py`
- **Output Artifact:** `data/checkpoints/EXP_MICROLENS_USER_PERSONALIZATION_V1/evaluation_results.json`
- **Protocol:**
  1. For each user, interactions are sorted strictly by timestamp ($t_1 \le t_2 \le \dots \le t_N$).
  2. Cutoff timestamp is set to $t_{\text{cutoff}} = t_N$ (the final held-out interaction).
  3. History $\mathcal{H}$ comprises items $\{I_1, \dots, I_{N-1}\}$ with $t_k < t_{\text{cutoff}}$.
  4. Target ground-truth item is $I_N$.
  5. Candidates pool is the full 2,960 item catalog minus historical seen items (`exclude_seen=True`).
- **Baselines:**
  - **Cold-Start Baseline:** Catalog popularity order without personalization.
  - **Content-Only Baseline:** Phase 9 nearest-neighbor candidate retrieval using the most recent item $I_{N-1}$'s embedding as query.
  - **Personalized Profile:** Recency-weighted profile vector $\mathbf{p} = \text{norm}\left(\sum_k \alpha^{N-2-k} \hat{\mathbf{e}}_k\right)$ as query.

### Empirical Metrics Comparison (500 Queries, 2,960 Candidates)

| Metric | Cold-Start Baseline | Content-Only Baseline (Phase 9) | Personalized Profile (Phase 10) | Absolute Delta ($\Delta$) | Relative Improvement |
|---|---|---|---|---|---|
| **Recall@1** | 0.0000 | 0.0120 | **0.0220** | **+0.0100** | **+83.3%** |
| **Recall@5** | 0.0100 | 0.0400 | **0.0500** | **+0.0100** | **+25.0%** |
| **Recall@10** | 0.0120 | 0.0480 | **0.0600** | **+0.0120** | **+25.0%** |
| **Precision@5** | 0.0020 | 0.0080 | **0.0100** | **+0.0020** | **+25.0%** |
| **Precision@10** | 0.0012 | 0.0048 | **0.0060** | **+0.0012** | **+25.0%** |
| **MRR** | 0.0033 | 0.0227 | **0.0327** | **+0.0100** | **+44.1%** |
| **NDCG@10** | 0.0054 | 0.0287 | **0.0392** | **+0.0105** | **+36.6%** |

*All metrics are verified empirical outputs from `scripts/evaluate_user_personalization.py`.*

### Latency and Efficiency
- **Profile Vector Construction:**
  - Mean: **0.147 ms**
  - p50: **0.132 ms**
  - p95: **0.255 ms**
- **Candidate Vector Ranking (2,960 items, CPU exact cosine search):**
  - Mean: **61.27 ms**
  - p50: **60.57 ms**
  - p95: **69.76 ms**
- **Fallback Rate on Valid User Cohort:** **0.0%** (0 / 500)

---

## 4. Test Suite and Regression Results

- **Test Command:** `python -m pytest tests/ -q`
- **Result:** **246 passed, 0 failed, 0 skipped** (previous baseline: 232 passed).
- **New Tests Added (14 tests in `tests/test_user_personalization.py`):**
  1. `test_profile_generation_dimension_and_norm` — strictly 512-D and unit L2 normalized
  2. `test_profile_generation_empty_and_zero_vectors` — returns `None` safely
  3. `test_profile_generation_nan_inf_safety` — skips non-finite values without corrupting output
  4. `test_profile_generation_deterministic` — identical output for identical inputs
  5. `test_profile_recency_decay_weighting` — newer items receive higher weight
  6. `test_interaction_repo_cutoff_leakage_protection` — interactions $\ge t_{\text{cutoff}}$ strictly excluded
  7. `test_interaction_user_isolation` — interaction histories isolated across users
  8. `test_personalized_recommendations_ranking_and_seen_exclusion` — candidate ranking and seen-item exclusion
  9. `test_personalized_recommendations_cold_start_fallback` — graceful fallback for 0-history users
  10. `test_personalized_recommendations_history_deduplication` — preserves latest position on repeat views
  11. `test_api_user_recommendations_success` — HTTP 200 personalized response schema
  12. `test_api_user_recommendations_cold_start` — HTTP 200 cold start fallback response
  13. `test_api_user_recommendations_invalid_limit` — HTTP 422 for limit out of bounds
  14. `test_evaluate_ranking_metric_correctness` — metric helper verified on hand-constructed ground truth

---

## 5. Honest Limitations and Non-Claims

1. **Text-Only Embeddings:** Recommendations rely on 512-D text co-engagement projections from BERT titles. Visual and audio embeddings are not utilized in this recommendation phase.
2. **Heuristic Profile Aggregation:** Personalization uses recency-decayed vector averaging, **not** a learned neural user tower, SASRec sequence model, or collaborative filtering matrix factorization.
3. **Vector Backend Environment:** Benchmarking used CPU in-memory exact cosine similarity; local environments lack `pgvector` and PostgreSQL C-bindings. HNSW index performance has not been measured on GPU or dedicated vector DB instances.
4. **Implicit Interactions:** Interactions in MicroLens represent user watch co-engagements without dwell-time, completion percentage, or explicit rating labels.

---

## 6. Phase Acceptance Summary

| Acceptance Criterion | Result | Evidence |
|---|---|---|
| Integrated with existing Phase 9 vector architecture | **PASS** | `UserPersonalizationService` queries `VectorIndexService.search_database()` |
| Genuine historical interactions & compatible embeddings | **PASS** | MicroLens-50K interaction sequences + `TEXT_COENGAGEMENT_V2` (512-D) |
| Strict temporal leakage prevention | **PASS** | Enforced via SQL `$t < t_{\text{cutoff}}$` and tested in unit tests |
| Cold-start and edge-case handling | **PASS** | Valid responses with `personalization_applied=False` and fallback metadata |
| API validated & documented | **PASS** | `GET /api/recommendations/user/{user_id}` passing 4 route tests |
| Controlled baseline comparison | **PASS** | Cold-Start vs Content-Only vs Personalized evaluated under identical protocol |
| Empirical evaluation metrics generated | **PASS** | Output saved to `data/checkpoints/EXP_MICROLENS_USER_PERSONALIZATION_V1/evaluation_results.json` |
| Test suite passes with zero regressions | **PASS** | 246 passed, 0 failed |
| Checkpoints intact | **PASS** | SHA-256 verified for Phase 8B and Phase 8C checkpoints |

**Overall Phase 10 Status: PASS**
