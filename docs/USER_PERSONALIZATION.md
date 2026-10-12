# Phase 10 — User Personalization Engine

## 1. Overview and Architecture

The **Phase 10 User Personalization Engine** introduces chronological user interest modeling built directly on top of the Phase 9 Vector Recommendation Engine and Phase 8C's verified text co-engagement embedding space (`TEXT_COENGAGEMENT_V2`, 512 dimensions).

> **Important Architecture Classification:**  
> This implementation is a **transparent, recency-weighted heuristic user personalization engine**, **not** a learned neural user tower, collaborative filtering matrix factorization model, or multimodal personalization model. Candidate representations are derived from Phase 8C text projections (`LinearProjectionModel`, 768 $\to$ 512 dimensions), and user interest profiles are synthesized via normalized historical embedding aggregation.

```
                           +---------------------------+
                           |  Client / Recommendation  |
                           |   GET /api/.../user/{id}  |
                           +-------------+-------------+
                                         |
                                         v
                         +-------------------------------+
                         |   UserPersonalizationService  |
                         +---------------+---------------+
                                         |
            +----------------------------+----------------------------+
            | 1. Fetch History (t < cutoff)                           | 2. Cold-Start Fallback
            v                                                         v
+------------------------+                              +---------------------------+
| InteractionRepository  |                              |  RecommendationService   |
| (user_interactions)    |                              |   (Phase 9 Content-Only   |
+-----------+------------+                              |    or Catalog Fallback)   |
            |                                           +---------------------------+
            v
+------------------------+
| History Deduplication  |
|  & Length Truncation   |
+-----------+------------+
            |
            v
+------------------------+
| build_user_profile()   |  w_k = alpha^(m - 1 - k)
| (Recency Weighting &   |  p = L2_normalize(sum(w_k * e_k))
|  L2 Normalization)     |
+-----------+------------+
            |
            v  p in R^512
+-------------------------------------------------------+
| VectorIndexService.search_database() / exact fallback |
| Candidates filtered by: exclude_seen, category, etc.  |
+---------------------------+---------------------------+
                            |
                            v
               +-------------------------+
               | Batch Metadata Hydration|
               | (VideoRepository)       |
               +------------+------------+
                            |
                            v
               +-------------------------+
               | UserRecommendationResp  |
               +-------------------------+
```

---

## 2. Profile Generation Formula

Given a user's chronological history of interactions strictly preceding the evaluation/request cutoff timestamp:

$$\mathcal{H} = \left\{ (e_0, t_0), (e_1, t_1), \dots, (e_{m-1}, t_{m-1}) \right\} \quad \text{where } t_k < t_{\text{cutoff}}$$

where $e_k \in \mathbb{R}^{512}$ is the unit-normalized text co-engagement embedding of item $k$.

### Exponential Position Decay (Default)
When discrete chronological interaction events are provided:

$$w_k = \alpha^{m - 1 - k}, \quad k \in \{0, 1, \dots, m - 1\}$$

- Default decay factor: $\alpha = 0.85$
- The most recent interaction ($k = m - 1$) receives weight $w_{m-1} = 1.0$.
- Preceding interactions decay exponentially with position offset ($0.85, 0.7225, 0.6141, \dots$).

### Exponential Elapsed Time Decay (Continuous)
When precise elapsed wall-clock timestamps are supplied:

$$\lambda = \frac{\ln 2}{T_{1/2}}, \quad \Delta t_k = \max(0, t_{\text{cutoff}} - t_k), \quad w_k = \exp(-\lambda \Delta t_k)$$

- Default half-life: $T_{1/2} = 7 \text{ days}$ ($604,800 \text{ seconds}$).

### Profile Synthesis and Safe Normalization
The unnormalized raw profile is computed as:

$$\mathbf{p}_{\text{raw}} = \sum_{k=0}^{m-1} w_k \hat{\mathbf{e}}_k$$

where $\hat{\mathbf{e}}_k = \frac{\mathbf{e}_k}{\|\mathbf{e}_k\|_2}$.

The final profile vector is normalized to the unit sphere:

$$\mathbf{p} = \begin{cases} 
\frac{\mathbf{p}_{\text{raw}}}{\|\mathbf{p}_{\text{raw}}\|_2}, & \text{if } \|\mathbf{p}_{\text{raw}}\|_2 > 10^{-8} \text{ and finite} \\
\text{None}, & \text{otherwise}
\end{cases}$$

---

## 3. Candidate Generation and Ranking

1. **Vector Index Querying:** The 512-dimensional profile vector $\mathbf{p}$ is passed directly into `VectorIndexService.search_database()` (or in-memory fallback), reusing the exact candidate indexing and retrieval infrastructure validated in Phase 9.
2. **Seen-Item Exclusion:** When `exclude_seen=True`, all unique item IDs appearing in the user's historical interaction sequence $\mathcal{H}$ are passed to `exclude_video_ids` to ensure next-item recommendations are novel.
3. **Seed Video Blending (Optional):** If a user provides an active `seed_video_id` (e.g., currently watching video $s$ while logged in as user $u$), candidate scoring blends personalized similarity with item-to-item content similarity:
   $$\text{Score}(c) = \beta \cdot \cos(\mathbf{p}, \mathbf{e}_c) + (1 - \beta) \cdot \cos(\mathbf{e}_s, \mathbf{e}_c)$$
   Default blend weight: $\beta = 0.70$.
4. **Metadata Hydration:** Candidates are hydrated with metadata (title, category, channel) in a single batched database query, eliminating N+1 roundtrips.

---

## 4. Cold-Start and Edge Case Handling

The engine handles all edge cases gracefully without raising HTTP 500 exceptions:

| Case | Condition | Fallback Action | `personalization_applied` | `fallback_reason` |
|---|---|---|---|---|
| **Zero History** | Brand new user with no interactions | Seed recommendation (if provided) or catalog fallback | `false` | `"no_history"` |
| **Missing Embeddings** | User interacted with items absent from catalog / un-embedded | Falls back to catalog / seed recommendations | `false` | `"no_embedded_items"` |
| **Single Interaction** | Exactly 1 historical item | Profile equals normalized single item embedding ($w_0 = 1.0$) | `true` | `null` |
| **All-Zero / NaN Vectors** | Invalid vectors in history | Sanitized and skipped; if all invalid, falls back safely | `false` | `"no_embedded_items"` |
| **Empty Candidate Pool** | Strict filters eliminate all items | Returns valid empty list `[]` with `count=0` | `true` | `null` |

---

## 5. API Specification

### Endpoint: `GET /api/recommendations/user/{user_id}`

#### Parameters:
- `user_id` (path, string): Unique identifier of the user.
- `limit` (query, integer, 1..50, default 10): Maximum number of recommended videos.
- `exclude_seen` (query, boolean, default true): Whether to exclude items the user has already interacted with.
- `as_of` (query, ISO-8601 datetime, optional): Historical cutoff timestamp for point-in-time / leakage-free evaluation.
- `seed_video_id` (query, string, optional): Current context video to blend with profile.
- `category_id` (query, string, optional): Metadata category filter.
- `channel_id` (query, string, optional): Metadata channel filter.

#### Example Response:
```json
{
  "user_id": "user_4210",
  "recommendations": [
    {
      "video_id": "18402",
      "title": "Machine Learning with Python Tutorial",
      "score": 0.8412,
      "rank": 1,
      "category_id": "28",
      "channel_title": "Tech Academy"
    }
  ],
  "count": 1,
  "personalization_applied": true,
  "fallback_reason": null,
  "history_length": 4,
  "cutoff_timestamp": "2025-06-15T08:30:00Z",
  "retrieval_mode": "TEXT_COENGAGEMENT_V2",
  "retrieval_backend": "exact_fallback",
  "model_version": "EXP_MICROLENS_USER_PERSONALIZATION_V1"
}
```

---

## 6. Verification and Performance

- **Leakage-Free Chronological Protocol:** Evaluated across 500 users with $\ge 3$ catalog interactions from MicroLens-50K against 2,960 candidates.
- **Latency Measurements:**
  - User profile construction: **0.132 ms** (p50) / **0.255 ms** (p95)
  - Candidate vector ranking: **60.569 ms** (p50) / **69.756 ms** (p95)
  - Total latency: **~61 ms** (CPU exact cosine search)
- **Database Indexing:** SQLite in-memory fallback tested; pgvector HNSW requires PostgreSQL deployment.
