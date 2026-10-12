# Phase 8A: MicroLens Dataset Audit and Training Readiness Report

**Audit Date:** 2026-10-09  
**Status:** Audit Completed — Text-Only Co-Engagement Feasible; Full Multimodal Training Remains Blocked

---

## 1. Executive Summary

This audit rigorously inspects the local contents and schemas of the MicroLens-50K dataset (`data/raw/microlens/`) and evaluates its readiness for recommendation model training within the existing Phase 8 training infrastructure.

Key conclusions:
1. **Interaction Metadata is Complete:** `MicroLens-50k_pairs.csv` contains **359,708 valid implicit feedback watch interactions** across **50,000 unique users** and **19,220 unique items**.
2. **Text Metadata is Complete:** `MicroLens-50k_titles.csv` contains **19,220 titles with 100% coverage** and zero missing or null fields.
3. **Multimodal Media is Almost Completely Missing:** Only **10 sample `.mp4` video files** exist locally (0.052% coverage). Zero standalone `.wav` audio files exist.
4. **Text-Only Co-Engagement Experiment is Feasible and Defensible:** The genuine sequential interaction history can support a high-integrity **Text-Only Metadata Co-Engagement Contrastive Experiment** using frozen BERT + trainable text projection head.
5. **Strict Scope Boundary Maintained:** This text-only configuration must never be labeled or represented as "Multimodal Training". Full multimodal training remains **BLOCKED** pending acquisition of video and audio media assets.

---

## 2. Actual CSV Schemas and Dataset Statistics

### A. `data/raw/microlens/MicroLens-50k_pairs.csv`
- **Total Records:** 359,708 interaction rows (excluding header).
- **Columns:**
  - `user` (int64): Unique user identifier ($0 \le \text{user} < 50,000$).
  - `item` (int64): Unique video item identifier ($1 \le \text{item} \le 19,220$).
  - `timestamp` (int64): Unix timestamp in milliseconds.
- **Unique Users:** 50,000.
- **Unique Items:** 19,220.
- **Duplicate Checks:**
  - Duplicate `(user, item)` pairs: **0** (each user-item interaction is strictly unique).
  - Exact duplicate rows `(user, item, timestamp)`: **0**.
- **Temporal Range:**
  - Minimum Timestamp: `1583378629552` (2020-03-05 03:23:49 UTC).
  - Maximum Timestamp: `1662984132429` (2022-09-12 12:02:12 UTC).
  - Total Time Span: ~2.5 years (921 days).
- **User Activity Distribution (5-Core Filtered):**
  - Minimum: 5 interactions per user.
  - 25th Percentile: 5.0 interactions.
  - Median (50th): 6.0 interactions.
  - Mean: 7.19 interactions.
  - 75th Percentile: 8.0 interactions.
  - 99th Percentile: 24.0 interactions.
  - Maximum: 218 interactions.
- **Item Popularity Distribution:**
  - Minimum: 1 interaction.
  - Median: 11.0 interactions.
  - Mean: 18.72 interactions.
  - Maximum: 342 interactions.

### B. `data/raw/microlens/MicroLens-50k_titles.csv`
- **Total Records:** 19,220 item titles (excluding header).
- **Columns:**
  - `item` (int64): Unique video item identifier.
  - `title` (string): Title text, descriptions, and hashtags translated to English.
- **Missing / Null Titles:** **0** (100% complete).
- **Alignment Overlap:**
  - Items in both `pairs.csv` and `titles.csv`: **19,220 (100.0%)**.
  - Items in `pairs.csv` missing in `titles.csv`: **0**.
  - Items in `titles.csv` missing in `pairs.csv`: **0**.

### C. Media Assets in `data/raw/microlens/videos/`
- **Total Files:** 10 `.mp4` video files (`10003.mp4`, `10005.mp4`, `10011.mp4`, `10016.mp4`, `1002.mp4`, `10034.mp4`, `10036.mp4`, `10037.mp4`, `10039.mp4`, `10040.mp4`).
- **Media Coverage:** $\frac{10}{19,220} = 0.052\%$.
- **Audio Files:** 0 standalone audio files.

---

## 3. Interaction Semantics and Data Provenance

1. **Origin & Benchmark:**
   - Sourced from the MicroLens-50K short-video recommendation dataset (Ni et al., 2023, WWW/SIGIR), derived from the Kuaishou short-video platform.
2. **Semantics:**
   - Implicit feedback interaction logs representing user video view / consumption events in chronological sequence.
   - Does not contain explicit ratings (1–5 stars) or explicit like/dislike flags.
   - Sequential consumption patterns: Users watch items sequentially in sessions.
3. **Co-Engagement Structure:**
   - When interactions are ordered chronologically by user:
     $$\text{Sequence}(u) = [ (i_1, t_1), (i_2, t_2), \dots, (i_K, t_K) ]$$
   - Total consecutive transition pairs $(i_k, i_{k+1})$ across all users: **309,708 pairs**.
   - Same-session pairs ($\Delta t \le 1 \text{ hour}$): **13,450 pairs (4.3%)**.
   - Same-day pairs ($\Delta t \le 24 \text{ hours}$): **54,666 pairs (17.7%)**.
   - These transition pairs constitute genuine behavioral co-engagement supervision without synthetic labels.

---

## 4. Defensibility of Text-Only Training

| Dimension | Assessment | Details |
|---|---|---|
| **Recommender Systems Theory** | **Defensible** | Two-Tower semantic matching from co-engagement logs is standard in industry (YouTube candidate generation, DSSM, Word2Vec/Item2Vec). |
| **Label Provenance** | **Genuine** | Supervision stems directly from actual user watch sequences, not synthetic heuristics. |
| **Architectural Compatibility** | **Verified** | Phase 8's `GatedFusion` correctly assigns gate weights `[0.0, 0.0, 1.0]` when visual/audio norms are zero, isolating the text representation cleanly. |
| **Contrastive Loss Hygiene** | **Verified** | `InfoNCELoss` handles `[False, False, True]` modality masks without numerical instability or zero-vector contamination. |
| **Scope Warning** | **Strict** | Must be explicitly designated as **Text-Only Metadata Co-Engagement Experiment** (`EXP_MICROLENS_TEXT_COENGAGEMENT`). Must **never** be labeled as multimodal training. |

---

## 5. Split and Leakage Risks

1. **Random Interaction Splitting (High Risk — Unacceptable):**
   - Splitting `MicroLens-50k_pairs.csv` randomly by row causes severe information leakage. If user $u$ watches item $A$ at $t_1$ (in Train) and item $B$ at $t_2$ (in Test), the model can memorize the user's session state.
2. **User-Disjoint Split (Recommended for Two-Tower Retrieval):**
   - Partition users: 80% Train Users (40,000), 10% Validation Users (5,000), 10% Test Users (5,000).
   - Training pairs are extracted exclusively from Train Users.
   - Validation and Test pairs are extracted exclusively from Validation and Test Users.
   - Guarantees zero user overlap across splits.
3. **Temporal Leave-One-Out Split (Standard for Sequential Recommendation):**
   - For each user:
     - Interactions $1 \dots K-2$: Train.
     - Interaction $K-1$: Validation.
     - Interaction $K$: Test.
   - Prevents future-to-past temporal leakage.
4. **Candidate Gallery Separation:**
   - Candidate pool for retrieval evaluation is fixed to the catalog of items.
   - Self-matches ($q_i == c_j$) must be explicitly masked out to prevent trivial rank-1 hits.

---

## 6. Recommended Smallest Legitimate Experiment

### Experiment Name: `EXP_MICROLENS_TEXT_COENGAGEMENT_V1`

To validate recommendation learning without downloading 19,000 videos:

1. **Cohort Selection:**
   - Select top 1,000 active users (users with 10–25 interactions).
   - Filter to consecutive interaction pairs within 24 hours ($\Delta t \le 86,400\text{ s}$).
   - Yields ~3,000–5,000 genuine positive co-engagement pairs over ~1,500 unique items.
2. **User-Disjoint Partition:**
   - 800 Train Users (~3,200 pairs).
   - 100 Validation Users (~400 pairs).
   - 100 Test Users (~400 pairs).
3. **Representation Pipeline:**
   - Use frozen BERT-base-uncased to extract 768-D embeddings for the ~1,500 titles.
   - Format into `MultimodalTrainingDataset` with `modality_availability = [False, False, True]`.
4. **Trainable Components:**
   - Train `TextEncoder.projection` ($768 \to 768 \to 512$) using AdamW ($\text{lr} = 10^{-4}$, weight decay $= 10^{-5}$).
   - Keep backbones frozen.
5. **Loss & Supervision:**
   - Symmetric InfoNCE ($\tau = 0.07$) with in-batch false-negative suppression.
6. **Evaluation Protocol:**
   - Compute Recall@1, Recall@5, Recall@10, MRR, NDCG@10 on the test split against the item candidate pool.
   - Compare trained text projection against untrained random projection baseline.

---

## 7. Exact Prerequisites for Full Multimodal Training

Full multimodal contrastive training (combining vision, audio, and text) remains **BLOCKED** until the following data prerequisites are fulfilled:

1. **Video Assets:** Download the remaining 19,210 raw video `.mp4` files into `data/raw/microlens/videos/`.
2. **Audio Assets:** Extract 16 kHz mono `.wav` audio waveforms from all 19,220 video files into `data/raw/microlens/audio/`.
3. **Alternative — Pre-Extracted Foundation Embeddings:**
   - Alternatively, acquire or compute cached feature tensors:
     - Visual: 2048-D ResNet-50 features (mean-pooled over 8 sampled frames per video).
     - Audio: 768-D Wav2Vec2-base features.
     - Text: 768-D BERT-base features.
   - Total cached tensor size: $\approx 276 \text{ MB}$, avoiding large raw video downloads (~50–100 GB).

---

## 8. Summary of Findings

- `MicroLens-50k_pairs.csv` and `titles.csv` are clean, aligned, and valid implicit feedback recommendation logs.
- Text-only co-engagement contrastive learning is conceptually and architecturally ready.
- Full multimodal training is honestly documented as **BLOCKED** pending video/audio media acquisition.
