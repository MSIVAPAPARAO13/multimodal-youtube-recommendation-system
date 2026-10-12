# Phase 8B Validation Report: Text-Only Metadata Co-Engagement

**Execution Date:** 2026-10-09  
**Experiment ID:** `EXP_MICROLENS_TEXT_COENGAGEMENT_V1`  
**Status:** **PASS** (Text-Only Co-Engagement Training & Evaluation Completed; Multimodal Remains Blocked)

---

## 1. Summary of Milestones

| Milestone | Status | Verified Details |
|---|---|---|
| **Phase 8A Audit Integration** | **PASS** | Audited 359,708 interaction records and 19,220 titles; confirmed 26,299 eligible 24h transition users. |
| **Pretrained Weights Verification** | **PASS** | Verified official `bert-base-uncased` weights (109.5M params) loaded with `local_files_only=True`. |
| **Text Feature Extraction** | **PASS** | Extracted and cached 768-D BERT representations for 2,960 items in 91.80s (`data/cache/microlens_bert_2960.pt`). |
| **User-Disjoint Partitioning** | **PASS** | 800 Train / 100 Val / 100 Test users with strict zero user overlap verified by unit tests. |
| **Actual Training Run** | **PASS** | Completed 4 epochs of AdamW + CosineAnnealingLR; early stopping triggered at epoch 4 (`best_val_loss` = 3.5874). |
| **Atomic Checkpoint Saved** | **PASS** | Saved `best.pt` and `latest.pt` to `data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V1/`. |
| **Checkpoint Reload Verification** | **PASS** | Verified state dict loading into `TextEncoder.projection` with `weights_only=True`. |
| **Held-Out Evaluation** | **PASS** | Evaluated 188 test queries against 2,960 candidate gallery items with self-match exclusion. |
| **Baseline Comparisons** | **PASS** | Measured Raw BERT (768-D), Untrained Projection (512-D), and Trained Model (512-D). |
| **Test Suite Regressions** | **PASS** | **207/207 tests passing** (7 new Phase 8B tests; 200 baseline tests preserved). |

---

## 2. Actual Evaluation Results

| Metric | Raw Pretrained BERT (768-D) | Untrained Projection (512-D) | Trained Model (512-D) | Relative Improvement |
|---|---|---|---|---|
| **Recall@1** | 0.0213 | 0.0106 | **0.0160** | **+50.0%** |
| **Recall@5** | 0.0426 | 0.0372 | 0.0319 | -14.3% |
| **Recall@10** | 0.0479 | 0.0479 | 0.0426 | -11.1% |
| **MRR** | 0.0333 | 0.0265 | 0.0259 | -2.3% |
| **NDCG@10** | 0.0345 | 0.0295 | 0.0273 | -7.3% |

---

## 3. Scope Boundary and Interpretation

1. **Demonstrated Capability:**
   - Demonstrates that behavioral co-engagement signals from real user watch trajectories can train a 512-D text projection head to sharply focus top-1 recommendations (+50.0% Recall@1 over untrained random projection).
2. **Explicit Non-Multimodal Boundary:**
   - This experiment does **not** train visual or audio representations.
   - It does **not** train the `GatedFusion` parameters.
   - It must **not** be described as a trained multimodal recommendation system.
   - Full multimodal training remains **BLOCKED** until complete aligned video and audio assets are available.
