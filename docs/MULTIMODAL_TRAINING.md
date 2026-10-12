# Multimodal Contrastive Training Pipeline

## 1. Overview and Architecture

The Multimodal YouTube Recommendation System employs a unified Two-Tower architecture designed for content-to-content and query-to-item retrieval. The architecture processes three distinct input modalities into a unified 512-dimensional metric space:

```
[ Video Frames (8 frames) ]    --> VideoEncoder (ResNet-50)  --> Projection Head (2048 -> 512) --+
[ Audio Waveform (16 kHz) ]    --> AudioEncoder (Wav2Vec2)   --> Projection Head (768 -> 512)  --+--> GatedFusion (1536 -> 256 -> 3) --> L2 Norm (512-D)
[ Text Metadata (Title/Desc) ] --> TextEncoder (BERT)        --> Projection Head (768 -> 512)  --+
```

### Key Architectural Invariants
- **Foundation Backbones:** ResNet-50 (visual), Wav2Vec2-base-960h (audio), BERT-base-uncased (text).
- **Projections:** Two-layer MLPs with ReLU and Dropout mapping raw backbone dimensions to 512-D.
- **Fusion Layer:** `GatedFusion` learning sample-specific importance weights dynamically conditioned on modality availability.
- **Output:** Unit-norm 512-dimensional vector optimized for cosine similarity / inner product search.

---

## 2. Trainable vs. Frozen Components

To ensure computational efficiency and prevent catastrophic forgetting of foundational representations, training isolates the task-specific alignment parameters:

| Component | Architecture | Status During Training | Parameter Count |
|---|---|---|---|
| Visual Backbone | ResNet-50 | **Frozen** (`requires_grad=False`) | ~23.5M |
| Visual Projection | Linear(2048, 2048) -> Linear(2048, 512) | **Trainable** | ~5.2M |
| Audio Backbone | Wav2Vec2-base-960h | **Frozen** (`requires_grad=False`) | ~94.4M |
| Audio Projection | Linear(768, 768) -> Linear(768, 512) | **Trainable** | ~0.98M |
| Text Backbone | BERT-base-uncased | **Frozen** (`requires_grad=False`) | ~109.5M |
| Text Projection | Linear(768, 768) -> Linear(768, 512) | **Trainable** | ~0.98M |
| Gated Fusion Gate | Linear(1536, 256) -> ReLU -> Linear(256, 3) | **Trainable** | ~394K |

Total Trainable Parameters: ~7.6M parameters.

---

## 3. Data Availability Gate & Requirements

### Required Data Schema
Each training sample must adhere to the validated `MultimodalTrainingSample` schema:
- `video_id`: Non-empty string identifier.
- `visual_embedding`: 512-D tensor, non-NaN/Inf, L2-norm > 1e-6 (or None if unavailable).
- `audio_embedding`: 512-D tensor, non-NaN/Inf, L2-norm > 1e-6 (or None if unavailable).
- `text_embedding`: 512-D tensor, non-NaN/Inf, L2-norm > 1e-6 (or None if unavailable).
- `positive_group_id` / `positive_video_ids`: Genuine semantic or interaction relationships.
- `split`: Explicitly assigned to `'train'`, `'val'`, or `'test'`.
- `modality_availability`: Boolean mask `[has_visual, has_audio, has_text]`.

### Data Gate Audit Status
- **MicroLens-50k Pairs:** 359,710 interaction events are available in CSV tabular format.
- **MicroLens-50k Titles:** 19,221 titles available in CSV format.
- **MicroLens Media Assets:** Only 10 sample `.mp4` video files exist locally; 0 audio `.wav` files exist.
- **YouTube Media Assets:** `data/raw/youtube/` contains no complete media fixtures.
- **Status:** **BLOCKED — TRAINING NOT VERIFIED**. Per Phase 8 critical instructions, training infrastructure and validation pipelines are fully implemented and verified, but model training is not claimed as completed without real aligned multimodal media. Synthetic labels or fake weights are strictly forbidden.

---

## 4. Positive and Negative Pair Construction

Implemented in `src/data/pair_generator.py`:

1. **Positive Pairs:**
   - Derived exclusively from validated human relationships, topic groupings, or user co-engagement sessions.
   - Self-pairs $(u, u)$ are rejected.
   - Cross-split positive pairs are detected and rejected as split leakage.

2. **Negative Pairs & False-Negative Suppression:**
   - In-batch negatives are generated for batch diversity.
   - **False-Negative Suppression Mask:** An $(N, N)$ boolean mask suppresses any sample pairs that belong to the same positive group or declared positive set, preventing penalty gradients on legitimate semantic relatives.
   - Deterministic pair generation is governed by `TRAINING_SEED`.

---

## 5. Contrastive Objective Formulation

Implemented in `src/training/losses.py`:

### InfoNCE Loss with False-Negative Suppression
$$\mathcal{L}_{i} = -\log \frac{\exp(\text{sim}(q_i, k_i^+) / \tau)}{\exp(\text{sim}(q_i, k_i^+) / \tau) + \sum_{j \in \mathcal{N}_i} \exp(\text{sim}(q_i, k_j^-) / \tau)}$$

where:
- $\text{sim}(u, v) = \frac{u^\top v}{\|u\|_2 \|v\|_2}$ (L2-normalized cosine similarity).
- $\tau = 0.07$ (validated temperature parameter $> 0$).
- $\mathcal{N}_i$ is the filtered negative set where false-negative pairs are masked out ($-\infty$ logit replacement).
- Missing modalities (all-zero vectors) are detected via norm masking and excluded from similarity computation.

### Symmetric Contrastive Loss
To prevent asymmetric tower bias, the symmetric loss averages both directions:
$$\mathcal{L}_{\text{sym}} = \frac{1}{2} \left( \mathcal{L}_{A \to B} + \mathcal{L}_{B \to A} \right)$$

---

## 6. Training Configuration and Execution

Configuration parameters in `src/db/config.py`:
- `TRAINING_ENABLED`: Default `False` (prevents startup execution).
- `TRAINING_SEED`: `42` (reproducible RNG).
- `TRAINING_BATCH_SIZE`: `32` (strictly positive validated).
- `TRAINING_EPOCHS`: `10` (strictly positive validated).
- `TRAINING_LEARNING_RATE`: `1e-4` (AdamW optimizer).
- `TRAINING_WEIGHT_DECAY`: `1e-5` (L2 regularization).
- `TRAINING_TEMPERATURE`: `0.07`.
- `TRAINING_VALIDATION_SPLIT`: `0.2`.
- `TRAINING_EARLY_STOPPING_PATIENCE`: `3`.
- `TRAINING_CHECKPOINT_DIR`: `"data/checkpoints"`.
- `TRAINING_FREEZE_BACKBONES`: `True`.

### Execution Command
```bash
python scripts/train_multimodal.py --data-path data/training_dataset.json --epochs 10 --batch-size 32
```

---

## 7. Checkpoint Security and Lifecycle

Implemented in `src/training/checkpoint.py`:
- **Atomic Persistence:** Checkpoints are written to temporary `.tmp` files and atomically renamed.
- **Metadata Envelope:** Checkpoints include epoch, global step, validation loss, feature dimension, architecture name, and parameter counts.
- **Safe Deserialization:** Utilizes `weights_only=True` to prevent arbitrary code execution vulnerabilities.
- **Dimension Check:** Rejects checkpoints whose embedding dimension does not match the active model architecture.

---

## 8. Evaluation Protocol

Implemented in `scripts/evaluate_multimodal.py`:
- **Metrics:** Recall@1, Recall@5, Recall@10, Precision@K, Mean Reciprocal Rank (MRR), NDCG@K.
- **Candidate Pool:** Query set is strictly partitioned from candidate gallery to evaluate generalization.
- **Leakage Safeguards:** Queries with 0 verified positive candidates are tracked and excluded from zero-division inflation.

```bash
python scripts/evaluate_multimodal.py --data-path data/eval_dataset.json --checkpoint-path data/checkpoints/best.pt
```
