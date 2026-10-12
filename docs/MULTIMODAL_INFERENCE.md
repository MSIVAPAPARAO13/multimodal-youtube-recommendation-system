# Multimodal Inference Architecture & Specification

**Phase 2 — Real Multimodal Inference Foundation**  
**Date:** October 8, 2026  
**Status:** **PHASE 2 STATUS: PASS**

---

## 1. System Architecture

The recommendation system processes three distinct media modalities (visual frames, audio waveforms, and textual metadata) through dedicated foundation encoders, maps them into a shared 512-dimensional embedding space, combines them via gated attention fusion, and normalizes the resulting vector onto the unit hypersphere for high-performance vector search with `pgvector`.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Input Streams (Single / Batch)                 │
└───────┬───────────────────────────┬───────────────────────────┬────────┘
        │ Video Frames              │ Audio Waveform            │ Text Metadata
        ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│ VideoProcessor   │        │ AudioProcessor   │        │ format_text_     │
│ - 8 frames       │        │ - 16 kHz resample│        │   input()        │
│ - 224x224 RGB    │        │ - 5.0s window    │        │ - Title, Desc,   │
│ - ImageNet norm  │        │ - Safe fallback  │        │   Tags, Transc.  │
└───────┬──────────┘        └───────┬──────────┘        └───────┬──────────┘
        │ (B, 8, 3, 224, 224)       │ (B, 80000)                │ Token IDs
        ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│ VideoEncoder     │        │ AudioEncoder     │        │ TextEncoder      │
│ - ResNet-50      │        │ - Wav2Vec2 Base  │        │ - BERT Base      │
│   (ImageNet1K)   │        │   (960h speech)  │        │   (uncased)      │
│ - Frozen backbone│        │ - Frozen backbone│        │ - Frozen backbone│
│ - Projection Head│        │ - Projection Head│        │ - Projection Head│
└───────┬──────────┘        └───────┬──────────┘        └───────┬──────────┘
        │ (B, 512)                  │ (B, 512)                  │ (B, 512)
        └───────────────────┬───────┴───────────────────────────┘
                            ▼
        ┌──────────────────────────────────────┐
        │            GatedFusion               │
        │ - 3-way Gating Network (MLP + ReLU)  │
        │ - Dynamic Zero-Modality Masking      │
        │ - Output Projection Head             │
        └───────────────────┬──────────────────┘
                            │ (B, 512)
                            ▼
        ┌──────────────────────────────────────┐
        │           L2 Normalization           │
        │ F.normalize(p=2, dim=-1)             │
        │ torch.nan_to_num(nan=0.0)            │
        └───────────────────┬──────────────────┘
                            │ (B, 512) ||v|| = 1.0
                            ▼
        ┌──────────────────────────────────────┐
        │   Two-Tower Cosine Retrieval / Index │
        │   pgvector: HNSW (vector_cosine_ops) │
        └──────────────────────────────────────┘
```

---

## 2. Models & Foundation Encoders

1. **Visual Backbone:** Pretrained `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`. Extracts 2048-dimensional convolutional features pooled over 8 sampled frames, projected to 512 dimensions.
2. **Audio Backbone:** Pretrained `transformers.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")`. Processes raw 16 kHz audio waveforms (80,000 samples for a 5-second slice), extracting 768-dimensional temporal representations, projected to 512 dimensions.
3. **Text Backbone:** Pretrained `transformers.BertModel.from_pretrained("bert-base-uncased")` with `BertTokenizer`. Processes token sequences up to 128 tokens, extracting 768-dimensional CLS token embeddings, projected to 512 dimensions.
4. **Fusion Network:** `GatedFusion` with dynamic gated importance scoring across active modalities.

---

## 3. Input Formats

### Video Input
- Accepted types: Local file path (`str` or `Path`), pre-extracted tensor `(num_frames, 3, H, W)`, or batch tensor `(B, num_frames, 3, H, W)`.
- Constraints: File must reside within configured `ALLOWED_MEDIA_DIRS` (`data`, `videos`, `tests`).

### Audio Input
- Accepted types: Local media file path (audio extracted via `librosa`), raw 1D waveform `(80000,)`, or batch waveform `(B, 80000)`.
- Fallback: Missing audio streams, corrupted files, or systems without ffmpeg degrade to zero-padded tensors without throwing exceptions.

### Text Input
- Accepted types: Raw string title (backward-compatible) or structured metadata dictionary:
  ```json
  {
    "title": "Introduction to Multimodal Deep Learning",
    "description": "Comprehensive tutorial covering ResNet, Wav2Vec2, and BERT.",
    "tags": ["AI", "MachineLearning", "PyTorch"],
    "transcript": "In this video we demonstrate multimodal embeddings..."
  }
  ```

---

## 4. Tensor Shapes & Dimensions

| Stage | Modality | Tensor Dimensions | Description |
| :--- | :--- | :--- | :--- |
| **Input** | Video | `(B, 8, 3, 224, 224)` | 8 sampled RGB frames resized to 224×224 |
| | Audio | `(B, 80000)` | 5.0 seconds sampled at 16,000 Hz |
| | Text | `(B, 128)` | Token IDs and attention mask |
| **Backbone** | Video | `(B * 8, 2048, 1, 1)` | ResNet-50 feature maps |
| | Audio | `(B, 249, 768)` | Wav2Vec2 sequence features |
| | Text | `(B, 128, 768)` | BERT contextual embeddings |
| **Projection** | Video | `(B, 512)` | Frame mean-pooled and projected |
| | Audio | `(B, 512)` | Sequence mean-pooled and projected |
| | Text | `(B, 512)` | CLS-pooled and projected |
| **Fusion** | All | `(B, 512)` | Gated weighted sum + projection |
| **Output** | Final | `(B, 512)` | L2-normalized unit vector |

---

## 5. Preprocessing Pipelines

- **Video Preprocessing (`VideoProcessor`):**
  1. Evenly samples 8 frames across the video duration using OpenCV.
  2. Converts BGR to RGB color space.
  3. Resizes frames to 224×224 pixels.
  4. Applies ImageNet normalization:
     $$\text{mean} = [0.485, 0.456, 0.406], \quad \text{std} = [0.229, 0.224, 0.225]$$
- **Audio Preprocessing (`AudioProcessor`):**
  1. Loads audio stream via Librosa with fallback to zero arrays on decode failure.
  2. Resamples to 16,000 Hz mono.
  3. Enforces constant length of 80,000 samples (pad with zeros or truncate).
  4. Filters out NaNs and Infs via `np.nan_to_num`.
- **Text Preprocessing (`format_text_input`):**
  1. Strips and joins title, truncated description (first 200 chars), comma-separated tags, and transcript (first 300 chars).
  2. Tokenizes into WordPiece tokens with max sequence length 128.

---

## 6. Fusion Strategies

The system includes three fusion architectures in `src/models/fusion.py`:
1. **`GatedFusion` (Default Production):** Learns gating coefficients $\alpha_v, \alpha_a, \alpha_t \in [0, 1]$ representing the dynamic importance of visual, auditory, and textual content. Employs dynamic zero-modality masking to eliminate bias when an individual modality is absent.
2. **`ConcatFusion`:** Baseline concatenating all three 512-d representations into 1536-d space followed by a 2-layer MLP projection.
3. **`AttentionFusion`:** Cross-modality multi-head attention weighting features across modalities.

---

## 7. Normalization Strategy

To ensure mathematical precision during vector search:
1. **Scale:** Modality projection heads map representations to $\mathbb{R}^{512}$.
2. **Fusion:** Gated combination produces a raw fused representation $z \in \mathbb{R}^{512}$.
3. **Unit Hypersphere Projection:**
   $$e = \frac{z}{\max(\|z\|_2, \epsilon)}$$
4. **NaN/Inf Sanitization:** `torch.nan_to_num(e, nan=0.0)` guarantees numeric cleanliness.
5. **Cosine Metric Equivalence:** Because $\|e\|_2 = 1.0$, cosine similarity between two videos $u, v$ reduces to the inner dot product:
   $$\text{cosine\_similarity}(u, v) = u^\top v$$
   In `pgvector`, the cosine distance metric `vector_cosine_ops` computes $1 - u^\top v$.

---

## 8. Checkpoint Policy & Execution Modes

The system explicitly distinguishes between:
1. **Pretrained Foundation Weights:** ResNet-50 ImageNet, Wav2Vec2 960h, BERT uncased. Loaded directly from trusted repositories.
2. **Project-Specific Weights:** Projection heads and fusion layers saved as `MODEL_PATH` (`checkpoints/best.pt`).

### Development Mode (`APP_ENV=development` or `ALLOW_UNINITIALIZED_MODEL=true`)
- Permits startup without a local project-specific `.pt` checkpoint.
- Uses pretrained foundation encoders with PyTorch-initialized projection and fusion heads.
- Emits explicit log warnings: `"[DEVELOPMENT MODE] Running with pretrained foundation encoders and initialized projection/fusion layers."`
- Sets `is_checkpoint_loaded = False` in health reports.

### Production Mode (`APP_ENV=production` and `ALLOW_UNINITIALIZED_MODEL=false`)
- Strictly requires a valid project checkpoint at `MODEL_PATH`.
- Fails startup immediately with a `RuntimeError` if the checkpoint does not exist.
- Strictly loads weights using `weights_only=True` for security against arbitrary code execution.

---

## 9. Benchmark Methodology & Latency Profile

Benchmarks were executed via `scripts/benchmark_inference.py` on an Intel/AMD CPU runtime (10 warmup + 10 benchmark iterations).

### Latency Profile

| Component | Mean Latency | Percentage |
| :--- | :---: | :---: |
| **Video Encoding** (ResNet-50, 8 frames) | 364.33 ms | 52.1% |
| **Audio Encoding** (Wav2Vec2, 5s audio) | 304.92 ms | 43.6% |
| **Text Encoding** (BERT, title/metadata) | 28.76 ms | 4.1% |
| **Multimodal Fusion** (GatedFusion + L2) | 0.56 ms | <0.1% |
| **Total Multimodal Inference Latency** | **698.57 ms** | **100.0%** |

- **Throughput (Single CPU Thread):** ~1.43 requests/second
- **Model Initialization (Foundation + Heads):** ~23.01 seconds (one-time startup cost)

---

## 10. Known Limitations

1. **No Checkpoint Trained on Co-View Pairs:** The repository does not yet include a `.pt` checkpoint trained with contrastive learning on video-to-video co-views.
2. **CPU Inference Latency:** Running ResNet-50 and Wav2Vec2 on CPU takes ~700 ms per video. GPU acceleration (CUDA) is recommended for high-throughput batch indexing.
3. **Absence of User Tower:** Recommendations are currently item-to-item based on content similarity. Personalization requires the User Tower scheduled for later phases.
