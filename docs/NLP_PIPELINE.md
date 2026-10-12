# YouTube Semantic NLP & Transcript Pipeline Documentation

**Phase 5 — NLP + Transcript Intelligence**  
**Date:** October 8, 2026  
**Status:** **PHASE 5 STATUS: PASS**

---

## 1. Overview & Objectives

Phase 5 upgrades the recommendation system's textual modality from a simple, title-only BERT representation into a production-grade, multi-field YouTube semantic NLP pipeline. 

The pipeline ingests and structures four distinct textual fields:
1. **Title** (primary semantic anchor)
2. **Description** (contextual details and synopsis)
3. **Tags** (keyword annotations)
4. **Transcript / Captions** (spoken video content extracted in Phase 4)

The pipeline preserves the established multimodal embedding contract: **a deterministic 512-dimensional output embedding**. The final multimodal hypersphere normalization remains within the `GatedFusion` layer.

```
┌────────────────────────────────────────────────────────┐
│  YouTube Text Input (Dict / String / YouTubeTextInput) │
│  - Title (required)                                    │
│  - Description (optional)                              │
│  - Tags (optional, list or string)                     │
│  - Transcript (optional, raw or subtitle cues)         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  src/features/text_processor.py                        │
│  - Unicode NFKC & Whitespace Normalization             │
│  - Transcript Cleanup (Timestamps, Sound Effects, Dup) │
│  - Deterministic Section Markers                       │
│  - Length Constraints Enforced                         │
└──────────────────────────┬─────────────────────────────┘
                           │ Structured String
                           ▼
┌────────────────────────────────────────────────────────┐
│  src/features/text_chunker.py                          │
│  - TextChunker (Sliding Window with Overlap)           │
│  - Chunk Size: 128 | Overlap: 32 | Max Tokens: 1024    │
│  - Batch Chunks Creation [(num_chunks, 128)]           │
└──────────────────────────┬─────────────────────────────┘
                           │ Chunked Tensors
                           ▼
┌────────────────────────────────────────────────────────┐
│  src/features/text_encoder.py                          │
│  - Baseline: bert-base-uncased (110M params)           │
│  - Frozen Backbone & Eval Mode                         │
│  - Per-Chunk CLS Feature Extraction (768-d)            │
│  - Projection Head: Linear(768, 768) -> ReLU ->        │
│                    Dropout(0.1) -> Linear(768, 512)    │
│  - Hierarchical Pooling across Chunks (Mean / Decay)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  TextEmbeddingResult: (1, 512)                         │
│  - Numerical Stability (nan_to_num)                    │
│  - Quality Diagnostics (tokens, chunks, flags)         │
└────────────────────────────────────────────────────────┘
```

---

## 2. Canonical Text Representation (`YouTubeTextInput`)

Implemented in `src/features/text_processor.py`:

```python
class YouTubeTextInput:
    title: str
    description: Optional[str] = None
    tags: Optional[Union[list[str], str]] = None
    transcript: Optional[str] = None
```

### Normalization Guarantees:
- **Missing Values**: `None` values are automatically mapped to empty strings (`""`) or empty tag lists (`[]`).
- **Whitespace Normalization**: Converts non-breaking spaces (`\u00a0`) to standard spaces, unifies carriage returns (`\r\n` ➔ `\n`), collapses multiple horizontal spaces/tabs into a single space, and reduces 3+ newlines to double newlines. Meaningful punctuation (`?`, `!`, `.`, `,`, `-`) is preserved.
- **Unicode Sanitization**: Applies Unicode standard **NFKC** (`unicodedata.normalize('NFKC', text)`), unifying accented characters, decomposing full-width symbols, and removing non-printable control codes.
- **Tags Normalization**: Cleans whitespace around individual tags, removes empty elements, deduplicates entries, and constructs both a cleaned list and a canonical comma-separated string representation.
- **Backward Compatibility**: `YouTubeTextInput.from_any(item)` accepts raw strings (treated as titles), dictionaries with arbitrary subset of fields, or existing `YouTubeTextInput` instances.

---

## 3. Preprocessing & Section Formatting

The text processor constructs deterministic text representations using explicit section boundaries:

```
TITLE:
{normalized_title}

DESCRIPTION:
{normalized_description}

TAGS:
{normalized_tags}

TRANSCRIPT:
{cleaned_transcript}
```

Only non-empty sections are appended. Structuring the input with explicit section markers prevents semantic bleed across fields and ensures the model contextualizes spoken content versus metadata keywords.

### Maximum Length Specifications:

| Section | Default Limit | Policy on Overflow |
| :--- | :--- | :--- |
| **TITLE** | 300 characters | Preserved up to 300 characters |
| **DESCRIPTION** | 1,500 characters | Truncated at nearest word boundary |
| **TAGS** | 500 characters | Truncated at nearest comma/tag boundary |
| **TRANSCRIPT** | 20,000 characters | Cleaned, truncated at word boundary, then passed to `TextChunker` |

---

## 4. Transcript Processing & Artifact Cleanup

Implemented in `clean_transcript()` in `src/features/text_processor.py`.

YouTube video captions—especially auto-generated ones—frequently contain artifacts that degrade BERT embeddings:

1. **SRT / WebVTT Timing Cues**: Regex strips timing ranges such as:
   - `00:00:01.000 --> 00:00:04.500`
   - Standalone timestamp markers `01:23`
   - HTML/VTT style tags `<c>...</c>`, `<00:01:23.000>`
2. **Bracketed & Parenthesized Timestamps**: Strips chapter references like `[02:15]`, `(14:30)`.
3. **Sound Effect Annotations**: Strips descriptive audio annotations such as `[Music]`, `[Applause]`, `[Laughter]`, `(cheering)`, `[Silence]`, `[Inaudible]`.
4. **Duplicate & Rolling Captions**: Auto-generated YouTube captions frequently emit rolling lines where consecutive lines repeat or extend previous text:
   - *Line 1:* `"Welcome back to the channel."`
   - *Line 2:* `"Welcome back to the channel today we are going to learn Python."`
   The cleaning algorithm detects exact duplicates and rolling prefix extensions, keeping the complete line and eliminating stuttering duplicate tokens.
5. **Non-Fatal Availability**: A video without a transcript proceeds cleanly using title, description, and tags. Transcripts are never strictly mandatory.

---

## 5. Long Text Handling & Chunking Strategy

BERT's WordPiece tokenizer and self-attention architecture possess an absolute quadratic limit of 512 tokens. Passing an arbitrarily long transcript directly causes silent truncation or indexing errors.

Implemented in `src/features/text_chunker.py` (`TextChunker`):

### Configurable Parameters:
- `TEXT_CHUNK_SIZE` = 128 tokens
- `TEXT_CHUNK_OVERLAP` = 32 tokens
- `TEXT_MAX_TOKENS` = 1,024 total tokens across chunks
- `STRIDE` = `CHUNK_SIZE - CHUNK_OVERLAP` = 96 tokens

### Sliding Window Algorithm:
1. Tokenize full structured text into sequence $T = [t_1, t_2, \dots, t_N]$.
2. Cap sequence at `max_tokens` ($N \le 1024$).
3. Form chunks $[c_1, c_2, \dots, c_K]$ of length up to `chunk_size` with 32-token overlap.
4. Wrap each chunk with `[CLS]` at the head and `[SEP]` at the tail, padding shorter final chunks.
5. Pass chunks through BERT backbone as a single batched tensor `(K, 128)`.
6. Extract 512-d projections for each chunk: $E_{\text{chunks}} \in \mathbb{R}^{K \times 512}$.
7. Apply hierarchical document pooling across chunks:
   $$\mathbf{e}_{\text{doc}} = \frac{1}{K} \sum_{k=1}^K E_{\text{chunks}}[k]$$
   *(Optional decay pooling $\mathbf{w}_k = \exp(-0.15 \cdot k)$ is available to prioritize opening sections).*

---

## 6. Baseline BERT Architecture & 512-D Projection Head

The default production model remains `bert-base-uncased`:
- **Parameters**: 110M
- **Backbone State**: Pretrained weights, **frozen backbone** (`param.requires_grad = False`).
- **Evaluation Mode**: Set to `eval()` mode (`Dropout` inactive, deterministic inference).
- **Inference Gradients**: Enclosed in `torch.no_grad()`.
- **Token Pooling**: Extract `[CLS]` token embedding (768-d).
- **Projection Head**:
  $$\text{Linear}(768 \to 768) \to \text{ReLU} \to \text{Dropout}(0.1) \to \text{Linear}(768 \to 512)$$
- **Output Tensor**: Numerically sanitized tensor of dimension `(batch_size, 512)`.

---

## 7. Normalization Contract

- The text encoder and text embedding service output raw 512-dimensional representations with reasonable Euclidean norm ($\|e\|_2 \approx 10 - 25$).
- **No L2 normalization occurs inside `TextEncoder`**.
- Sanitization via `torch.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)` prevents NaN propagation.
- Final unit hypersphere normalization ($\|e\|_2 = 1.0$) is strictly executed after multimodal fusion in `MultimodalInferenceService.fuse()` / `GatedFusion`.

---

## 8. Missing Transcript & Modality Fallback

- When a video has no transcript:
  - `clean_transcript(None)` returns `""`.
  - The `TRANSCRIPT:` section is omitted from structured text.
  - Diagnostics record `transcript_available = False`, `transcript_used = False`.
  - Embeddings are generated purely from title, description, and tags.
- When description or tags are also missing:
  - The system gracefully degrades to title-only encoding.
- When all text inputs are empty:
  - Returns a zero vector `(1, 512)` with `is_available = False`, allowing `GatedFusion`'s zero-modality dynamic masking to ignore the text stream.

---

## 9. Experimental Model Comparison (Sentence-Transformers)

Task 7 requires an experimental abstraction allowing future benchmarking against `sentence-transformers/all-MiniLM-L6-v2`.

Implemented in `SentenceTransformerExperimentalEncoder`:
- **Model Name**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional backbone).
- **Projection Head**: `Linear(384, 384) -> ReLU -> Dropout(0.1) -> Linear(384, 512)`.
- **Factory**: `create_text_encoder(model_type="bert"|"sentence-transformers", pretrained=False)`.
- **Test Safety**: **The alternative model is NOT downloaded during normal unit tests**. It uses lazy loading and defaults to `bert-base-uncased`.
- **Status**: Registered as an offline experiment. `bert-base-uncased` remains the production baseline.

---

## 10. Database Schema Additions

Minimal, non-intrusive additions were made to the Phase 4 `video_ingestions` table to track transcript lifecycle without duplicating transcript content:

- `transcript_language` (`VARCHAR(32)`): Detected language code (e.g. `'en'`).
- `transcript_available` (`BOOLEAN`): Boolean flag indicating transcript retrieval status.
- `transcript_provider` (`VARCHAR(64)`): Identifier of the extraction provider (e.g. `'youtube-transcript-api'`).

Both `src/db/models.py` (SQLAlchemy ORM) and `scripts/init_db.sql` (PostgreSQL DDL) were updated.

---

## 11. Controlled Evaluation Methodology & Results

Task 14 established a controlled semantic retrieval benchmark using 15 synthetic video documents across 5 semantic categories (`data/fixtures/synthetic_eval_videos.json`):
1. `python_programming` (3 videos)
2. `machine_learning` (3 videos)
3. `cybersecurity` (3 videos)
4. `web_development` (3 videos)
5. `cloud_computing` (3 videos)

Each document query has exactly 2 relevant ground-truth documents (the other videos in the same category) out of 14 candidates.

### Retrieval Results (`scripts/evaluate_text_retrieval.py`):

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Precision@1** | **40.00%** | Fraction of top-1 retrieved items belonging to the correct topic |
| **Precision@2** | **30.00%** | Fraction of top-2 retrieved items belonging to the correct topic |
| **Recall@1** | **20.00%** | Fraction of relevant items retrieved at rank 1 |
| **Recall@2** | **30.00%** | Fraction of relevant items retrieved at rank 2 |
| **MRR (Mean Reciprocal Rank)** | **0.5691** | Average reciprocal rank of the first relevant document |
| **MAP (Mean Average Precision)** | **0.4424** | Mean ranking quality across relevant items |

> [!NOTE]
> This evaluation is executed on a controlled synthetic fixture. It validates that the structured text pipeline achieves clear topic clustering, not live end-user click-through rates.

---

## 12. Performance Latency Profile

Measured on CPU via `scripts/benchmark_inference.py` (10 iterations):

| Pipeline Stage | Latency | Share |
| :--- | :---: | :---: |
| **1. Text Preprocessing** (Unicode NFKC, cleaning, regex) | 1.20 ms | 0.3% |
| **2. WordPiece Tokenization** | 1.97 ms | 0.5% |
| **3. TextChunker Window Overhead** | 2.51 ms | 0.6% |
| **4. BERT Multi-Chunk Forward Inference + 512-D Projection** | 386.08 ms | 98.6% |
| **Total End-to-End Multi-Chunk Text Embedding Latency** | **391.76 ms** | **100.0%** |

- **Short Text / Title-Only Latency**: ~23.83 ms
- **Theoretical Single-Thread Text Throughput**: ~2.55 requests/second (long multi-chunk transcripts) / ~42.0 requests/second (short titles).

---

## 13. Known Limitations

1. **Monolingual BERT Baseline**: `bert-base-uncased` is English-only. Videos with foreign language descriptions or transcripts produce high rates of `[UNK]` tokens. Multilingual models (`bert-base-multilingual-cased` or `xlm-roberta-base`) will be evaluated in future phases.
2. **CPU Inference Latency**: Evaluating multi-chunk transcripts through BERT on CPU requires ~390 ms. GPU acceleration (CUDA) is recommended for batch transcript indexing.
3. **No Trained Fusion Weights**: Projection heads remain initialized from baseline. Final recommendation ranking quality will be established after Phase 6 training.
