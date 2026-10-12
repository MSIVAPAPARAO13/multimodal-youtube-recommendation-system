# Phase 6 — Vision + Thumbnail Intelligence Pipeline

## 1. Executive Overview

Phase 6 elevates the visual modality of the Multimodal YouTube Recommendation System from a static single-frame representation into a production-grade, deterministic, video-level visual feature pipeline. It integrates both sequential multi-frame video extraction and high-resolution YouTube thumbnail intelligence while keeping torchvision **ResNet-50** as the production baseline.

---

## 2. Architecture & Data Flow

```
YouTube MP4
    ↓
Frame Sampler (Deterministic 8-frame sequential seeking)
    ↓
Sampled RGB Frames
    ↓
ImageNet Preprocessing (Resize/Crop to 224x224, Normalized)
    ↓
ResNet-50 Backbone (Frozen ImageNet-1K)
    ↓
2048-D Raw CNN Feature Representation
    ↓
512-D Shared Projection Layer
    ↓
Frame Pooling (Mean / Temporal Decay)
    ↓
Video Visual Embedding (512-D)
         +
YouTube Thumbnail (Validated local thumbnail.jpg)
    ↓
ImageNet Preprocessing (Identical Normalization)
    ↓
ResNet-50 + 512-D Projection
    ↓
Thumbnail Embedding (512-D)
         ↓
Video + Thumbnail Visual Fusion (0.7 * Video + 0.3 * Thumbnail)
         ↓
L2 Normalized 512-D Vision Embedding
```

---

## 3. Core Components

### A. FrameSampler (`src/features/frame_sampler.py`)
- **Deterministic Sampling**: Samples evenly-spaced frames using `np.linspace(0, total_frames - 1, num_frames, dtype=int)`. Default: 8 frames.
- **Low Memory Overhead**: Leverages OpenCV sequential/index-based seeking (`cap.set(cv2.CAP_PROP_POS_FRAMES, idx)`), avoiding loading entire MP4 video files into memory.
- **Short Video Handling**: When a video contains fewer frames than requested, it captures all available valid frames and deterministically duplicates the last valid frame to preserve downstream tensor dimensions without failing.
- **Partial Decode Tolerance**: If a single frame fails decoding, it falls back to the adjacent valid frame or black frame, recording the exact decoded count in diagnostics without crashing the recommendation service.
- **Rich Diagnostics**: Returns `requested_frames`, `decoded_frames`, `frame_indices`, `total_frames`, `fps`, `duration_seconds`, `video_path`, `success`, and `error_message`.

### B. VisualProcessor (`src/features/visual_processor.py`)
- **Strict ImageNet Preprocessing**: Preprocesses all visual media into `(3, 224, 224)` RGB tensors.
- **Color Format Standardization**: Converts PIL, numpy arrays, BGR OpenCV frames, RGBA, and grayscale inputs to RGB.
- **Canonical ImageNet Normalization**:
  $$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$
  Applied identically to both video frames and YouTube thumbnails to ensure representation parity.

### C. ResNet-50 Production Baseline (`src/features/visual_encoder.py`)
- **Model**: Pretrained torchvision `models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`.
- **Frozen Backbone**: All backbone convolutional weights are frozen (`param.requires_grad = False`). Inference runs in `model.eval()` and `torch.no_grad()`.
- **Feature Extraction**: Extracts the 2048-dimensional feature representation before the classification layer.
- **Projection**: Maps 2048-D features to the system's 512-D shared multimodal embedding space via a 2-layer MLP (`Linear(2048, 2048) -> ReLU -> Dropout(0.1) -> Linear(2048, 512)`).

### D. Multi-Frame Aggregation (`src/features/visual_encoder.py`)
Aggregates $T$ frame embeddings (each 512-D) into a single 512-D video representation:
1. **Mean Pooling (Default)**:
   $$\mathbf{e}_{\text{video}} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{e}_{t}$$
2. **Temporal Decay / Weighted Pooling**:
   Applies exponential decay giving higher emphasis to later frames:
   $$w_t = \frac{\exp(0.2 \cdot (t - (T - 1)))}{\sum_j \exp(0.2 \cdot (j - (T - 1)))}, \quad \mathbf{e}_{\text{video}} = \sum_{t=1}^{T} w_t \mathbf{e}_{t}$$

### E. YouTube Thumbnail Intelligence (`src/features/thumbnail_encoder.py`)
- Uses validated local thumbnails stored during Phase 4 (`data/raw/youtube/{video_id}/thumbnail.jpg`).
- Never downloads remote thumbnails if a validated local file is available.
- Passes thumbnail through the identical ImageNet preprocessing and ResNet-50 encoder to yield a 512-D thumbnail embedding.
- Missing or corrupted thumbnails fail gracefully without failing the recommendation request.

### F. Video + Thumbnail Visual Fusion (`src/services/multimodal_inference.py`)
Combines video and thumbnail features deterministically:
$$\mathbf{e}_{\text{visual}} = \begin{cases}
\text{Normalize}(0.7 \cdot \mathbf{e}_{\text{video}} + 0.3 \cdot \mathbf{e}_{\text{thumb}}), & \text{if both available} \\
\text{Normalize}(\mathbf{e}_{\text{video}}), & \text{if video only} \\
\text{Normalize}(\mathbf{e}_{\text{thumb}}), & \text{if thumbnail only} \\
\mathbf{0}_{512}, & \text{if neither available (is\_available=False)}
\end{cases}$$
Weights are configurable via settings (`VISUAL_VIDEO_WEIGHT=0.7`, `VISUAL_THUMBNAIL_WEIGHT=0.3`).

### G. Missing Modality Handling
The visual service never raises uncaught exceptions on missing media:
- Missing video, corrupt video, or 0 decoded frames return `is_available=False` with zero vector and structured diagnostics.
- Missing or corrupted thumbnail does not affect video encoding.
- The downstream `GatedFusion` layer cleanly ignores missing modalities via zero-masking.

### H. Experimental CLIP Encoder (`src/features/visual_encoder.py`)
- Abstraction: `BaseVisualEncoder` interface implemented by `ResNet50VisualEncoder` and `CLIPExperimentalEncoder`.
- Factory function: `get_visual_encoder("resnet50")` or `get_visual_encoder("clip")`.
- Default: `VISUAL_ENCODER=resnet50`. CLIP remains strictly opt-in and experimental.
- Offline Safety: `pretrained=False` uses local architectural configurations without downloading remote checkpoints.

---

## 4. Configuration

Configured via typed Pydantic settings in `src/db/config.py`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `VISUAL_ENCODER` | `str` | `resnet50` | Active visual encoder backbone (`resnet50`, `clip`) |
| `VISUAL_FRAME_COUNT` | `int` | `8` | Number of video frames to sample |
| `VISUAL_FRAME_POOLING` | `str` | `mean` | Frame aggregation strategy (`mean`, `decay`, `max`) |
| `VISUAL_VIDEO_WEIGHT` | `float` | `0.7` | Weight of video frames in visual fusion |
| `VISUAL_THUMBNAIL_WEIGHT` | `float` | `0.3` | Weight of thumbnail in visual fusion |

---

## 5. Empirical Benchmarks (CPU)

Measured on Python 3.13 / PyTorch 2.14.1 (CPU execution, CUDA unavailable):

| Pipeline Stage | Average Latency |
| :--- | :--- |
| **1. Frame Sampling (OpenCV, 8 frames from MP4)** | **37.50 ms** |
| **2. Image Preprocessing (ImageNet 224x224)** | **20.33 ms** |
| **3. ResNet-50 Frame Inference (2048 → 512)** | **345.00 ms** |
| **4. Frame Aggregation (Mean Pooling)** | **0.15 ms** |
| **5. Thumbnail Inference (ImageNet + ResNet-50)** | **62.94 ms** |
| **6. Total Video Visual Embedding Latency** | **411.68 ms** |
| **7. Video + Thumbnail Visual Fusion Latency** | **0.16 ms** |
| **Total Video + Thumbnail Latency** | **474.77 ms** |

---

## 6. Synthetic Visual Evaluation Benchmark

Evaluated on 15 synthetic visual media bundles across 5 distinct categories (`programming`, `cybersecurity`, `machine_learning`, `web_development`, `cloud_computing`):

- **512-D Dimension**: 100% Passed (15/15)
- **L2 Hypersphere Normalization**: 100% Passed (15/15)
- **Deterministic Repeatability**: 100% Passed (15/15, Cosine Similarity = 1.0000)
- **NaN / Inf Immunity**: 100% Passed (15/15)
- **Precision@1**: 1.0000 (100.0%)
- **Precision@3**: 0.6667 (66.7% - exactly 2 out of 3 retrieved items are relevant ground truth)
- **Recall@3**: 1.0000 (100.0%)
- **MRR**: 1.0000
- **MAP**: 1.0000

*Notice: This benchmark tests representation quality, mathematical consistency, and nearest-neighbor pipeline functionality on synthetic geometric patterns. It does not claim real-world YouTube recommendation accuracy.*

---

## 7. Checkpoint Policy & Limitations

- **Frozen Pretrained Backbone**: ResNet-50 uses torchvision ImageNet-1K pretrained weights. No fake trained recommendation checkpoints are used.
- **Untrained Fusion / Projection**: The 2048 → 512 projection layer and multimodal fusion layer currently operate in their development baseline state until Phase 8 multimodal training.
- **CPU Execution**: Latency reflects single-thread CPU execution. With CUDA, batch frame inference latency is expected to decrease significantly.
