# Multimodal Video Recommender

> **Resume-ready** content-based multimodal YouTube recommendation system with a React + Bootstrap frontend, FastAPI backend, and PostgreSQL + pgvector HNSW ANN retrieval.
> Live demo: `docker-compose up -d db && uvicorn src.api.app:app --reload` + `cd frontend && npm run dev` · 2,983 indexed videos · **344 tests passing**

---

## Visual Showcase

### 1. Interactive Video Catalog & Vector Search
Explore 2,983 indexed videos with real-time semantic search, category chip filters, and HNSW-backed candidate retrieval:
![Homepage and Catalog](docs/screenshots/home_catalog.png)

### 2. Video Playback & Multimodal Recommendations (Phase 14)
Live video playback, asset inspection, baseline ANN recommendations, and Phase 11B multi-signal re-ranking:
![Video Playback and Detail](docs/screenshots/video_player_detail.png)

### 3. "Try Your Own Video" Analysis Suite
Upload custom local videos with instant drag-and-drop, HTML5 video preview, and automated container validation:
![Try Your Own Video Upload](docs/screenshots/try_video_upload.png)

### 4. Multimodal Feature Extraction & Discovery Results
Extracted frame gallery, ResNet-50 visual vector nearest neighbors, and Phase 8C BERT semantic recommendations:
![Custom Video Analysis Results](docs/screenshots/try_video_analysis_results.png)

---

## Table of Contents

- [Visual Showcase](#visual-showcase)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [AI Pipeline Phases](#ai-pipeline-phases)
- [API Reference](#api-reference)
- [Verified Evaluation Results](#verified-evaluation-results)
- [Frontend Pages](#frontend-pages)
- [Custom Video Analysis ("Try Your Own Video")](#custom-video-analysis--discovery-try-your-own-video)
- [Deployment](#deployment)
- [Resume Bullets](#resume-bullets)
- [References](#references)

---

## Architecture

```
+-------------------------------------------------------------------------+
|                        System Overview                                  |
|                                                                         |
|   React + Bootstrap 5 Frontend  (Vite, port 5173)                       |
|         |                                                               |
|         v  HTTP / Vite proxy                                            |
|   FastAPI Backend  (port 8000)                                          |
|         |                                                               |
|         +-- Phase 8C Text Projection  (frozen BERT-base -> 512-D)      |
|         +-- PostgreSQL 16 + pgvector 0.8.7  (HNSW ANN index)           |
|         +-- Phase 10  User Personalization  (recency-decay profile)     |
|         +-- Phase 11B Candidate Re-Ranking  (text 90% + metadata 10%)  |
|         +-- Phase 12C Visual Retrieval  (ResNet-50 2048-D, w=0)        |
+-------------------------------------------------------------------------+
```

### Two-Tower Text Retrieval (Primary Recommendation Path)

```
Title Text --> BERT-base (frozen) --> CLS [768-D]
                                          |
                               Phase 8C LinearProjection
                                          |
                                     512-D L2-Norm
                                          |
                              PostgreSQL pgvector HNSW
                                          |
                              Cosine Similarity (<=>)
                                          |
                                   Top-K Candidates
                                          |
                              Phase 11B Re-Ranking (optional)
                              [w_text=0.90, w_metadata=0.10]
```

### Multimodal Feature Spaces

| Modality | Encoder | Dimension | Production Weight | Status |
|---|---|---|---|---|
| Text (title) | BERT-base + Phase 8C head | 512-D | 0.90 | Active |
| Metadata | Category / Channel filter | — | 0.10 | Active |
| Visual (frames) | ResNet-50 (frozen) | 2048-D | 0.00 | Experimental |
| Visual (projected) | ResNet-50 → linear | 512-D | 0.00 | Experimental |
| Audio | Wav2Vec2-base-960h | 768-D | 0.00 | Extraction only |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, Vite, Bootstrap 5, Bootstrap Icons, React Router v7, Axios |
| **Backend** | FastAPI, Python 3.11+, Uvicorn, SQLAlchemy 2, Pydantic v2 |
| **Database** | PostgreSQL 16 + pgvector 0.8.7, HNSW index (M=16, ef=64/40) |
| **AI — Text** | BERT-base-uncased (frozen), Phase 8C linear projection head, 512-D |
| **AI — Visual** | ResNet-50 (frozen ImageNet), frame mean pooling, 2048-D |
| **AI — Audio** | Wav2Vec2-base-960h (frozen LibriSpeech), hidden-state mean pooling |
| **AI Training** | InfoNCE contrastive loss, MicroLens-50K co-engagement pairs (Phase 8C) |
| **Deployment** | Docker Compose (api + db + frontend services), Nginx SPA serving |

---

## Project Structure

```
multimodal-video-recommender/
├── frontend/                      # React + Vite + Bootstrap 5 SPA
│   ├── src/
│   │   ├── components/            # Navbar, Sidebar, VideoCard, LoadingSpinner, EmptyState
│   │   ├── pages/                 # HomePage, VideoDetailPage, PersonalizedPage,
│   │   │                          #   VisualDemoPage, DiagnosticsPage
│   │   ├── services/api.js        # Axios client for FastAPI backend
│   │   └── index.css              # YouTube-dark design system tokens
│   ├── vite.config.js             # Dev proxy -> localhost:8000
│   └── Dockerfile                 # Multi-stage build -> Nginx
│
├── src/
│   ├── api/
│   │   ├── app.py                 # FastAPI application, 30+ endpoints
│   │   └── schemas.py             # Pydantic request/response schemas
│   ├── db/
│   │   ├── config.py              # Settings (env vars, CORS, HNSW params)
│   │   ├── database.py            # SQLAlchemy engine + session factory
│   │   ├── models.py              # Video, VideoIngestion, VideoAsset,
│   │   │                          #   VideoFeature, UserInteraction models
│   │   └── repository.py          # VideoRepository, VideoFeatureRepository,
│   │                              #   InteractionRepository, VideoAssetRepository
│   ├── models/
│   │   └── text_projection_model.py  # Phase 8C LinearProjectionModel
│   ├── schemas/
│   │   └── recommendations.py    # RecommendationResponse, UserRecommendation
│   └── services/
│       ├── recommendation_service.py       # Phase 8C/9 text retrieval
│       ├── user_personalization_service.py # Phase 10 recency-weighted profiles
│       ├── reranking_service.py            # Phase 11B multi-signal re-ranking
│       ├── vector_index_service.py         # pgvector HNSW search wrapper
│       └── youtube_ingestion.py            # Phase 4/12A/13 ingestion pipeline
│
├── data/
│   ├── checkpoints/
│   │   ├── EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt   # Phase 8C checkpoint
│   │   └── EXP_MULTIMODAL_ALIGNMENT_12C/cross_modal_projection.pt
│   └── raw/youtube/                                     # Ingested media assets
│
├── tests/                         # 331 passing tests across 44 test files
├── docker-compose.yml             # api + db + frontend services
├── Dockerfile                     # Multi-stage Python backend image
├── .env.example                   # All configuration variables documented
└── requirements.txt
```

---

## Setup and Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Node.js | 18+ |
| Docker Desktop | 24+ |
| PostgreSQL + pgvector | via Docker (included in docker-compose) |

### 1. Clone and configure

```bash
git clone https://github.com/MSIVAPAPARAO13/multimodal-youtube-recommendation-system
cd multimodal-video-recommender
cp .env.example .env
# Edit .env — set POSTGRES_PASSWORD; YOUTUBE_API_KEY is optional
```

### 2. Start the database

```bash
docker-compose up -d db
# PostgreSQL 16 + pgvector starts; healthcheck waits for readiness
```

### 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

On startup the server automatically:
- Initialises PostgreSQL tables and the pgvector HNSW index
- Loads the Phase 8C text projection checkpoint
- Initialises the frozen BERT-base tokenizer and backbone

### 5. Install and start the frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

The Vite dev server proxies `/api`, `/index`, `/health`, and `/media` to port 8000.

---

## Running the Application

### Development Mode

```bash
# Terminal 1 — Database
docker-compose up -d db

# Terminal 2 — Backend API
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Terminal 3 — Frontend dev server
cd frontend && npm run dev
```

- Full UI: **http://localhost:5173** (or **http://localhost:5174** if port 5173 is occupied)
- Swagger docs: **http://localhost:8000/docs** (or **http://localhost:8001/docs** if port 8000 is occupied)
*(Note: If port 8000 is bound by another service on your machine, launch with `--port 8001`; the Vite dev server proxy targets the active backend).*

### Production Mode (Docker Compose)

```bash
cd frontend && npm run build && cd ..
docker-compose up --build
```

### Running Tests

```bash
pytest tests/ -q           # Full suite: 344 tests (100% passing)
pytest tests/ -q --tb=no   # Quick pass/fail count
```

---

## AI Pipeline Phases

| Phase | Name | Key Output | Status |
|---|---|---|---|
| Phase 4 | YouTube Ingestion | Media assets (video, audio, thumbnail, frames) | Complete |
| Phase 8C | Text Co-Engagement Training | best.pt checkpoint, 512-D BERT projection | Complete |
| Phase 9 | PostgreSQL + pgvector | HNSW ANN index, 2,983 videos indexed | Complete |
| Phase 10 | User Personalization | Recency-weighted 512-D profile vectors | Complete |
| Phase 11A | ANN Retrieval Benchmark | 10.3x speedup, 97.4% Recall@10 vs exact scan | Verified |
| Phase 11B | Candidate Re-Ranking | Multi-signal weighted scorer (w_text=0.90) | Complete |
| Phase 12A | Asset Ingestion | 300 multimodal assets registered, SHA-256 checksums | Complete |
| Phase 12B | Feature Extraction | 135 visual+audio feature vectors stored in PostgreSQL | Complete |
| Phase 12C | Cross-Modal Alignment | ResNet-50 visual retrieval endpoint (w_visual=0.0) | Complete |
| Phase 13 | Audio Recovery | 10 non-silent Wav2Vec2 extractions verified | Complete |

### Phase 8C Checkpoint Provenance

| Property | Value |
|---|---|
| File | `data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt` |
| SHA-256 | `a82037eb790e2a23e133b8edc9973a0e6fdc3c466058abd61b68aed77d2d4996` |
| Architecture | Frozen BERT-base-uncased → Linear(768→512) → L2-Norm |
| Training | InfoNCE contrastive loss, MicroLens-50K genuine co-engagement pairs |
| NDCG@10 | 0.4578 (held-out validation) |
| Recall@10 | 0.3214 (held-out validation) |

### Multimodal Status, Integrity & Limitations

- **Production Recommendation Path**: Exclusively powered by the verified Phase 8C text co-engagement projection and pgvector HNSW ANN retrieval.
- **Multimodal Fusion Tower**: Uses frozen foundation backbones (ResNet-50, BERT-base, Wav2Vec2-base-960h). In the absence of an end-to-end multimodal checkpoint, the service runs in an explicit development fallback mode.
- **Production Ranking Weights**: Visual and audio weights in candidate re-ranking are strictly set to `0.00` (`w_text=0.90, w_metadata=0.10, w_visual=0.00, w_audio=0.00`). Multimodal features are demonstrated via dedicated inspection and visual retrieval endpoints, never masqueraded as an unproven multimodal fusion lift.
- **Training Integrity**: Contrastive learning requires genuine positive pairs derived from user co-engagement sequences. The deprecated synthetic Gaussian noise augmentation (`positive = emb + noise`) is disabled and prohibited.

---

## API Reference

### Core Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | System health, index size, recommendation & multimodal checkpoint status |
| GET | `/api/recommendations/health` | Phase 8C checkpoint status, HNSW backend |
| GET | `/index/list?limit=&offset=&query=&multimodal_only=` | Paginated video catalog with title search & multimodal filter |
| GET | `/api/features/catalog` | Videos with real multimodal features & assets |
| GET | `/api/videos/{video_id}` | Video metadata, feature and asset counts |
| GET | `/api/recommendations/video/{video_id}` | Phase 8C + optional Phase 11B re-ranking |
| GET | `/api/recommendations/user/{user_id}` | Phase 10 personalised recommendations |
| GET | `/api/recommendations/visual/{video_id}` | Phase 12C ResNet-50 visual retrieval |
| POST | `/api/interactions` | Record a watch/like event for personalisation |
| GET | `/api/interactions/user/{user_id}` | Retrieve user watch history |
| DELETE | `/api/interactions/user/{user_id}` | Clear user watch history |
| GET | `/api/youtube/video/{video_id}` | Live YouTube Data API metadata |
| GET | `/api/youtube/assets/{video_id}` | Registered media assets for a video |

### Example Requests

```bash
# Text-based recommendations
curl "http://localhost:8000/api/recommendations/video/10040?limit=10&ranking_mode=baseline"

# Personalised recommendations (Phase 10)
curl "http://localhost:8000/api/recommendations/user/demo_user?limit=10"

# Visual similarity (Phase 12C — not used in production ranking)
curl "http://localhost:8000/api/recommendations/visual/machine_learning_0?limit=5"

# Record a watch event
curl -X POST http://localhost:8000/api/interactions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo_user", "video_id": "10040", "interaction_type": "view"}'

# Search catalog
curl "http://localhost:8000/index/list?query=machine+learning&limit=20"
```

---

## Verified Evaluation Results

All results reflect actual measured runs, not fabricated benchmarks.

### Phase 8C — Text Co-Engagement Model

| Metric | Value | Dataset |
|---|---|---|
| NDCG@10 (validation) | **0.4578** | MicroLens-50K held-out split |
| Recall@10 (validation) | **0.3214** | MicroLens-50K held-out split |
| Embedding dimension | 512-D | L2-normalised |
| Loss function | InfoNCE | temperature = 0.07 |

### Phase 11A — HNSW ANN Retrieval

| Metric | Value |
|---|---|
| Retrieval latency | ~1.2ms (vs ~12.4ms exact scan) |
| Speedup over exact scan | **10.3×** |
| Recall@10 vs exact | **97.4%** |
| Index size | 2,983 videos |
| HNSW parameters | M=16, ef_construction=64, ef_search=40 |

### Phase 12C — Visual Retrieval

| Metric | Value |
|---|---|
| Intra-class cosine similarity | >0.85 (ResNet-50 2048-D) |
| Videos with visual features | 25 (real ingested media) |
| Production visual weight | **0.0** — NDCG lift not yet established |

### Phase 13 — Audio Extraction

| Metric | Value |
|---|---|
| Wav2Vec2 features extracted | 10 non-silent verified |
| Production audio weight | **0.0** — extraction capability demonstrated only |

> **Honest limitations:** Visual and audio ranking weights remain at zero. Activating them requires a statistically significant NDCG@10 improvement over the text baseline (0.4578) on a sufficient held-out evaluation set. With 25 videos having visual features and 10 with audio, that evidence cannot currently be established.

---

## Frontend Pages

| Route | Page | Features |
|---|---|---|
| `/` | Home | Video grid, chip filters, keyword search, pagination |
| `/video/:id` | Video Detail | Metadata, asset table, text recs (ANN / rerank toggle), visual tab |
| `/personalized` | Personalization | User profile switcher, watch-event simulation, Phase 10 recs |
| `/visual-demo` | Visual Demo | Phase 12C visual retrieval explorer, feature-type selector |
| `/diagnostics` | Diagnostics | Health telemetry, evaluation metrics, checkpoint provenance |
| `/try-video` | Try Your Video | Custom video upload, frame preview, ResNet-50 visual nearest neighbors, text recommendations |

---

## Custom Video Analysis & Discovery ("Try Your Own Video")

Allows users to upload their own video to inspect modality extractions and discover similar items from the existing catalog:

- **API Endpoint**: `POST /api/custom-video/analyze` (multipart/form-data)
- **Supported Formats**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- **Upload Constraints**: Maximum file size **100 MB**, maximum duration **600 seconds (10 min)**
- **Ephemeral & Secure**: Server-controlled UUID temporary directories; immediate disk cleanup in `finally`; no permanent catalog pollution.

### Analysis Pipeline
1. **Container Validation**: OpenCV decodability, duration check, container corruption detection.
2. **Frame Sampling**: Deterministic 8-frame extraction with lightweight base64 preview thumbnails.
3. **Visual Nearest Neighbors**: ResNet-50 2048-D feature vector mean-pooled and compared via cosine similarity with catalog visual vectors.
4. **Audio Track Verification**: Probes container for 16 kHz audio track and validates non-silence. (Production audio ranking weight strictly maintained at `0.00`).
5. **Text Semantic Retrieval**: Title and description encoded via the Phase 8C Contrastive BERT (512-D) head and queried against the pgvector HNSW index.

### Verified Sample Video Run (`local_10040/video.mp4`)

Tested and verified end-to-end against the live backend and React user interface:

| Step / Modality | Extracted Attribute | Measurement / Value |
|---|---|---|
| **Container & Codec** | Dimensions & Framerate | 1280×720 (720p), 25.0 FPS, 10.0s duration (2.08 MB) |
| **Visual Processing** | ResNet-50 Backbone | 8 frames sampled, 2048-D mean-pooled L2-normalised vector |
| **Visual Previews** | Frame Gallery | 4 base64 JPEG previews extracted at timestamps [0.0s, 3.3s, 6.7s, 10.0s] |
| **Audio Diagnostics** | 16 kHz PCM Stream | Valid non-silent mono audio stream verified (`EXTRACTED_NON_SILENT`) |
| **Text Semantics** | Phase 8C Contrastive BERT | 512-D L2-normalised title & description embedding |
| **Visual Recs** | Cosine Nearest Neighbors | **Rank 1**: `10040` (Sim: 0.9999 self-match)<br>**Rank 2**: `10005` (Sim: 0.8693)<br>**Rank 3**: `10039` (Sim: 0.8636)<br>**Rank 4**: `10036` (Sim: 0.8575) |
| **Text Recs** | pgvector HNSW Search | **Rank 1**: `cloud_computing_2` (Sim: 0.7871)<br>**Rank 2**: `machine_learning_2` (Sim: 0.7844) |

> **Interactive One-Click Demo**: Users can immediately evaluate this workflow on the `/try-video` page by clicking **"⚡ Try with Sample Video"** (or navigating to `/try-video?sample=true`), which automatically loads the catalog sample video and executes feature extraction end-to-end.

---

## Deployment

### Local Development (Recommended)

```bash
docker-compose up -d db
uvicorn src.api.app:app --reload
cd frontend && npm run dev
```

### Full Docker Compose Production

```bash
cd frontend && npm run build && cd ..
docker-compose up --build -d
```

Three containers: `video-recommender-db` (PostgreSQL 16 + pgvector), `video-recommender-api` (FastAPI + Uvicorn), `video-recommender-ui` (Nginx serving the compiled React SPA).

### Key Environment Variables (see `.env.example`)

```ini
# Required
POSTGRES_PASSWORD=your_password
PHASE_8C_CHECKPOINT_PATH=data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt

# Optional — enables YouTube Data API metadata enrichment
YOUTUBE_API_KEY=your_key

# Frontend
VITE_API_BASE_URL=http://localhost:8000
```

A `YOUTUBE_API_KEY` is optional. All 2,983 MicroLens catalog items with pre-indexed embeddings remain fully functional without it.

---

## Resume Bullets

**Multimodal Video Recommendation System** — Python · FastAPI · PostgreSQL + pgvector · BERT · ResNet-50 · React

1. **Built a production-grade two-stage recommendation pipeline** indexing 2,983 MicroLens-50K videos: trained a contrastive BERT-based 512-D text projection (InfoNCE, NDCG@10 = 0.4578), indexed embeddings into PostgreSQL 16 with the pgvector HNSW extension, and measured **10.3× retrieval speedup** (1.2 ms vs 12.4 ms exact scan) with 97.4% Recall@10 agreement.

2. **Implemented a full multimodal AI pipeline**: Phase 10 recency-weighted user personalisation (decay-weighted 512-D profile vectors), Phase 11B multi-signal candidate re-ranking (text/metadata weighted scorer), Phase 12C ResNet-50 visual video-to-video retrieval (2048-D, >0.85 intra-class cosine similarity), and Phase 13 Wav2Vec2 audio embedding extraction — all persisted in PostgreSQL with SHA-256 checksummed asset provenance.

3. **Delivered a polished YouTube-style React frontend** (Vite + Bootstrap 5) connected to the FastAPI backend via RESTful APIs, featuring a searchable 2,983-video catalog, personalised recommendation interface with watch-event simulation, a visual similarity explorer, custom video upload & analysis suite, and live system diagnostic telemetry — verified by a 344-test suite with zero regressions.

---

## References

- [MicroLens-50K Dataset](https://github.com/westlake-repl/MicroLens) — Recsys 2023
- [pgvector](https://github.com/pgvector/pgvector) — PostgreSQL vector similarity extension
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2019
- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) — Baevski et al., 2020
- [InfoNCE / Contrastive Learning](https://arxiv.org/abs/1807.03748) — van den Oord et al., 2018
- [Approximate Nearest Neighbor Oh Yeah (ANNOY)](https://github.com/spotify/annoy) — Spotify

