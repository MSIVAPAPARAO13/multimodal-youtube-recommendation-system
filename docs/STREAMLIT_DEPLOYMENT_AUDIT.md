# Streamlit Community Cloud Deployment Audit Report

**Date**: 2026-10-12  
**Target Repository**: [multimodal-youtube-recommendation-system](https://github.com/MSIVAPAPARAO13/multimodal-youtube-recommendation-system)  
**Target Platform**: Streamlit Community Cloud  
**Main Entry Point**: `streamlit_app.py`  
**Target Python Version**: 3.11  

---

## 1. Repository Structure & Checkpoint Verification

| Component | Verified Path | State / Checkpoint Details |
|---|---|---|
| **Streamlit Entrypoint** | `streamlit_app.py` | Exists at repository root (548 lines) |
| **Phase 8C Checkpoint** | `data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt` | Verified present (1.57 MB, SHA-256: `a82037eb790e...`) |
| **MicroLens Catalog** | `data/raw/microlens/MicroLens-50k_titles.csv` | Verified present (2.39 MB, tracked in Git) |
| **Evaluation Fixtures** | `data/fixtures/synthetic_eval_videos.json` | Verified present (tracked in Git) |
| **Local Video Media** | `data/raw/microlens/videos/*.mp4` (10 items) | Tracked in Git (`10040.mp4`, `10005.mp4`, etc.) |
| **Automated Tests** | `tests/` | 344 tests passing (100% pass rate) |

---

## 2. Recommendation & Data Retrieval Flow

1. **Catalog Metadata**:
   - Primary: Queried via PostgreSQL `Video` table (2,983 indexed videos).
   - Fallback for Cloud/Standalone: Ingested titles from `MicroLens-50k_titles.csv` and `synthetic_eval_videos.json`.
2. **Text Embeddings & Recommendations**:
   - Primary: Phase 8C Contrastive BERT (frozen BERT-base + Linear 768→512) queried against PostgreSQL `pgvector` HNSW index (`embedding <=> query_vec`).
   - Re-Ranking: Phase 11B multi-signal re-ranking ($w_{text}=0.90, w_{meta}=0.10, w_{visual}=0.00, w_{audio}=0.00$).
3. **Visual & Audio Processing**:
   - ResNet-50 2048-D mean-pooled visual vector across sampled frames.
   - Wav2Vec2 / PCM audio stream extraction and silence checking (`AudioExtractor`).
   - Production weights strictly maintained at $w_{visual}=0.00$ and $w_{audio}=0.00$.

---

## 3. Cloud Incompatibilities & Gaps Identified

1. **Localhost Hardcoding**:
   - `streamlit_app.py` previously attempted to probe `http://localhost:8001` and defaulted to it, which will fail on Streamlit Community Cloud.
   - **Correction**: Must use configurable `API_BASE_URL` (or `FASTAPI_URL`). Never default to localhost in cloud mode.
2. **Missing `packages.txt`**:
   - Streamlit Cloud runs on Debian/Ubuntu Linux where `import cv2` fails without `libgl1` and `libglib2.0-0`, and audio extraction requires `ffmpeg`.
   - **Correction**: Add `packages.txt` containing `ffmpeg`, `libgl1`, and `libglib2.0-0`.
3. **`DATABASE_URL` Environment Variable Parsing**:
   - `src/db/config.py` constructed database URLs from discrete host/port parameters and ignored standard URI format `DATABASE_URL=postgresql://...`.
   - **Correction**: Support standard `DATABASE_URL` connection strings from Streamlit Cloud secrets (Neon, Supabase, RDS).
4. **Standalone Zero-Crash Demo Mode**:
   - When deployed without an external database or backend URL configured, the app must not crash. It must provide an honest, self-contained demonstration using the verified local catalog and Phase 8C checkpoint with clear diagnostic explanations.
5. **Watch Links & Video Player**:
   - Must only display clickable YouTube links when a valid 11-character YouTube video ID exists, falling back cleanly to catalog playback.
