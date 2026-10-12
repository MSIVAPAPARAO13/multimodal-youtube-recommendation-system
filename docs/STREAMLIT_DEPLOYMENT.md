# Streamlit Deployment Guide — Multimodal Video Recommender

This guide covers deploying the interactive **Multimodal Video Recommendation System** using Streamlit across local environments, **Streamlit Community Cloud**, **Docker**, and **Hugging Face Spaces**.

---

## 1. Quickstart: Local Execution

### Prerequisites
- Python 3.10+
- Installed dependencies: `pip install -r requirements.txt`
- (Optional) Running FastAPI backend (`uvicorn src.api.app:app --port 8001`) and PostgreSQL 16 + pgvector container.

### Launch Streamlit
```bash
streamlit run streamlit_app.py
```
Or via Python module:
```bash
python -m streamlit run streamlit_app.py --server.port 8501
```

Access the UI at: **http://localhost:8501**

---

## 2. Deploying to Streamlit Community Cloud (1-Click Free Hosting)

Streamlit Community Cloud allows free hosting directly connected to your GitHub repository.

### Step 1: Push Repository to GitHub
Ensure your repository is pushed to GitHub with `main` branch:
```bash
git push origin main
```

### Step 2: Create App on Streamlit Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `MSIVAPAPARAO13/multimodal-youtube-recommendation-system`.
4. Branch: `main`.
5. Main file path: `streamlit_app.py`.
6. Click **"Advanced settings"** to configure secrets.

### Step 3: Configure Secrets (Optional but Recommended)
In the Streamlit Cloud dashboard under **App Settings > Secrets**, paste your configuration:

```toml
# Optional: Connect to your hosted FastAPI backend (e.g. Railway, Render, Fly.io, or AWS)
FASTAPI_URL = "https://your-api-domain.com"

# Optional: Connect directly to a managed PostgreSQL + pgvector instance (Neon, Supabase, AWS RDS)
DATABASE_URL = "postgresql://postgres:your_password@your-db-host:5432/video_recommender"

# Optional: YouTube Data API key for live video enrichment
YOUTUBE_API_KEY = "your-api-key"
```

### Step 4: Deploy
Click **"Deploy!"**. Streamlit Cloud installs `requirements.txt` and starts `streamlit_app.py`.

---

## 3. Configuration & Theme Setup

The application is pre-configured via [.streamlit/config.toml](file:///c:/Users/msiva/Music/multimodal-video-recommender/.streamlit/config.toml):

```toml
[theme]
primaryColor = "#ff0000"              # YouTube red accent
backgroundColor = "#0f0f0f"           # Dark theme background
secondaryBackgroundColor = "#212121"  # Elevated card surface
textColor = "#f1f1f1"                 # Crisp light text
font = "sans serif"

[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 100                   # Supports 100 MB video uploads

[browser]
gatherUsageStats = false
```

---

## 4. Deploying with Docker

You can run the Streamlit application alongside PostgreSQL and FastAPI in Docker Compose.

### Dockerfile Command for Streamlit
```dockerfile
# Run Streamlit on container port 8501
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Standalone Docker Run
```bash
docker build -t video-recommender-streamlit -f Dockerfile .
docker run -p 8501:8501 -e FASTAPI_URL=http://host.docker.internal:8001 video-recommender-streamlit
```

---

## 5. Application Features & Tab Walkthrough

| Tab | Capability | Underlying Technology |
|---|---|---|
| **🔍 Catalog & Search** | Paginated catalog browser, real-time title search, category filter, multimodal asset toggle | 2,983 MicroLens videos, pgvector HNSW candidate index |
| **🎯 Recommendation Studio** | Video-to-Video recommendations with baseline ANN and Phase 11B Candidate Re-Ranking sliders | Phase 8C BERT 512-D projection, multi-signal scorer ($w_{text}=0.90, w_{meta}=0.10$) |
| **👤 User Personalization** | Watch-event simulation and recency-decay personalized recommendation feeds | Phase 10 dynamic 512-D user profile vectors ($e^{-\lambda \Delta t}$) |
| **📹 Try Your Own Video** | Upload custom video (up to 100 MB) or 1-click sample test, extracted frames, audio silence verification, dual recommendations | OpenCV decodability, ResNet-50 2048-D mean-pooled vector, AudioExtractor 16 kHz stream check, BERT 512-D projection |
| **📊 Architecture & Telemetry** | Model checkpoint provenance, SHA-256 integrity, HNSW latency benchmarks (1.2ms vs 12.4ms exact scan) | Scientific evaluation verification table |

---

## 6. Resilience & Graceful Fallback Strategy

The Streamlit deployment is architected for zero-crash fault tolerance:
1. **Live Backend Mode**: Detects local or hosted FastAPI instance (`http://localhost:8001`, `http://localhost:8000`, or `FASTAPI_URL`), reusing all existing API endpoints without duplicating logic.
2. **Direct DB Mode**: If FastAPI is offline, connects to PostgreSQL via `DATABASE_URL` using SQLAlchemy ORM and pgvector.
3. **Offline Demo Mode**: If neither is accessible, gracefully renders cached catalog telemetry with informative status badges rather than raising unhandled connection exceptions.
