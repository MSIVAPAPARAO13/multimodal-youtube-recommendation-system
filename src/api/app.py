"""
FastAPI Application - Stabilized Multimodal Video Recommender
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    VideoInput,
    RecommendRequest,
    SimilarityRequest,
    Recommendation,
    RecommendResponse,
    SimilarityResponse,
    HealthResponse,
    IndexResponse,
    IndexMicroLensRequest,
    IndexMicroLensResponse,
    YouTubeVideoResponse,
    YouTubeIngestRequest,
    YouTubeIngestResponse,
    YouTubeIngestStatusResponse,
    RecommendationItem,
    SimilarVideosRequest,
    RecommendationResponse,
    RecommendationHealthResponse,
    UserRecommendationResponse,
    VideoAssetsResponse,
    VideoAssetItem,
    BatchIngestRequest,
    BatchIngestResponse,
    IngestionListResponse,
    IngestionSummaryItem,
    VisualRetrievalResponse,
    VisualRetrievalItem,
    CustomVideoAnalysisResponse,
    CustomVideoRecommendationItem,
)
from integrations.youtube import (
    YouTubeMetadataService,
    YouTubeAPIError,
    YouTubeAuthenticationError,
    YouTubeConfigurationError,
    YouTubeInvalidVideoIdError,
    YouTubeNotFoundError,
    YouTubePrivateVideoError,
    YouTubeQuotaExceededError,
)
from db.config import get_settings
from db.database import get_database, init_database
from db.models import Video
from db.repository import VideoRepository
from data.processors import VideoProcessor, AudioProcessor
from utils.device import get_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

settings = get_settings()


# ============ Dependency ============
def get_db():
    """FastAPI dependency for database sessions."""
    db = get_database()
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ============ App Setup ============
app = FastAPI(
    title=settings.API_TITLE,
    description="Content-based multimodal video recommender with PostgreSQL + pgvector",
    version=settings.API_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Static media serving for ingested assets
_media_dir = Path("data/raw/youtube")
if _media_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/media", StaticFiles(directory=str(_media_dir)), name="media")

# Static frontend distribution serving
_frontend_dist = Path("frontend/dist")
if _frontend_dist.exists() and (_frontend_dist / "assets").exists():
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="spa-assets")

    @app.get("/app", include_in_schema=False)
    @app.get("/app/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = ""):
        return FileResponse(_frontend_dist / "index.html")


# ============ Global Model State ============
class ModelState:
    def __init__(self):
        self.model = None
        self.inference_service = None
        self.recommendation_service = None
        self.personalization_service = None
        self.device = None
        self.video_processor = None
        self.audio_processor = None
        self.is_trained = False

    def is_loaded(self) -> bool:
        return (
            self.model is not None
            or self.inference_service is not None
            or self.recommendation_service is not None
            or self.personalization_service is not None
        )


state = ModelState()


def get_recommendation_service():
    """Dependency returning active RecommendationService instance."""
    if state.recommendation_service is None:
        from src.services.recommendation_service import RecommendationService
        state.recommendation_service = RecommendationService(settings=settings, device=state.device)
    return state.recommendation_service


def get_personalization_service():
    """Dependency returning active UserPersonalizationService instance."""
    if state.personalization_service is None:
        from src.services.user_personalization_service import UserPersonalizationService
        rec_svc = get_recommendation_service()
        state.personalization_service = UserPersonalizationService(
            settings=settings,
            recommendation_service=rec_svc,
            vector_index_service=rec_svc.vector_index_service,
        )
    return state.personalization_service




def validate_media_path(path_str: Optional[str]) -> Optional[Path]:
    """
    Validate and constrain media path to prevent arbitrary filesystem access.
    """
    if not path_str:
        return None

    raw_path = Path(path_str)
    resolved = raw_path.resolve()
    project_root = Path(__file__).resolve().parent.parent.parent

    # Allowed paths: project root and configured subdirectories
    allowed_dirs = [project_root / d for d in settings.ALLOWED_MEDIA_DIRS]
    allowed_dirs.append(project_root)

    is_allowed = any(
        resolved == allowed or allowed in resolved.parents
        for allowed in allowed_dirs
    )

    if not is_allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Security violation: Media path '{path_str}' is outside allowed directories."
        )

    return resolved


# ============ Startup ============
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    try:
        await run_in_threadpool(init_database)
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        logger.info("Ensure PostgreSQL is running: docker-compose up -d db")

    logger.info("Initializing MultimodalInferenceService...")
    state.device = get_device()
    state.video_processor = VideoProcessor(num_frames=8, frame_size=(224, 224))
    state.audio_processor = AudioProcessor(sample_rate=16000, duration=5.0)

    try:
        from src.services.multimodal_inference import MultimodalInferenceService
        state.inference_service = MultimodalInferenceService(
            settings=settings,
            device=state.device
        )
        state.model = state.inference_service.model
        state.is_trained = state.inference_service.is_checkpoint_loaded
        if not state.inference_service.is_checkpoint_loaded:
            logger.info(
                "[DEVELOPMENT FALLBACK] Multimodal fusion checkpoint not loaded from MODEL_PATH. "
                "Foundation backbones (ResNet, BERT, Wav2Vec2) are active, but multimodal projection/fusion heads "
                "are untrained initializations. Primary recommendations are reliably powered by Phase 8C text co-engagement projection."
            )
        else:
            logger.info("MultimodalInferenceService trained checkpoint loaded successfully.")
    except Exception as e:
        logger.error("Error loading MultimodalInferenceService: %s", e, exc_info=True)
        state.model = None
        state.inference_service = None
        state.is_trained = False

    logger.info("Initializing RecommendationService (Phase 9)...")
    try:
        from src.services.recommendation_service import RecommendationService
        state.recommendation_service = RecommendationService(
            settings=settings,
            device=state.device
        )
        logger.info(
            "RecommendationService initialized successfully (checkpoint_loaded=%s, sha256=%s)",
            state.recommendation_service.is_checkpoint_loaded,
            state.recommendation_service.checkpoint_sha256[:12] if state.recommendation_service.checkpoint_sha256 else "none"
        )
    except Exception as e:
        logger.warning("Could not initialize RecommendationService: %s", e)
        state.recommendation_service = None

    logger.info("Initializing UserPersonalizationService (Phase 10)...")
    try:
        from src.services.user_personalization_service import UserPersonalizationService
        if state.recommendation_service is not None:
            state.personalization_service = UserPersonalizationService(
                settings=settings,
                recommendation_service=state.recommendation_service,
                vector_index_service=state.recommendation_service.vector_index_service,
            )
            logger.info("UserPersonalizationService initialized successfully")
    except Exception as e:
        logger.warning("Could not initialize UserPersonalizationService: %s", e)
        state.personalization_service = None




# ============ Endpoints ============

@app.get("/")
async def root():
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "environment": settings.APP_ENV,
        "storage": "PostgreSQL + pgvector",
        "app_url": "/app",
        "docs_url": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health(db: Session = Depends(get_db)):
    repo = VideoRepository(db)
    try:
        index_size = await run_in_threadpool(repo.count)
    except Exception as e:
        logger.warning("Could not query video count in healthcheck: %s", e)
        index_size = 0

    rec_checkpoint_loaded = (
        state.recommendation_service.is_checkpoint_loaded
        if state.recommendation_service is not None
        else False
    )
    multi_checkpoint_loaded = (
        state.inference_service.is_checkpoint_loaded
        if state.inference_service is not None
        else False
    )
    checkpoint_prov = (
        f"Phase 8C Text Co-Engagement ({state.recommendation_service.checkpoint_sha256[:12]})"
        if (state.recommendation_service and state.recommendation_service.checkpoint_sha256)
        else "None"
    )

    return HealthResponse(
        status="ok",
        model_loaded=state.is_loaded(),
        model_trained=rec_checkpoint_loaded,
        recommendation_checkpoint_loaded=rec_checkpoint_loaded,
        multimodal_checkpoint_loaded=multi_checkpoint_loaded,
        checkpoint_provenance=checkpoint_prov,
        environment=settings.APP_ENV,
        index_size=index_size,
        device=str(state.device) if state.device else "unknown",
        storage="postgresql"
    )


@app.post("/index/microlens", response_model=IndexMicroLensResponse)
async def index_microlens(request: IndexMicroLensRequest, db: Session = Depends(get_db)):
    """
    Index videos from MicroLens dataset into PostgreSQL.
    """
    if not state.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    import pandas as pd

    repo = VideoRepository(db)
    data_dir = Path("data/raw/microlens")
    video_dir = data_dir / "videos"

    if not (data_dir / "MicroLens-50k_titles.csv").exists():
        raise HTTPException(status_code=404, detail="MicroLens titles CSV not found")

    titles_df = pd.read_csv(data_dir / "MicroLens-50k_titles.csv")
    available = {int(p.stem) for p in video_dir.glob("*.mp4")}
    titles_df = titles_df[titles_df["item"].isin(available)].head(request.max_videos)

    indexed = 0
    skipped = 0

    for _, row in titles_df.iterrows():
        video_id = str(row["item"])
        title = row["title"]
        video_path = video_dir / f"{video_id}.mp4"

        existing = await run_in_threadpool(repo.get_by_video_id, video_id)
        if existing and existing.embedding is not None:
            skipped += 1
            continue

        try:
            # Extract frames and audio off the main event loop
            frames = await run_in_threadpool(state.video_processor.extract_frames, str(video_path))
            audio = await run_in_threadpool(state.audio_processor.extract_audio, str(video_path))

            frames = frames.unsqueeze(0).to(state.device)
            audio = audio.unsqueeze(0).to(state.device)

            with torch.no_grad():
                embedding = await run_in_threadpool(
                    state.model.encode_video,
                    frames,
                    audio,
                    [title]
                )

            embedding_list = embedding.squeeze(0).cpu().tolist()

            if existing:
                await run_in_threadpool(repo.update_embedding, video_id, embedding_list)
            else:
                await run_in_threadpool(
                    repo.create,
                    video_id=video_id,
                    title=title,
                    video_path=str(video_path),
                    embedding=embedding_list
                )

            indexed += 1
            logger.info("Indexed: %s - %s...", video_id, title[:40])

        except Exception as e:
            logger.error("Failed to index %s: %s", video_id, e)

    total_count = await run_in_threadpool(repo.count)
    return IndexMicroLensResponse(
        indexed=indexed,
        skipped=skipped,
        total_in_index=total_count,
        message=f"Indexed {indexed} videos, skipped {skipped} (already in DB)"
    )


@app.post("/index", response_model=IndexResponse)
async def index_video(video: VideoInput, db: Session = Depends(get_db)):
    """Index a single video by path or metadata."""
    if not state.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    repo = VideoRepository(db)
    validated_path = validate_media_path(video.video_path)
    metadata = {
        "title": video.title,
        "description": video.description
    }

    if state.inference_service is not None:
        result = await run_in_threadpool(
            state.inference_service.encode_multimodal,
            video_path=validated_path if (validated_path and validated_path.exists()) else None,
            text=metadata
        )
        embedding_list = result.embedding_list
    else:
        # Fallback when running with mocked model
        if validated_path and validated_path.exists():
            frames = await run_in_threadpool(state.video_processor.extract_frames, str(validated_path))
            audio = await run_in_threadpool(state.audio_processor.extract_audio, str(validated_path))
            frames = frames.unsqueeze(0).to(state.device)
            audio = audio.unsqueeze(0).to(state.device)
            embedding = await run_in_threadpool(
                state.model.encode_video,
                frames,
                audio,
                [metadata]
            )
        else:
            dummy_frames = torch.zeros(1, 8, 3, 224, 224).to(state.device)
            dummy_audio = torch.zeros(1, 80000).to(state.device)
            embedding = await run_in_threadpool(
                state.model.encode_video,
                dummy_frames,
                dummy_audio,
                [metadata]
            )
        embedding_list = embedding.squeeze(0).cpu().tolist()

    existing = await run_in_threadpool(repo.get_by_video_id, video.video_id)
    if existing:
        await run_in_threadpool(repo.update_embedding, video.video_id, embedding_list)
    else:
        await run_in_threadpool(
            repo.create,
            video_id=video.video_id,
            title=video.title,
            description=video.description,
            video_path=str(validated_path) if validated_path else None,
            embedding=embedding_list
        )

    current_count = await run_in_threadpool(repo.count)
    return IndexResponse(
        video_id=video.video_id,
        indexed=True,
        index_size=current_count,
        message="Video indexed successfully"
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest, db: Session = Depends(get_db)):
    """Get recommendations for a video using pgvector similarity search."""
    if not state.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not state.is_trained and settings.APP_ENV == "production":
        raise HTTPException(
            status_code=503,
            detail="Trained model checkpoint is required in production."
        )

    repo = VideoRepository(db)

    # Query video
    query_video = await run_in_threadpool(repo.get_by_video_id, request.video_id)
    if not query_video:
        raise HTTPException(status_code=404, detail=f"Video '{request.video_id}' not found")

    if query_video.embedding is None:
        raise HTTPException(status_code=400, detail=f"Video '{request.video_id}' has no embedding")

    total_videos = await run_in_threadpool(repo.count)
    if total_videos < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 videos in index for recommendations")

    # Similarity search
    try:
        similar_videos = await run_in_threadpool(
            repo.find_similar,
            embedding=query_video.embedding,
            limit=request.top_k,
            exclude_video_id=request.video_id
        )
    except Exception as e:
        logger.warning("pgvector search failed, falling back to NumPy similarity: %s", e)
        db.rollback()
        similar_videos = await run_in_threadpool(
            repo.find_similar_fallback,
            embedding=query_video.embedding,
            limit=request.top_k,
            exclude_video_id=request.video_id
        )

    recommendations = [
        Recommendation(
            video_id=video.video_id,
            score=round(float(score), 4),
            title=video.title[:60] if video.title else ""
        )
        for video, score in similar_videos
    ]

    return RecommendResponse(
        query_video_id=request.video_id,
        query_title=query_video.title[:60] if query_video.title else "",
        recommendations=recommendations,
        storage="postgresql"
    )


@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest, db: Session = Depends(get_db)):
    """Compute cosine similarity between two indexed videos."""
    repo = VideoRepository(db)
    v1 = await run_in_threadpool(repo.get_by_video_id, request.video_id_1)
    v2 = await run_in_threadpool(repo.get_by_video_id, request.video_id_2)

    if not v1 or v1.embedding is None:
        raise HTTPException(status_code=404, detail=f"Video '{request.video_id_1}' not found or has no embedding")
    if not v2 or v2.embedding is None:
        raise HTTPException(status_code=404, detail=f"Video '{request.video_id_2}' not found or has no embedding")

    emb1 = torch.tensor(v1.embedding).unsqueeze(0)
    emb2 = torch.tensor(v2.embedding).unsqueeze(0)

    if state.model is not None:
        sim = state.model.get_similarity(emb1, emb2).item()
    else:
        # Fallback direct cosine dot product
        sim = (emb1 * emb2).sum().item()

    return SimilarityResponse(
        video_id_1=request.video_id_1,
        video_id_2=request.video_id_2,
        similarity=round(float(sim), 4)
    )


@app.get("/index/list")
async def list_videos(
    limit: int = Query(default=100, ge=1, le=500, description="Max videos to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    query: Optional[str] = Query(default=None, description="Optional title search query"),
    multimodal_only: bool = Query(default=False, description="Filter only videos with multimodal features or assets"),
    db: Session = Depends(get_db),
):
    """List indexed videos from database with optional search, multimodal filtering, and pagination."""
    from src.db.models import VideoFeature, VideoAsset, Video
    from sqlalchemy import select, func

    # Query distinct videos with multimodal features/assets
    feat_vids = set(db.execute(select(VideoFeature.video_id).where(
        VideoFeature.is_valid.is_(True),
        VideoFeature.is_stale.is_(False)
    ).distinct()).scalars().all())
    asset_vids = set(db.execute(select(VideoAsset.video_id).distinct()).scalars().all())
    multimodal_set = feat_vids.union(asset_vids)

    stmt = select(Video)
    count_stmt = select(func.count(Video.id))

    if query:
        stmt = stmt.where(Video.title.ilike(f"%{query}%"))
        count_stmt = count_stmt.where(Video.title.ilike(f"%{query}%"))

    if multimodal_only:
        stmt = stmt.where(Video.video_id.in_(list(multimodal_set)))
        count_stmt = count_stmt.where(Video.video_id.in_(list(multimodal_set)))

    total_count = db.scalar(count_stmt) or 0
    stmt = stmt.offset(offset).limit(limit)
    videos = list(db.execute(stmt).scalars().all())

    return {
        "videos": [
            {
                "video_id": v.video_id,
                "youtube_video_id": v.youtube_video_id,
                "title": v.title if v.title else "",
                "description": v.description,
                "channel_title": v.channel_title,
                "thumbnail_url": v.thumbnail_url,
                "category_id": v.category_id,
                "duration": v.duration,
                "view_count": v.view_count,
                "has_embedding": v.embedding is not None,
                "has_visual": v.video_id in feat_vids,
                "has_assets": v.video_id in asset_vids,
            }
            for v in videos
        ],
        "total": total_count,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/videos/{video_id:path}")
async def get_video_details(video_id: str, db: Session = Depends(get_db)):
    """Retrieve detailed video record and feature availability from local catalog."""
    repo = VideoRepository(db)
    video = await run_in_threadpool(repo.get_by_video_id, video_id)
    if not video:
        video = await run_in_threadpool(repo.get_by_youtube_id, video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found in catalog")

    from src.db.repository import VideoFeatureRepository, VideoAssetRepository
    feat_repo = VideoFeatureRepository(db)
    features = await run_in_threadpool(feat_repo.get_features_for_video, video.video_id)
    asset_repo = VideoAssetRepository(db)
    assets = await run_in_threadpool(asset_repo.get_assets_for_video, video.video_id)

    video_url = None
    thumb_url = video.thumbnail_url
    for a in assets:
        if a.asset_type == "VIDEO" and a.storage_path:
            video_url = f"/media/{a.storage_path}"
        elif a.asset_type == "THUMBNAIL" and a.storage_path and not thumb_url:
            thumb_url = f"/media/{a.storage_path}"

    if not video_url:
        local_v = Path("data/raw/youtube") / f"local_{video.video_id}" / "video.mp4"
        if local_v.exists():
            video_url = f"/media/local_{video.video_id}/video.mp4"
    if not thumb_url:
        local_t = Path("data/raw/youtube") / f"local_{video.video_id}" / "thumbnail.jpg"
        if local_t.exists():
            thumb_url = f"/media/local_{video.video_id}/thumbnail.jpg"

    is_valid_youtube = bool(
        video.youtube_video_id
        and not video.youtube_video_id.startswith("local_")
        and len(video.youtube_video_id) == 11
    )

    return {
        "video_id": video.video_id,
        "youtube_video_id": video.youtube_video_id,
        "title": video.title,
        "description": video.description,
        "channel_title": video.channel_title,
        "channel_id": video.channel_id,
        "thumbnail_url": thumb_url,
        "playable_url": video_url,
        "is_youtube_embed": is_valid_youtube,
        "youtube_embed_id": video.youtube_video_id if is_valid_youtube else None,
        "duration": video.duration,
        "view_count": video.view_count,
        "like_count": video.like_count,
        "comment_count": video.comment_count,
        "category_id": video.category_id,
        "tags": video.tags,
        "has_embedding": video.embedding is not None,
        "feature_count": len(features),
        "features": [f.feature_type for f in features],
        "asset_count": len(assets),
        "assets": [a.asset_type for a in assets],
    }


@app.post("/api/interactions")
async def record_user_interaction(
    payload: dict,
    db: Session = Depends(get_db),
):
    """Record a user interaction event to dynamically update user personalization profile."""
    user_id = payload.get("user_id")
    video_id = payload.get("video_id")
    interaction_type = payload.get("interaction_type", "view")
    if not user_id or not video_id:
        raise HTTPException(status_code=400, detail="user_id and video_id are required")

    from src.db.repository import InteractionRepository
    interaction_repo = InteractionRepository(db)
    record = await run_in_threadpool(
        interaction_repo.create,
        user_id=user_id,
        video_id=video_id,
        interaction_type=interaction_type,
    )
    return {
        "id": str(record.id),
        "user_id": record.user_id,
        "video_id": record.video_id,
        "interaction_type": record.interaction_type,
        "timestamp": record.timestamp.isoformat() if record.timestamp else None,
    }


@app.get("/api/interactions/user/{user_id:path}")
async def get_user_interactions(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Retrieve recent interaction history for a user."""
    from src.db.repository import InteractionRepository
    interaction_repo = InteractionRepository(db)
    history = await run_in_threadpool(interaction_repo.get_user_history, user_id=user_id, limit=limit)
    return {
        "user_id": user_id,
        "count": len(history),
        "interactions": [
            {
                "id": str(h.id),
                "video_id": h.video_id,
                "interaction_type": h.interaction_type,
                "timestamp": h.timestamp.isoformat() if h.timestamp else None,
            }
            for h in history
        ],
    }


@app.delete("/api/interactions/user/{user_id:path}")
async def clear_user_interactions(
    user_id: str,
    db: Session = Depends(get_db),
):
    """Clear all recorded interaction history for a user."""
    from src.db.repository import InteractionRepository
    interaction_repo = InteractionRepository(db)
    deleted_count = await run_in_threadpool(interaction_repo.clear_user_history, user_id=user_id)
    return {
        "user_id": user_id,
        "deleted_count": deleted_count,
        "message": f"Successfully cleared {deleted_count} interactions for user '{user_id}'",
    }


@app.delete("/index/{video_id}")
async def delete_video(video_id: str, db: Session = Depends(get_db)):
    """Delete a single video from the index."""
    repo = VideoRepository(db)
    success = await run_in_threadpool(repo.delete, video_id)
    if success:
        return {"deleted": True, "video_id": video_id}
    raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")


@app.delete("/index")
async def clear_index(db: Session = Depends(get_db)):
    """Clear all videos from the index using efficient single delete query."""
    repo = VideoRepository(db)
    deleted_count = await run_in_threadpool(repo.delete_all)
    return {"cleared": deleted_count, "message": f"Deleted {deleted_count} videos from index"}


# ============ YouTube Integration Endpoints ============

@app.get("/api/youtube/video/{video_id:path}", response_model=YouTubeVideoResponse)
async def get_youtube_video(video_id: str):
    """
    Fetch and return normalized YouTube metadata for a given video ID or YouTube URL.
    """
    if not settings.YOUTUBE_API_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="YouTube Data API integration is disabled in system configuration."
        )

    service = YouTubeMetadataService(settings=settings)
    try:
        metadata = await run_in_threadpool(service.get_video_metadata, video_id)
        return metadata.to_api_response()
    except YouTubeInvalidVideoIdError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except YouTubeNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except YouTubePrivateVideoError as e:
        raise HTTPException(status_code=403, detail=e.message)
    except YouTubeAuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="YouTube API authentication failed. Verify API key configuration."
        )
    except YouTubeQuotaExceededError as e:
        raise HTTPException(status_code=429, detail=e.message)
    except YouTubeConfigurationError as e:
        raise HTTPException(status_code=500, detail=e.message)
    except YouTubeAPIError as e:
        status_code = e.status_code if (e.status_code and 400 <= e.status_code < 600) else 502
        raise HTTPException(status_code=status_code, detail=e.message)
    except Exception as e:
        logger.error("Unexpected error in YouTube endpoint: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error processing YouTube video.")


# ============ YouTube Ingestion Endpoints (Phase 4) ============

@app.post("/api/youtube/ingest", response_model=YouTubeIngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_youtube_video(
    request: YouTubeIngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Initiate asynchronous ingestion and media acquisition for a validated YouTube video.
    Returns immediately with HTTP 202 Accepted and PENDING status.
    """
    from sqlalchemy.orm import sessionmaker
    from src.services.youtube_ingestion import YouTubeIngestionService

    try:
        bind = db.get_bind()
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=bind)
        service = YouTubeIngestionService(settings=settings, session_factory=session_factory)
        record = await run_in_threadpool(service.register_ingestion_request, request.video_id)

        # Offload download and processing to background worker
        background_tasks.add_task(service.ingest_video, request.video_id)

        return YouTubeIngestResponse(
            video_id=record["video_id"],
            youtube_video_id=record["youtube_video_id"],
            status=record["status"],
            attempt_count=record.get("attempt_count", 0),
            message="Ingestion accepted and queued for background media acquisition"
        )
    except YouTubeInvalidVideoIdError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.error("Error initiating YouTube ingestion: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to initiate ingestion: {str(e)}")


@app.get("/api/youtube/ingest/{video_id:path}", response_model=YouTubeIngestStatusResponse)
async def get_ingest_status(video_id: str, db: Session = Depends(get_db)):
    """
    Check lifecycle and validation status of a YouTube video ingestion job.
    Never exposes internal filesystem paths.
    """
    from sqlalchemy.orm import sessionmaker
    from src.services.youtube_ingestion import YouTubeIngestionService

    bind = db.get_bind()
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=bind)
    service = YouTubeIngestionService(settings=settings, session_factory=session_factory)
    status_data = await run_in_threadpool(service.get_ingestion_status, video_id)
    if not status_data:
        raise HTTPException(status_code=404, detail=f"Ingestion record for '{video_id}' not found")

    return YouTubeIngestStatusResponse(
        video_id=status_data.get("video_id"),
        youtube_video_id=status_data["youtube_video_id"],
        status=status_data["status"],
        attempt_count=status_data.get("attempt_count", 0),
        max_attempts=status_data.get("max_attempts", 3),
        error=status_data.get("error_message"),
        started_at=status_data.get("started_at"),
        completed_at=status_data.get("completed_at"),
        created_at=status_data.get("created_at"),
        updated_at=status_data.get("updated_at")
    )


@app.get("/api/youtube/ingest", response_model=IngestionListResponse)
async def list_ingestion_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="Max ingestion records to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    status_filter: Optional[str] = Query(default=None, description="Optional status filter"),
    db: Session = Depends(get_db),
):
    """List recent video ingestion records with status and frame count."""
    from src.db.models import VideoIngestion
    from sqlalchemy import select, func

    stmt = select(VideoIngestion)
    if status_filter:
        stmt = stmt.where(VideoIngestion.status == status_filter)
    total = await run_in_threadpool(lambda: db.scalar(select(func.count(VideoIngestion.id))))
    stmt = stmt.order_by(VideoIngestion.updated_at.desc()).offset(offset).limit(limit)
    rows = await run_in_threadpool(lambda: list(db.execute(stmt).scalars().all()))

    items = [
        IngestionSummaryItem(
            video_id=r.video_id,
            youtube_video_id=r.youtube_video_id,
            status=r.status,
            attempt_count=r.attempt_count,
            error_message=r.error_message,
            frames_count=r.frames_count or 0,
            pipeline_version=r.pipeline_version,
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
            updated_at=r.updated_at.isoformat() if r.updated_at else None,
        )
        for r in rows
    ]
    return IngestionListResponse(total=total or 0, limit=limit, offset=offset, items=items)


@app.post("/api/youtube/ingest/batch", response_model=BatchIngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def batch_ingest_videos(
    request: BatchIngestRequest,
    db: Session = Depends(get_db),
):
    """
    Submit a bounded batch of YouTube videos for background ingestion with concurrency bounds.
    """
    from sqlalchemy.orm import sessionmaker
    from src.services.youtube_ingestion import YouTubeIngestionService, AssetIngestionOrchestrator

    bind = db.get_bind()
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=bind)
    service = YouTubeIngestionService(settings=settings, session_factory=session_factory)
    orchestrator = AssetIngestionOrchestrator(
        ingestion_service=service,
        max_concurrency=request.max_concurrency or 2,
    )
    job_id = await run_in_threadpool(orchestrator.submit_batch, request.video_ids)
    return BatchIngestResponse(
        job_id=job_id,
        total_videos=min(len(request.video_ids), 50),
        status="RUNNING",
        message="Batch ingestion accepted with bounded concurrency pool"
    )


@app.get("/api/youtube/assets/{video_id:path}", response_model=VideoAssetsResponse)
async def get_video_assets(video_id: str, db: Session = Depends(get_db)):
    """
    Retrieve registered multimodal assets (video, audio, thumbnail, frame)
    aligned with a specific video record.
    """
    from src.db.repository import VideoAssetRepository

    repo = VideoAssetRepository(db)
    assets = await run_in_threadpool(repo.get_assets_for_video, video_id)
    if not assets:
        from src.db.models import VideoAsset
        from sqlalchemy import select
        stmt = select(VideoAsset).where(VideoAsset.youtube_video_id == video_id)
        assets = await run_in_threadpool(lambda: list(db.execute(stmt).scalars().all()))

    asset_items = [
        VideoAssetItem(
            id=str(a.id),
            video_id=a.video_id,
            youtube_video_id=a.youtube_video_id,
            asset_type=a.asset_type,
            storage_path=a.storage_path,
            mime_type=a.mime_type,
            file_size_bytes=a.file_size_bytes,
            checksum_sha256=a.checksum_sha256,
            status=a.status,
            attempt_count=a.attempt_count,
            error_category=a.error_category,
            pipeline_version=a.pipeline_version,
            extra_metadata=a.extra_metadata,
            created_at=a.created_at.isoformat() if a.created_at else None,
            updated_at=a.updated_at.isoformat() if a.updated_at else None,
        )
        for a in assets
    ]
    return VideoAssetsResponse(
        video_id=video_id,
        total_assets=len(asset_items),
        assets=asset_items
    )


# ============ Phase 9 Vector Recommendation Endpoints ============

@app.get("/api/recommendations/health", response_model=RecommendationHealthResponse)
async def recommendation_health(
    db: Session = Depends(get_db),
    rec_service = Depends(get_recommendation_service),
):
    """Report retrieval subsystem readiness and Phase 8C checkpoint status."""
    return await run_in_threadpool(rec_service.get_health_status, session=db)


@app.get("/api/recommendations/video/{video_id:path}", response_model=RecommendationResponse)
async def get_video_recommendations(
    video_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Max candidate recommendations to return"),
    category_id: Optional[str] = Query(default=None, description="Optional category filter"),
    channel_id: Optional[str] = Query(default=None, description="Optional channel filter"),
    ranking_mode: str = Query(default="baseline", pattern="^(baseline|rerank)$", description="Ranking mode ('baseline' or 'rerank')"),
    candidate_pool_size: int = Query(default=50, ge=10, le=200, description="Stage 1 candidate generation pool size for re-ranking"),
    db: Session = Depends(get_db),
    rec_service = Depends(get_recommendation_service),
):
    """Retrieve nearest video candidates for a known video using Phase 8C representations."""
    try:
        response = await run_in_threadpool(
            rec_service.get_recommendations_for_video,
            session=db,
            video_id=video_id,
            top_k=limit,
            category_id=category_id,
            channel_id=channel_id,
            ranking_mode=ranking_mode,
            candidate_pool_size=candidate_pool_size,
        )
        return response
    except ValueError as e:
        err_msg = str(e)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error("Recommendation retrieval error for video '%s': %s", video_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Recommendation retrieval failed")


@app.post("/api/recommendations/similar", response_model=RecommendationResponse)
async def get_similar_recommendations(
    request: SimilarVideosRequest,
    db: Session = Depends(get_db),
    rec_service = Depends(get_recommendation_service),
):
    """Retrieve similar video candidates with custom exclusions and metadata filters."""
    try:
        response = await run_in_threadpool(
            rec_service.get_recommendations_for_video,
            session=db,
            video_id=request.video_id,
            top_k=request.top_k,
            exclude_video_ids=request.exclude_video_ids,
            category_id=request.category_id,
            channel_id=request.channel_id,
            ranking_mode=request.ranking_mode or "baseline",
            candidate_pool_size=request.candidate_pool_size or 50,
        )
        return response
    except ValueError as e:
        err_msg = str(e)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error("Similarity retrieval error for video '%s': %s", request.video_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Similarity recommendation failed")


@app.get("/api/recommendations/user/{user_id:path}", response_model=UserRecommendationResponse)
async def get_user_personalized_recommendations(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Max candidate recommendations to return"),
    exclude_seen: bool = Query(default=True, description="Exclude previously interacted videos from recommendations"),
    as_of: Optional[datetime] = Query(default=None, description="Evaluation cutoff timestamp preventing future-data leakage"),
    seed_video_id: Optional[str] = Query(default=None, description="Optional current video for content/context blending"),
    category_id: Optional[str] = Query(default=None, description="Optional category filter"),
    channel_id: Optional[str] = Query(default=None, description="Optional channel filter"),
    ranking_mode: str = Query(default="baseline", pattern="^(baseline|rerank)$", description="Ranking mode ('baseline' or 'rerank')"),
    candidate_pool_size: int = Query(default=50, ge=10, le=200, description="Stage 1 candidate generation pool size for re-ranking"),
    db: Session = Depends(get_db),
    pers_service = Depends(get_personalization_service),
):
    """Retrieve personalized video recommendations for a user with recency weighting and cutoff protection."""
    try:
        response = await run_in_threadpool(
            pers_service.get_personalized_recommendations,
            session=db,
            user_id=user_id,
            top_k=limit,
            exclude_seen=exclude_seen,
            cutoff_timestamp=as_of,
            seed_video_id=seed_video_id,
            category_id=category_id,
            channel_id=channel_id,
            ranking_mode=ranking_mode,
            candidate_pool_size=candidate_pool_size,
        )
        return response
    except Exception as e:
        logger.error("Personalized recommendation retrieval error for user '%s': %s", user_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Personalized recommendation failed")


# ============ Phase 12C Visual Video-to-Video Retrieval Endpoint ============

@app.get("/api/recommendations/visual/{video_id:path}", response_model=VisualRetrievalResponse)
async def get_visual_video_recommendations(
    video_id: str,
    limit: int = Query(default=5, ge=1, le=50, description="Max visual candidates to return"),
    feature_type: str = Query(default="resnet50_mean_2048", description="Visual feature type in video_features table"),
    db: Session = Depends(get_db),
    rec_service = Depends(get_recommendation_service),
):
    """
    Phase 12C: Direct Visual Video-to-Video Retrieval within compatible visual feature space.
    Strictly separate from production text/personalization recommendation endpoints.
    """
    try:
        dim = 2048 if "2048" in feature_type else 512
        modality = "VISUAL_THUMBNAIL" if "thumbnail" in feature_type else "VISUAL_FRAMES"
        raw_candidates = await run_in_threadpool(
            rec_service.retrieve_visual_candidates,
            session=db,
            video_id=video_id,
            top_k=limit,
            modality=modality,
            feature_type=feature_type,
            expected_dim=dim,
        )
        items = [
            VisualRetrievalItem(
                video_id=c["video_id"],
                score=c["score"],
                rank=c["rank"],
                feature_type=c["feature_type"],
                dimension=c["dimension"],
            )
            for c in raw_candidates
        ]
        return VisualRetrievalResponse(
            query_video_id=video_id,
            feature_type=feature_type,
            dimension=dim,
            count=len(items),
            candidates=items,
        )
    except ValueError as e:
        err_msg = str(e)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error("Visual retrieval error for video '%s': %s", video_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Visual candidate retrieval failed")


@app.get("/api/features/catalog")
async def get_multimodal_catalog(
    feature_type: Optional[str] = Query(default=None, description="Optional feature type filter"),
    db: Session = Depends(get_db)
):
    """Retrieve all videos in catalog that possess extracted multimodal features or media assets."""
    from src.db.models import VideoFeature, VideoAsset, Video
    from sqlalchemy import select

    stmt = select(VideoFeature).where(
        VideoFeature.is_valid.is_(True),
        VideoFeature.is_stale.is_(False)
    )
    if feature_type:
        stmt = stmt.where(VideoFeature.feature_type == feature_type)

    features = db.execute(stmt).scalars().all()
    vids_with_features = {}
    for f in features:
        if f.video_id not in vids_with_features:
            vids_with_features[f.video_id] = []
        if f.feature_type not in vids_with_features[f.video_id]:
            vids_with_features[f.video_id].append(f.feature_type)

    all_vids = list(vids_with_features.keys())
    if not all_vids and not feature_type:
        asset_vids = set(db.execute(select(VideoAsset.video_id).distinct()).scalars().all())
        all_vids = list(asset_vids)

    if not all_vids:
        return {"count": 0, "videos": []}

    video_rows = db.execute(select(Video).where(Video.video_id.in_(all_vids))).scalars().all()
    videos = [
        {
            "video_id": v.video_id,
            "title": v.title or f"Video {v.video_id}",
            "channel_title": v.channel_title or ("MicroLens Partner" if v.youtube_video_id else "Catalog Item"),
            "category_id": v.category_id or "Educational",
            "has_embedding": v.embedding is not None,
            "feature_types": vids_with_features.get(v.video_id, []),
        }
        for v in video_rows
    ]
    return {"count": len(videos), "videos": videos}


# ============ Feature: Try Your Own Video Endpoint ============

MAX_CUSTOM_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB limit
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@app.post("/api/custom-video/analyze", response_model=CustomVideoAnalysisResponse)
async def analyze_custom_video(
    file: UploadFile = File(..., description="Uploaded video file (.mp4, .avi, .mov, .mkv, .webm)"),
    title: Optional[str] = Form(default=None, description="Optional title for text encoding and recommendation"),
    description: Optional[str] = Form(default=None, description="Optional description for text context"),
    max_recommendations: int = Form(default=6, ge=1, le=20, description="Max candidate recommendations to return"),
    db: Session = Depends(get_db),
    rec_service = Depends(get_recommendation_service),
):
    """
    User-Facing Custom Video Upload and Analysis Workflow.
    1. Validates container format, decodability, duration, and file size.
    2. Ephemerally samples video frames and validates audio availability.
    3. Extracts ResNet-50 2048-D visual embedding and retrieves visually similar catalog videos.
    4. Encodes title/description with Phase 8C Contrastive BERT (512-D) and retrieves text recommendations.
    5. Returns truthful modality statuses and cleans up all temporary storage.
    """
    import base64
    import tempfile
    import uuid
    import cv2
    import numpy as np

    from src.integrations.youtube.media.validator import (
        MediaValidator,
        MediaValidationError,
        MediaSizeExceededError,
        MediaDurationExceededError,
        MediaCorruptError,
    )
    from src.features.frame_sampler import FrameSampler
    from src.features.visual_processor import VisualProcessor
    from src.features.visual_encoder import ResNet50VisualEncoder
    from src.integrations.youtube.media.audio import AudioExtractor, AudioExtractionError

    # 1. Validate file extension
    raw_filename = file.filename or "video.mp4"
    safe_ext = Path(raw_filename).suffix.lower()
    if safe_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{safe_ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Ephemeral server-controlled storage with size limit
    upload_uuid = uuid.uuid4().hex[:12]
    tmp_dir_obj = tempfile.TemporaryDirectory(prefix=f"custom_upload_{upload_uuid}_")
    tmp_dir = Path(tmp_dir_obj.name)
    tmp_video_path = tmp_dir / f"input_{upload_uuid}{safe_ext}"

    try:
        bytes_read = 0
        with open(tmp_video_path, "wb") as f_out:
            while chunk := await file.read(64 * 1024):
                bytes_read += len(chunk)
                if bytes_read > MAX_CUSTOM_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds maximum limit of 100 MB ({bytes_read / (1024 * 1024):.2f} MB)",
                    )
                f_out.write(chunk)

        if bytes_read == 0:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty (0 bytes).")

        # 3. Media validation (container integrity, decodability, duration)
        validator = MediaValidator(max_video_size_mb=100, max_duration_seconds=600)
        try:
            video_meta = validator.validate_video_file(tmp_video_path)
        except MediaCorruptError as e:
            raise HTTPException(status_code=400, detail=f"Corrupt or unreadable video file: {e}")
        except MediaDurationExceededError as e:
            raise HTTPException(status_code=400, detail=f"Video duration exceeds limit: {e}")
        except MediaSizeExceededError as e:
            raise HTTPException(status_code=413, detail=str(e))
        except MediaValidationError as e:
            raise HTTPException(status_code=400, detail=f"Media validation failed: {e}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to validate video container: {e}")

        # 4. Frame sampling and lightweight base64 preview generation
        frame_sampler = FrameSampler(default_num_frames=8)
        sample_res = frame_sampler.sample_frames(tmp_video_path, num_frames=8)
        if not sample_res.success or not sample_res.frames:
            raise HTTPException(
                status_code=400,
                detail=f"Frame decode failure: {sample_res.error_message or 'No frames could be decoded'}",
            )

        extracted_frames_b64: list[str] = []
        preview_indices = [
            0,
            len(sample_res.frames) // 3,
            (2 * len(sample_res.frames)) // 3,
            len(sample_res.frames) - 1,
        ]
        seen_indices = set()
        for p_idx in preview_indices:
            if p_idx < len(sample_res.frames) and p_idx not in seen_indices:
                seen_indices.add(p_idx)
                rgb_f = sample_res.frames[p_idx]
                bgr_f = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2BGR)
                thumb = cv2.resize(bgr_f, (320, 180), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                extracted_frames_b64.append(f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}")

        # 5. Visual Feature Extraction (ResNet-50 2048-D) and Visual Recommendations
        visual_status = "EXTRACTED"
        visual_recs: list[CustomVideoRecommendationItem] = []
        try:
            visual_proc = VisualProcessor(target_size=(224, 224), normalize=True)
            frames_tensor = visual_proc.preprocess_frames(sample_res.frames).to(state.device)

            if not hasattr(state, "resnet_encoder") or state.resnet_encoder is None:
                state.resnet_encoder = ResNet50VisualEncoder(
                    feature_dim=512,
                    pretrained=True,
                    freeze_backbone=True,
                    aggregation="mean",
                ).to(state.device)
                state.resnet_encoder.eval()

            with torch.no_grad():
                raw_feats = state.resnet_encoder.extract_backbone_features(frames_tensor)
                pooled_2048 = raw_feats.mean(dim=0)
                norm_val = torch.norm(pooled_2048, p=2)
                if norm_val > 1e-7:
                    q_unit = (pooled_2048 / norm_val).cpu().numpy().astype(np.float32)
                else:
                    q_unit = None

            if q_unit is not None:
                from src.db.models import VideoFeature, Video
                cand_features = db.query(VideoFeature).filter(
                    VideoFeature.feature_type == "resnet50_mean_2048",
                    VideoFeature.is_valid.is_(True),
                    VideoFeature.is_stale.is_(False),
                ).all()

                scored_candidates = []
                for cf in cand_features:
                    if not cf.embedding:
                        continue
                    c_arr = np.asarray(cf.embedding, dtype=np.float32)
                    c_norm = np.linalg.norm(c_arr)
                    if c_norm > 1e-7:
                        sim = float(np.dot(q_unit, c_arr / c_norm))
                        sim_bounded = max(-1.0, min(1.0, sim))
                        scored_candidates.append((cf.video_id, sim_bounded))

                scored_candidates.sort(key=lambda item: (-item[1], str(item[0])))
                top_visual = scored_candidates[:max_recommendations]

                if top_visual:
                    vis_vids = [v[0] for v in top_visual]
                    video_rows = {v.video_id: v for v in db.query(Video).filter(Video.video_id.in_(vis_vids)).all()}
                    for rank, (vid, score) in enumerate(top_visual, start=1):
                        v_obj = video_rows.get(vid)
                        visual_recs.append(
                            CustomVideoRecommendationItem(
                                video_id=vid,
                                title=v_obj.title if v_obj else f"Video {vid}",
                                score=round(score, 4),
                                rank=rank,
                                thumbnail_url=v_obj.thumbnail_url if v_obj else None,
                                duration=v_obj.duration if v_obj else None,
                                category_id=v_obj.category_id if v_obj else None,
                                channel_title=v_obj.channel_title if v_obj else None,
                            )
                        )
        except Exception as e_vis:
            logger.warning("Visual feature extraction error: %s", e_vis, exc_info=True)
            visual_status = f"FAILED: {e_vis}"

        # 6. Audio Stream Analysis
        audio_status = "NO_AUDIO_STREAM"
        audio_details = "Video container has no audio stream"
        try:
            audio_extractor = AudioExtractor(sample_rate=16000, duration=5.0)
            probe = audio_extractor.probe_audio_stream(tmp_video_path)
            if probe.get("has_audio"):
                wav_tmp_path = tmp_dir / "extracted_audio.wav"
                try:
                    audio_extractor.extract_and_save_wav(tmp_video_path, wav_tmp_path, check_silence=True)
                    if wav_tmp_path.exists() and wav_tmp_path.stat().st_size > 0:
                        audio_status = "EXTRACTED_NON_SILENT"
                        audio_details = "Valid 16 kHz mono audio track detected & decoded"
                    else:
                        audio_status = "AUDIO_EXTRACTION_EMPTY"
                        audio_details = "Extracted audio file was empty"
                except AudioExtractionError as aee:
                    if "SILENT_AUDIO" in str(aee):
                        audio_status = "SILENT_AUDIO_DETECTED"
                        audio_details = "Audio track is present but signal is completely silent"
                    elif "NO_AUDIO_STREAM" in str(aee):
                        audio_status = "NO_AUDIO_STREAM"
                        audio_details = "Video container has no audio stream"
                    else:
                        audio_status = "EXTRACTION_FAILED"
                        audio_details = f"Audio extraction error: {aee}"
        except Exception as e_aud:
            logger.warning("Audio probing failed: %s", e_aud)
            audio_status = "UNAVAILABLE"
            audio_details = f"Audio probing error: {e_aud}"

        # 7. Text Analysis & Text Recommendations
        text_input = f"{title or ''} {description or ''}".strip()
        text_status = "NO_TEXT_PROVIDED"
        text_details = "Provide a title or description to query the Phase 8C text co-engagement recommendation index."
        text_recs: list[CustomVideoRecommendationItem] = []

        if text_input and rec_service and rec_service.is_checkpoint_loaded:
            try:
                text_emb = rec_service.encode_text(text_input)
                results, backend, latency = rec_service.vector_index_service.search_database(
                    session=db,
                    query_embedding=text_emb,
                    top_k=max_recommendations,
                )
                for rank, (cand_video, score) in enumerate(results, start=1):
                    text_recs.append(
                        CustomVideoRecommendationItem(
                            video_id=cand_video.video_id,
                            title=cand_video.title or "",
                            score=round(float(score), 4),
                            rank=rank,
                            thumbnail_url=cand_video.thumbnail_url,
                            duration=cand_video.duration,
                            category_id=cand_video.category_id,
                            channel_title=cand_video.channel_title,
                        )
                    )
                text_status = "EMBEDDING_GENERATED"
                text_details = f"Phase 8C Contrastive BERT (512-D) query retrieved via {backend} ({latency:.1f}ms)"
            except Exception as e_txt:
                logger.warning("Text recommendation error: %s", e_txt)
                text_status = f"FAILED: {e_txt}"
                text_details = str(e_txt)

        return CustomVideoAnalysisResponse(
            status="success",
            video_metadata={
                "filename": raw_filename,
                "size_mb": video_meta.get("size_mb", 0.0),
                "duration_seconds": video_meta.get("duration_seconds", 0.0),
                "fps": video_meta.get("fps", 0.0),
                "frame_count": video_meta.get("frame_count", 0),
                "width": video_meta.get("width", 0),
                "height": video_meta.get("height", 0),
            },
            extracted_frames=extracted_frames_b64,
            modalities={
                "visual": {
                    "status": visual_status,
                    "model": "torchvision_resnet50",
                    "feature_type": "resnet50_mean_2048",
                    "dimension": 2048,
                    "frames_sampled": len(sample_res.frames),
                    "details": "8-frame mean-pooled ResNet-50 2048-D normalized embedding",
                },
                "audio": {
                    "status": audio_status,
                    "details": audio_details,
                },
                "text": {
                    "status": text_status,
                    "model": "Phase 8C Contrastive BERT",
                    "dimension": 512 if text_status == "EMBEDDING_GENERATED" else 0,
                    "input_text": text_input if text_input else None,
                    "details": text_details,
                },
            },
            visual_recommendations=visual_recs,
            text_recommendations=text_recs,
            audio_status_note=(
                "Audio extraction status validated. Ranking weights for audio are 0.00 in production to maintain catalog consistency."
            ),
        )

    finally:
        # Deterministic cleanup of all ephemeral files
        try:
            tmp_dir_obj.cleanup()
        except Exception as e_clean:
            logger.warning("Error cleaning temporary upload directory: %s", e_clean)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
