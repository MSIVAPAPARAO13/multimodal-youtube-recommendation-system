"""
Pydantic Schemas for API request/response validation.

Pydantic automatically validates incoming JSON and provides clear error messages.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


# ============ Request Schemas ============

class VideoInput(BaseModel):
    """Input for encoding a single video."""
    video_id: str = Field(..., description="Unique identifier for the video")
    video_path: Optional[str] = Field(None, description="Path to video file")
    title: str = Field(..., description="Video title")
    description: Optional[str] = Field("", description="Video description")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "abc123",
                "video_path": "data/raw/videos/abc123.mp4",
                "title": "Easy pasta recipe",
                "description": "Easy 10-minute recipe"
            }
        }
    )


class RecommendRequest(BaseModel):
    """Request for video recommendations."""
    video_id: str = Field(..., description="ID of video to find recommendations for")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "abc123",
                "top_k": 10
            }
        }
    )


class SimilarityRequest(BaseModel):
    """Request to compute similarity between two videos."""
    video_id_1: str
    video_id_2: str


class IndexMicroLensRequest(BaseModel):
    """Request to batch index videos from local MicroLens dataset."""
    max_videos: int = Field(default=10, ge=1, le=1000, description="Max videos to process")


class IndexMicroLensResponse(BaseModel):
    """Response after batch indexing MicroLens videos."""
    indexed: int
    skipped: int
    total_in_index: int
    message: str


# ============ Response Schemas ============

class VideoEmbedding(BaseModel):
    """Response containing video embedding."""
    video_id: str
    embedding: list[float]
    dimension: int


class Recommendation(BaseModel):
    """A single recommendation item."""
    video_id: str
    score: float = Field(..., description="Similarity score (0-1)")
    title: Optional[str] = None


class RecommendResponse(BaseModel):
    """Response with list of recommendations."""
    query_video_id: str
    query_title: Optional[str] = ""
    recommendations: list[Recommendation]
    storage: Optional[str] = "postgresql"


class SimilarityResponse(BaseModel):
    """Response with similarity score."""
    video_id_1: str
    video_id_2: str
    similarity: float = Field(..., description="Cosine similarity (-1 to 1)")


class HealthResponse(BaseModel):
    """Health check response with explicit model and checkpoint provenance."""
    status: str
    model_loaded: bool
    model_trained: bool = False
    recommendation_checkpoint_loaded: bool = False
    multimodal_checkpoint_loaded: bool = False
    checkpoint_provenance: Optional[str] = None
    environment: str = "development"
    index_size: int = Field(..., description="Number of videos in index")
    device: str = "cpu"
    storage: str = "postgresql"


class IndexResponse(BaseModel):
    """Response after indexing a single video."""
    video_id: str
    indexed: bool
    index_size: int = 0
    message: Optional[str] = None


# ============ YouTube Schemas Re-Export ============
from src.integrations.youtube.schemas import (
    YouTubeVideoResponse,
    YouTubeThumbnails,
    YouTubeStatistics,
    YouTubeVideoMetadata,
)


# ============ YouTube Ingestion Schemas (Phase 4) ============

class YouTubeIngestRequest(BaseModel):
    """Request payload to initiate YouTube video ingestion."""
    video_id: str = Field(..., description="YouTube video ID or full YouTube URL")


class YouTubeIngestResponse(BaseModel):
    """Immediate response after queuing video ingestion."""
    video_id: str
    youtube_video_id: str
    status: str
    attempt_count: int = 0
    message: Optional[str] = "Ingestion initiated asynchronously"


class YouTubeIngestStatusResponse(BaseModel):
    """Status tracking response for video ingestion."""
    video_id: Optional[str] = None
    youtube_video_id: str
    status: str
    attempt_count: int = 0
    max_attempts: int = 3
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ============ Phase 9 Vector Recommendation Schemas ============
from src.schemas.recommendations import (
    RecommendationItem,
    SimilarVideosRequest,
    RecommendationResponse,
    RecommendationHealthResponse,
    UserRecommendationResponse,
)


# ============ Multimodal Asset Schemas (Phase 12A) ============

class VideoAssetItem(BaseModel):
    """Schema for aligned multimodal video asset."""
    id: str
    video_id: str
    youtube_video_id: Optional[str] = None
    asset_type: str
    storage_path: str
    mime_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    checksum_sha256: Optional[str] = None
    status: str
    attempt_count: int = 1
    error_category: Optional[str] = None
    pipeline_version: str = "12A"
    extra_metadata: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class VideoAssetsResponse(BaseModel):
    """Response containing all registered multimodal assets for a video."""
    video_id: str
    total_assets: int
    assets: list[VideoAssetItem]


class BatchIngestRequest(BaseModel):
    """Request payload to initiate bounded batch ingestion."""
    video_ids: list[str] = Field(..., max_length=50, description="List of YouTube video IDs or URLs")
    max_concurrency: Optional[int] = Field(2, ge=1, le=8, description="Worker pool concurrency limit")


class BatchIngestResponse(BaseModel):
    """Response returned upon submitting a batch ingestion job."""
    job_id: str
    total_videos: int
    status: str
    message: str


class IngestionSummaryItem(BaseModel):
    """Summary item for listed ingestion jobs."""
    video_id: Optional[str] = None
    youtube_video_id: str
    status: str
    attempt_count: int
    error_message: Optional[str] = None
    frames_count: int = 0
    pipeline_version: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None


class IngestionListResponse(BaseModel):
    """Paginated list of video ingestion records."""
    total: int
    limit: int
    offset: int
    items: list[IngestionSummaryItem]


# ============ Phase 12C Visual Retrieval Schemas ============

class VisualRetrievalItem(BaseModel):
    """Candidate video retrieved via direct visual feature similarity."""
    video_id: str
    score: float
    rank: int
    feature_type: str
    dimension: int


class VisualRetrievalResponse(BaseModel):
    """Response payload for visual video-to-video candidate retrieval."""
    query_video_id: str
    feature_type: str
    dimension: int
    count: int
    candidates: list[VisualRetrievalItem]


# ============ Custom Video Analysis & Recommendation Schemas ============

class CustomVideoRecommendationItem(BaseModel):
    """A recommended candidate video from catalog matching a custom upload."""
    video_id: str
    title: str
    score: float
    rank: int
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None
    category_id: Optional[str] = None
    channel_title: Optional[str] = None


class CustomVideoAnalysisResponse(BaseModel):
    """Full analysis, modality status, and recommendations for a custom video upload."""
    status: str
    video_metadata: dict
    extracted_frames: list[str]
    modalities: dict
    visual_recommendations: list[CustomVideoRecommendationItem]
    text_recommendations: list[CustomVideoRecommendationItem]
    audio_status_note: str




