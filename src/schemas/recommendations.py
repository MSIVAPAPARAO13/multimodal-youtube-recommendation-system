"""
Pydantic Schemas for Vector Recommendation Retrieval.

Experiment / Phase 9 Mode: TEXT_COENGAGEMENT_V2
Exposes clean, validated request and response payloads for recommendation endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class RecommendationItem(BaseModel):
    """A single retrieved and hydrated candidate recommendation."""
    video_id: str = Field(..., description="Unique video identifier")
    title: str = Field(..., description="Video title")
    score: float = Field(..., description="Cosine similarity score in range [-1.0, 1.0]")
    rank: int = Field(..., ge=1, description="1-indexed recommendation position")
    category_id: Optional[str] = Field(None, description="YouTube / dataset category ID")
    channel_title: Optional[str] = Field(None, description="Channel title if available")
    thumbnail_url: Optional[str] = Field(None, description="Verified thumbnail or poster URL")
    duration: Optional[int] = Field(None, description="Verified duration in seconds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "vid_1024",
                "title": "Machine Learning Crash Course",
                "score": 0.8924,
                "rank": 1,
                "category_id": "27",
                "channel_title": "TechTutorials"
            }
        }
    )


class SimilarVideosRequest(BaseModel):
    """Payload to request candidates similar to a query video."""
    video_id: str = Field(..., min_length=1, max_length=255, description="Query video identifier")
    top_k: int = Field(10, ge=1, le=50, description="Maximum number of candidates to retrieve")
    exclude_video_ids: Optional[List[str]] = Field(None, description="List of already-seen or excluded video IDs")
    category_id: Optional[str] = Field(None, description="Filter candidates by category ID")
    channel_id: Optional[str] = Field(None, description="Filter candidates by channel ID")
    ranking_mode: Optional[str] = Field("baseline", description="Candidate ranking mode ('baseline' or 'rerank')")
    candidate_pool_size: Optional[int] = Field(50, ge=10, le=200, description="Stage 1 candidate generation pool size for re-ranking")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "vid_1024",
                "top_k": 10,
                "exclude_video_ids": ["vid_1024", "vid_501"],
                "category_id": "27",
                "ranking_mode": "baseline",
                "candidate_pool_size": 50,
            }
        }
    )


class RecommendationResponse(BaseModel):
    """Top-K recommendation result for a query video."""
    query_video_id: str = Field(..., description="Query video identifier")
    query_title: str = Field(..., description="Query video title")
    recommendations: List[RecommendationItem] = Field(default_factory=list, description="Ranked list of candidates")
    count: int = Field(..., description="Number of candidates returned")
    retrieval_mode: str = Field("TEXT_COENGAGEMENT_V2", description="Retrieval representation mode")
    retrieval_backend: str = Field(..., description="Retrieval backend utilized ('pgvector' or 'exact_fallback')")
    model_version: str = Field("EXP_MICROLENS_TEXT_COENGAGEMENT_V2", description="Trained checkpoint experiment ID")
    ranking_mode: Optional[str] = Field("baseline", description="Active ranking mode ('baseline' or 'rerank')")
    reranking_applied: Optional[bool] = Field(False, description="Whether Phase 11B candidate re-ranking was applied")


class RecommendationHealthResponse(BaseModel):
    """Health and capability diagnostic for recommendation engine."""
    status: str = Field("ok", description="Subsystem health status")
    retrieval_mode: str = Field("TEXT_COENGAGEMENT_V2", description="Active retrieval representation mode")
    checkpoint_loaded: bool = Field(..., description="Whether Phase 8C checkpoint is loaded")
    checkpoint_sha256: Optional[str] = Field(None, description="SHA256 fingerprint of loaded checkpoint")
    vector_backend: str = Field(..., description="Active vector search backend ('pgvector' or 'exact_fallback')")
    pgvector_available: bool = Field(..., description="Whether pgvector extension is available in environment")
    total_indexed_videos: int = Field(..., description="Total count of videos with embeddings in database")


class UserRecommendationResponse(BaseModel):
    """Personalized top-K recommendations for a specific user."""
    user_id: str = Field(..., description="Target user identifier")
    recommendations: List[RecommendationItem] = Field(default_factory=list, description="Ranked list of personalized candidates")
    count: int = Field(..., description="Number of candidates returned")
    personalization_applied: bool = Field(..., description="Whether user history personalization was applied")
    fallback_reason: Optional[str] = Field(None, description="Reason if cold-start fallback was triggered")
    history_length: int = Field(0, description="Number of valid historical events incorporated into profile")
    cutoff_timestamp: Optional[str] = Field(None, description="ISO timestamp cutoff used to prevent temporal leakage")
    retrieval_mode: str = Field("TEXT_COENGAGEMENT_V2", description="Active item embedding representation mode")
    retrieval_backend: str = Field(..., description="Vector retrieval backend utilized ('pgvector' or 'exact_fallback')")
    model_version: str = Field("EXP_MICROLENS_USER_PERSONALIZATION_V1", description="Personalization model version")
    ranking_mode: Optional[str] = Field("baseline", description="Active ranking mode ('baseline' or 'rerank')")
    reranking_applied: Optional[bool] = Field(False, description="Whether Phase 11B candidate re-ranking was applied")

