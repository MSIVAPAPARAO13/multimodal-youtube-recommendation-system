"""
SQLAlchemy Database Models

These define the structure of our database tables.
SQLAlchemy automatically creates tables and handles SQL generation.
"""

import uuid
from datetime import datetime
from typing import Optional

from enum import Enum
from sqlalchemy import Column, String, Text, DateTime, Integer, BigInteger, Float, Index, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


class IngestionStatus(str, Enum):
    """Lifecycle states for video ingestion pipeline."""
    PENDING = "PENDING"
    METADATA_READY = "METADATA_READY"
    MEDIA_DOWNLOADING = "MEDIA_DOWNLOADING"
    MEDIA_READY = "MEDIA_READY"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AssetType(str, Enum):
    """Types of multimodal video assets tracked by the ingestion system."""
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    THUMBNAIL = "THUMBNAIL"
    FRAME = "FRAME"
    METADATA = "METADATA"


class AssetStatus(str, Enum):
    """Integrity and processing states for aligned media assets."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    CORRUPTED = "CORRUPTED"


class FeatureModality(str, Enum):
    """Supported modalities for extracted video representations."""
    VISUAL_FRAMES = "VISUAL_FRAMES"
    VISUAL_THUMBNAIL = "VISUAL_THUMBNAIL"
    AUDIO = "AUDIO"
    CLIP_THUMBNAIL = "CLIP_THUMBNAIL"

# Try to import pgvector
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    print("pgvector not installed. Using ARRAY for embeddings.")


class CompatibleUUID(TypeDecorator):
    """Native UUID on PostgreSQL, String(36) on SQLite/others."""
    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class CompatibleArray(TypeDecorator):
    """Native ARRAY on PostgreSQL, JSON on SQLite/others."""
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(ARRAY(String))
        return dialect.type_descriptor(JSON)


class CompatibleEmbedding(TypeDecorator):
    """Vector(512) on PostgreSQL with pgvector, JSON on SQLite/others."""
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            if HAS_PGVECTOR:
                return dialect.type_descriptor(Vector(512))
            return dialect.type_descriptor(ARRAY(Float))
        return dialect.type_descriptor(JSON)


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Video(Base):
    """
    Video metadata and embeddings.

    This table stores all information about indexed videos,
    including their computed embeddings for similarity search.
    """

    __tablename__ = "videos"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        CompatibleUUID(),
        primary_key=True,
        default=uuid.uuid4
    )

    # Video identifier (from external source)
    video_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    # Optional YouTube external video ID
    youtube_video_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True
    )

    # Core Metadata
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Extended YouTube Metadata (Phase 4)
    channel_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    channel_title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    category_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    view_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    like_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    comment_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    privacy_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Ingestion State Tracking on Video Entity
    ingestion_status: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        default=IngestionStatus.PENDING.value
    )
    ingestion_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_ingestion_attempt: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Tags stored as array or JSON
    tags: Mapped[Optional[list]] = mapped_column(CompatibleArray(), nullable=True)

    # Embedding (512-dimensional vector)
    embedding = Column(CompatibleEmbedding(), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    def __repr__(self):
        return f"<Video(video_id='{self.video_id}', title='{self.title[:30]}...')>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "video_id": self.video_id,
            "youtube_video_id": self.youtube_video_id,
            "title": self.title,
            "description": self.description,
            "video_path": self.video_path,
            "channel_id": self.channel_id,
            "channel_title": self.channel_title,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "category_id": self.category_id,
            "thumbnail_url": self.thumbnail_url,
            "duration": self.duration,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "privacy_status": self.privacy_status,
            "ingestion_status": self.ingestion_status,
            "ingestion_error": self.ingestion_error,
            "last_ingestion_attempt": self.last_ingestion_attempt.isoformat() if self.last_ingestion_attempt else None,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class VideoIngestion(Base):
    """
    Dedicated model for tracking asynchronous ingestion lifecycle, media files,
    attempt counts, and errors for YouTube videos.
    """
    __tablename__ = "video_ingestions"

    id: Mapped[uuid.UUID] = mapped_column(
        CompatibleUUID(),
        primary_key=True,
        default=uuid.uuid4
    )
    video_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True
    )
    youtube_video_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default=IngestionStatus.PENDING.value,
        nullable=False,
        index=True
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    video_path: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True
    )
    audio_path: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True
    )
    thumbnail_path: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True
    )
    transcript_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    transcript_language: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True
    )
    transcript_available: Mapped[bool] = mapped_column(
        default=False,
        nullable=False
    )
    transcript_provider: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    frames_path: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True
    )
    frames_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    pipeline_version: Mapped[str] = mapped_column(
        String(32),
        default="12A",
        nullable=False
    )

    def to_dict(self) -> dict:
        """Convert to dict without exposing internal filesystem paths."""
        return {
            "id": str(self.id),
            "video_id": self.video_id,
            "youtube_video_id": self.youtube_video_id,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "error_message": self.error_message,
            "transcript_language": self.transcript_language,
            "transcript_available": self.transcript_available,
            "transcript_provider": self.transcript_provider,
            "frames_count": self.frames_count,
            "pipeline_version": self.pipeline_version,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class VideoAsset(Base):
    """
    Tracks granular multimodal assets (video, audio, thumbnail, frame, metadata)
    associated with catalog and YouTube videos.
    Records storage location, checksums, byte size, MIME type, status, and metadata.
    """
    __tablename__ = "video_assets"

    id: Mapped[uuid.UUID] = mapped_column(
        CompatibleUUID(),
        primary_key=True,
        default=uuid.uuid4
    )
    # References internal video_id in videos table (e.g. '2' or 'yt_dQw4w9WgXcQ')
    video_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True
    )
    # External YouTube video ID if known
    youtube_video_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True
    )
    # Asset type: VIDEO, AUDIO, THUMBNAIL, FRAME, METADATA
    asset_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True
    )
    # Source URL or origin reference
    source_url: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True
    )
    # Relative storage path inside project media storage root
    storage_path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False
    )
    mime_type: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True
    )
    file_size_bytes: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        nullable=True
    )
    checksum_sha256: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default=AssetStatus.READY.value,
        nullable=False,
        index=True
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False
    )
    error_category: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    pipeline_version: Mapped[str] = mapped_column(
        String(32),
        default="12A",
        nullable=False
    )
    # Structured extra metadata (e.g. sample_rate, duration, frame_idx, timestamps, dimensions)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    __table_args__ = (
        Index("ix_video_asset_lookup", "video_id", "asset_type"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "video_id": self.video_id,
            "youtube_video_id": self.youtube_video_id,
            "asset_type": self.asset_type,
            "storage_path": self.storage_path,
            "mime_type": self.mime_type,
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "pipeline_version": self.pipeline_version,
            "extra_metadata": self.extra_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class VideoFeature(Base):
    """
    Normalized multimodal feature representation table.
    Stores extracted vector representations, dimensions, model identity,
    source checksum provenance, and validation status.
    """
    __tablename__ = "video_features"

    id: Mapped[uuid.UUID] = mapped_column(
        CompatibleUUID(),
        primary_key=True,
        default=uuid.uuid4
    )
    video_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True
    )
    youtube_video_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True
    )
    asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        CompatibleUUID(),
        nullable=True
    )
    modality: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True
    )
    feature_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True
    )
    encoder_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False
    )
    encoder_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False
    )
    checkpoint_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True
    )
    dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )
    preprocessing_config: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True
    )
    source_checksum: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True
    )
    embedding: Mapped[list] = mapped_column(
        JSON,
        nullable=False
    )
    is_valid: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    is_stale: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    is_trained: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    validation_status: Mapped[str] = mapped_column(
        String(32),
        default="VALID",
        nullable=False
    )
    pipeline_version: Mapped[str] = mapped_column(
        String(32),
        default="12B",
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    __table_args__ = (
        Index("ix_video_feature_lookup", "video_id", "modality", "feature_type", "encoder_name", unique=True),
    )

    def to_dict(self, include_embedding: bool = False) -> dict:
        data = {
            "id": str(self.id),
            "video_id": self.video_id,
            "youtube_video_id": self.youtube_video_id,
            "asset_id": str(self.asset_id) if self.asset_id else None,
            "modality": self.modality,
            "feature_type": self.feature_type,
            "encoder_name": self.encoder_name,
            "encoder_version": self.encoder_version,
            "checkpoint_id": self.checkpoint_id,
            "dimension": self.dimension,
            "preprocessing_config": self.preprocessing_config,
            "source_checksum": self.source_checksum,
            "is_valid": self.is_valid,
            "is_stale": self.is_stale,
            "is_trained": self.is_trained,
            "validation_status": self.validation_status,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_embedding:
            data["embedding"] = self.embedding
        return data


class UserInteraction(Base):
    """
    User interactions with videos (for future user-based recommendations).

    Stores events like: views, likes, shares, watch time.
    """

    __tablename__ = "user_interactions"

    id: Mapped[uuid.UUID] = mapped_column(
        CompatibleUUID(),
        primary_key=True,
        default=uuid.uuid4
    )

    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    video_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Interaction type: 'view', 'like', 'share', 'complete'
    interaction_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Additional data (e.g., watch_time in seconds)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # When the interaction happened
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    def __repr__(self):
        return f"<Interaction(user='{self.user_id}', video='{self.video_id}', type='{self.interaction_type}')>"


# Create index for fast similarity search with pgvector
if HAS_PGVECTOR:
    # HNSW index for approximate nearest neighbor search
    # Supports incremental additions without requiring pre-computed centroids
    video_embedding_index = Index(
        'ix_video_embedding',
        Video.embedding,
        postgresql_using='hnsw',
        postgresql_ops={'embedding': 'vector_cosine_ops'}
    )
