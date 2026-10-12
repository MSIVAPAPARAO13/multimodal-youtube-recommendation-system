"""
Database Configuration

Handles connection settings and database URL construction.
Uses environment variables for sensitive data.
"""

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file
load_dotenv()


def resolve_ffmpeg_path() -> Optional[str]:
    """Resolve FFmpeg binary path from environment, system PATH, or imageio_ffmpeg."""
    env_path = os.getenv("FFMPEG_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())
    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def resolve_ffprobe_path() -> Optional[str]:
    """Resolve FFprobe binary path from environment or system PATH."""
    env_path = os.getenv("FFPROBE_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())
    which_path = shutil.which("ffprobe")
    if which_path:
        return which_path
    return None


class Settings(BaseModel):
    """Centralized typed application settings."""

    # Database
    POSTGRES_HOST: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    POSTGRES_PORT: int = Field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    POSTGRES_USER: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    POSTGRES_PASSWORD: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres"))
    POSTGRES_DB: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "video_recommender"))
    DB_POOL_SIZE: int = Field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "5")))
    DB_MAX_OVERFLOW: int = Field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "10")))

    # Model & Checkpoints
    MODEL_PATH: str = Field(default_factory=lambda: os.getenv("MODEL_PATH", "checkpoints/best.pt"))
    DEVICE: str = Field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    ALLOW_UNINITIALIZED_MODEL: bool = Field(
        default_factory=lambda: os.getenv("ALLOW_UNINITIALIZED_MODEL", "false").lower() in ("true", "1", "yes")
    )
    RESNET_BACKBONE: str = Field(default_factory=lambda: os.getenv("RESNET_BACKBONE", "resnet50"))
    WAV2VEC2_MODEL: str = Field(default_factory=lambda: os.getenv("WAV2VEC2_MODEL", "facebook/wav2vec2-base-960h"))
    BERT_MODEL: str = Field(default_factory=lambda: os.getenv("BERT_MODEL", "bert-base-uncased"))
    FUSION_TYPE: str = Field(default_factory=lambda: os.getenv("FUSION_TYPE", "gated"))
    FEATURE_DIM: int = Field(default_factory=lambda: int(os.getenv("FEATURE_DIM", "512")))
    PRETRAINED_FOUNDATION: bool = Field(
        default_factory=lambda: os.getenv("PRETRAINED_FOUNDATION", "true").lower() in ("true", "1", "yes")
    )
    DETERMINISTIC_INFERENCE: bool = Field(
        default_factory=lambda: os.getenv("DETERMINISTIC_INFERENCE", "true").lower() in ("true", "1", "yes")
    )

    # Contrastive Training (Phase 8)
    TRAINING_ENABLED: bool = Field(
        default_factory=lambda: os.getenv("TRAINING_ENABLED", "false").lower() in ("true", "1", "yes")
    )
    TRAINING_SEED: int = Field(default_factory=lambda: int(os.getenv("TRAINING_SEED", "42")))
    TRAINING_BATCH_SIZE: int = Field(default_factory=lambda: int(os.getenv("TRAINING_BATCH_SIZE", "32")))
    TRAINING_EPOCHS: int = Field(default_factory=lambda: int(os.getenv("TRAINING_EPOCHS", "10")))
    TRAINING_LEARNING_RATE: float = Field(default_factory=lambda: float(os.getenv("TRAINING_LEARNING_RATE", "1e-4")))
    TRAINING_WEIGHT_DECAY: float = Field(default_factory=lambda: float(os.getenv("TRAINING_WEIGHT_DECAY", "1e-5")))
    TRAINING_TEMPERATURE: float = Field(default_factory=lambda: float(os.getenv("TRAINING_TEMPERATURE", "0.07")))
    TRAINING_VALIDATION_SPLIT: float = Field(default_factory=lambda: float(os.getenv("TRAINING_VALIDATION_SPLIT", "0.2")))
    TRAINING_EARLY_STOPPING_PATIENCE: int = Field(default_factory=lambda: int(os.getenv("TRAINING_EARLY_STOPPING_PATIENCE", "3")))
    TRAINING_CHECKPOINT_DIR: str = Field(default_factory=lambda: os.getenv("TRAINING_CHECKPOINT_DIR", "data/checkpoints"))
    TRAINING_FREEZE_BACKBONES: bool = Field(
        default_factory=lambda: os.getenv("TRAINING_FREEZE_BACKBONES", "true").lower() in ("true", "1", "yes")
    )
    TRAINING_NUM_WORKERS: int = Field(default_factory=lambda: int(os.getenv("TRAINING_NUM_WORKERS", "0")))
    TRAINING_MODALITY_DROPOUT_PROB: float = Field(default_factory=lambda: float(os.getenv("TRAINING_MODALITY_DROPOUT_PROB", "0.0")))

    @field_validator("TRAINING_BATCH_SIZE", "TRAINING_EPOCHS", "TRAINING_EARLY_STOPPING_PATIENCE")
    @classmethod
    def validate_positive_integers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Training integer parameter must be strictly positive.")
        return v

    @field_validator("TRAINING_LEARNING_RATE", "TRAINING_TEMPERATURE")
    @classmethod
    def validate_positive_floats(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("Training float parameter must be strictly positive.")
        return v

    @field_validator("TRAINING_VALIDATION_SPLIT")
    @classmethod
    def validate_validation_split(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError("TRAINING_VALIDATION_SPLIT must be in [0.0, 1.0).")
        return v

    @field_validator("TRAINING_MODALITY_DROPOUT_PROB")
    @classmethod
    def validate_dropout_prob(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError("TRAINING_MODALITY_DROPOUT_PROB must be in [0.0, 1.0).")
        return v

    # Visual Pipeline & Thumbnail Intelligence (Phase 6)
    VISUAL_ENCODER: str = Field(default_factory=lambda: os.getenv("VISUAL_ENCODER", "resnet50"))
    VISUAL_FRAME_COUNT: int = Field(default_factory=lambda: int(os.getenv("VISUAL_FRAME_COUNT", "8")))
    VISUAL_FRAME_POOLING: str = Field(default_factory=lambda: os.getenv("VISUAL_FRAME_POOLING", "mean"))
    VISUAL_VIDEO_WEIGHT: float = Field(default_factory=lambda: float(os.getenv("VISUAL_VIDEO_WEIGHT", "0.7")))
    VISUAL_THUMBNAIL_WEIGHT: float = Field(default_factory=lambda: float(os.getenv("VISUAL_THUMBNAIL_WEIGHT", "0.3")))

    # Audio & Speech Intelligence (Phase 7)
    AUDIO_ENCODER: str = Field(default_factory=lambda: os.getenv("AUDIO_ENCODER", "wav2vec2"))
    AUDIO_SAMPLE_RATE: int = Field(default_factory=lambda: int(os.getenv("AUDIO_SAMPLE_RATE", "16000")))
    AUDIO_CHUNK_DURATION_SECONDS: float = Field(default_factory=lambda: float(os.getenv("AUDIO_CHUNK_DURATION_SECONDS", "20.0")))
    AUDIO_CHUNK_OVERLAP_SECONDS: float = Field(default_factory=lambda: float(os.getenv("AUDIO_CHUNK_OVERLAP_SECONDS", "2.0")))
    AUDIO_MAX_DURATION_SECONDS: float = Field(default_factory=lambda: float(os.getenv("AUDIO_MAX_DURATION_SECONDS", "600.0")))
    AUDIO_MAX_CHUNKS: int = Field(default_factory=lambda: int(os.getenv("AUDIO_MAX_CHUNKS", "30")))
    AUDIO_POOLING: str = Field(default_factory=lambda: os.getenv("AUDIO_POOLING", "mean"))
    AUDIO_SPEECH_RECOGNITION_ENABLED: bool = Field(
        default_factory=lambda: os.getenv("AUDIO_SPEECH_RECOGNITION_ENABLED", "false").lower() in ("true", "1", "yes")
    )

    # YouTube Data API v3
    YOUTUBE_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY"))
    YOUTUBE_API_BASE_URL: str = Field(
        default_factory=lambda: os.getenv("YOUTUBE_API_BASE_URL", "https://www.googleapis.com/youtube/v3")
    )
    YOUTUBE_API_TIMEOUT: float = Field(
        default_factory=lambda: float(os.getenv("YOUTUBE_API_TIMEOUT", "10.0"))
    )
    YOUTUBE_API_ENABLED: bool = Field(
        default_factory=lambda: os.getenv("YOUTUBE_API_ENABLED", "true").lower() in ("true", "1", "yes")
    )

    # Media Acquisition & Ingestion (Phase 4)
    MEDIA_STORAGE_DIR: str = Field(
        default_factory=lambda: os.getenv("MEDIA_STORAGE_DIR", "data/raw/youtube")
    )
    MAX_VIDEO_SIZE_MB: int = Field(
        default_factory=lambda: int(os.getenv("MAX_VIDEO_SIZE_MB", "100"))
    )
    MAX_AUDIO_SIZE_MB: int = Field(
        default_factory=lambda: int(os.getenv("MAX_AUDIO_SIZE_MB", "50"))
    )
    MAX_VIDEO_DURATION_SECONDS: int = Field(
        default_factory=lambda: int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "600"))
    )
    THUMBNAIL_MAX_SIZE_MB: int = Field(
        default_factory=lambda: int(os.getenv("THUMBNAIL_MAX_SIZE_MB", "10"))
    )
    YT_DLP_ENABLED: bool = Field(
        default_factory=lambda: os.getenv("YT_DLP_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    YT_DLP_TIMEOUT: float = Field(
        default_factory=lambda: float(os.getenv("YT_DLP_TIMEOUT", "60.0"))
    )
    INGESTION_MAX_RETRIES: int = Field(
        default_factory=lambda: int(os.getenv("INGESTION_MAX_RETRIES", "3"))
    )
    TRANSCRIPT_ENABLED: bool = Field(
        default_factory=lambda: os.getenv("TRANSCRIPT_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    FFMPEG_PATH: Optional[str] = Field(default_factory=resolve_ffmpeg_path)
    FFPROBE_PATH: Optional[str] = Field(default_factory=resolve_ffprobe_path)
    MAX_MEDIA_STORAGE_GB: float = Field(
        default_factory=lambda: float(os.getenv("MAX_MEDIA_STORAGE_GB", "10.0"))
    )
    INGESTION_CONCURRENCY: int = Field(
        default_factory=lambda: int(os.getenv("INGESTION_CONCURRENCY", "4"))
    )
    INGESTION_RATE_LIMIT_DELAY_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("INGESTION_RATE_LIMIT_DELAY_SECONDS", "0.5"))
    )

    # API & Security
    APP_ENV: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    API_TITLE: str = Field(default="Multimodal Video Recommender v3")
    API_VERSION: str = Field(default="3.1.0")
    API_HOST: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    API_PORT: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:8000,http://localhost:8001").split(",")
            if origin.strip()
        ]
    )
    ALLOWED_MEDIA_DIRS: list[str] = Field(
        default_factory=lambda: [
            d.strip()
            for d in os.getenv("ALLOWED_MEDIA_DIRS", "data,videos,tests").split(",")
            if d.strip()
        ]
    )

    # Vector Recommendation Engine (Phase 9)
    RECOMMENDATION_MODE: str = Field(default_factory=lambda: os.getenv("RECOMMENDATION_MODE", "TEXT_COENGAGEMENT_V2"))
    PHASE_8C_CHECKPOINT_PATH: str = Field(
        default_factory=lambda: os.getenv("PHASE_8C_CHECKPOINT_PATH", "data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt")
    )
    VECTOR_SEARCH_BACKEND: str = Field(default_factory=lambda: os.getenv("VECTOR_SEARCH_BACKEND", "auto"))
    VECTOR_DEFAULT_TOP_K: int = Field(default_factory=lambda: int(os.getenv("VECTOR_DEFAULT_TOP_K", "10")))
    VECTOR_MAX_TOP_K: int = Field(default_factory=lambda: int(os.getenv("VECTOR_MAX_TOP_K", "50")))
    VECTOR_BACKFILL_BATCH_SIZE: int = Field(default_factory=lambda: int(os.getenv("VECTOR_BACKFILL_BATCH_SIZE", "128")))
    VECTOR_HNSW_M: int = Field(default_factory=lambda: int(os.getenv("VECTOR_HNSW_M", "16")))
    VECTOR_HNSW_EF_CONSTRUCTION: int = Field(default_factory=lambda: int(os.getenv("VECTOR_HNSW_EF_CONSTRUCTION", "64")))
    VECTOR_HNSW_EF_SEARCH: int = Field(default_factory=lambda: int(os.getenv("VECTOR_HNSW_EF_SEARCH", "40")))

    # Candidate Re-Ranking (Phase 11B)
    RERANKING_ENABLED: bool = Field(
        default_factory=lambda: os.getenv("RERANKING_ENABLED", "false").lower() in ("true", "1", "yes")
    )
    RERANKING_DEFAULT_POOL_SIZE: int = Field(default_factory=lambda: int(os.getenv("RERANKING_DEFAULT_POOL_SIZE", "50")))
    RERANKING_WEIGHT_TEXT: float = Field(default_factory=lambda: float(os.getenv("RERANKING_WEIGHT_TEXT", "0.70")))
    RERANKING_WEIGHT_VISUAL: float = Field(default_factory=lambda: float(os.getenv("RERANKING_WEIGHT_VISUAL", "0.15")))
    RERANKING_WEIGHT_AUDIO: float = Field(default_factory=lambda: float(os.getenv("RERANKING_WEIGHT_AUDIO", "0.05")))
    RERANKING_WEIGHT_METADATA: float = Field(default_factory=lambda: float(os.getenv("RERANKING_WEIGHT_METADATA", "0.10")))


    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Singleton instance for direct import
settings = get_settings()


class DatabaseConfig:
    """Database configuration for backward compatibility."""

    @classmethod
    def get_database_url(cls, async_driver: bool = False) -> str:
        current_settings = get_settings()
        if async_driver:
            return current_settings.ASYNC_DATABASE_URL
        return current_settings.DATABASE_URL

    @classmethod
    def print_config(cls):
        s = get_settings()
        print(f"Database Configuration:")
        print(f"  Host: {s.POSTGRES_HOST}")
        print(f"  Port: {s.POSTGRES_PORT}")
        print(f"  User: {s.POSTGRES_USER}")
        print(f"  Database: {s.POSTGRES_DB}")
        print(f"  Pool Size: {s.DB_POOL_SIZE}")


# Default .env template
ENV_TEMPLATE = """# Environment & Application
APP_ENV=development
ALLOW_UNINITIALIZED_MODEL=true
MODEL_PATH=checkpoints/best.pt
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
ALLOWED_MEDIA_DIRS=data,videos,tests

# Database Configuration (Canonical port: 5432)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=video_recommender

# Connection Pool
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
"""


def create_env_template():
    """Create a .env.template file for reference."""
    env_path = Path(__file__).parent.parent.parent / ".env.template"
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(ENV_TEMPLATE)
        print(f"Created {env_path}")


if __name__ == "__main__":
    DatabaseConfig.print_config()
    print(f"\nDatabase URL: {DatabaseConfig.get_database_url()}")
