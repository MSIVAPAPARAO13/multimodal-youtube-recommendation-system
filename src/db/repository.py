"""
Repository Pattern for Database Operations

Repositories provide a clean interface for database operations,
hiding the SQL complexity from the rest of the application.
"""

from datetime import datetime
import math
from typing import Optional, Union, Tuple, List
import uuid
from sqlalchemy import select, text, func, delete
from sqlalchemy.orm import Session

from .models import (
    Video,
    VideoIngestion,
    VideoAsset,
    VideoFeature,
    UserInteraction,
    IngestionStatus,
    AssetType,
    AssetStatus,
    FeatureModality,
)


class VideoRepository:
    """
    Repository for Video operations.

    Provides methods for CRUD operations and similarity search.
    """

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        video_id: str,
        title: str,
        description: str = None,
        video_path: str = None,
        tags: list[str] = None,
        embedding: list[float] = None
    ) -> Video:
        """
        Create a new video record.

        Args:
            video_id: Unique identifier
            title: Video title
            description: Video description
            video_path: Path to video file
            tags: List of tags
            embedding: 512-dimensional embedding vector

        Returns:
            Created Video object
        """
        video = Video(
            video_id=video_id,
            title=title,
            description=description,
            video_path=video_path,
            tags=tags,
            embedding=embedding
        )

        self.session.add(video)
        self.session.commit()
        self.session.refresh(video)

        return video

    def get_by_video_id(self, video_id: str) -> Optional[Video]:
        """Get video by its external ID."""
        stmt = select(Video).where(Video.video_id == video_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_youtube_id(self, youtube_video_id: str) -> Optional[Video]:
        """Get video by its external YouTube ID."""
        stmt = select(Video).where(Video.youtube_video_id == youtube_video_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_all(self, limit: int = 100, offset: int = 0, query: Optional[str] = None) -> list[Video]:
        """Get all videos with pagination and optional search filter."""
        stmt = select(Video)
        if query:
            stmt = stmt.where(Video.title.ilike(f"%{query}%"))
        stmt = stmt.offset(offset).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def update_embedding(self, video_id: str, embedding: list[float]) -> Optional[Video]:
        """Update the embedding for a video."""
        video = self.get_by_video_id(video_id)
        if video:
            video.embedding = embedding
            self.session.commit()
            self.session.refresh(video)
        return video

    def delete(self, video_id: str) -> bool:
        """Delete a video by ID."""
        video = self.get_by_video_id(video_id)
        if video:
            self.session.delete(video)
            self.session.commit()
            return True
        return False

    def delete_all(self) -> int:
        """Delete all videos efficiently in a single query."""
        stmt = delete(Video)
        result = self.session.execute(stmt)
        self.session.commit()
        return result.rowcount or 0

    def count(self, query: Optional[str] = None) -> int:
        """Count total videos using optimized SQL COUNT query with optional search filter."""
        stmt = select(func.count(Video.id))
        if query:
            stmt = stmt.where(Video.title.ilike(f"%{query}%"))
        return self.session.execute(stmt).scalar() or 0

    def find_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        exclude_video_id: str = None
    ) -> list[tuple[Video, float]]:
        """
        Find similar videos using cosine similarity.

        This uses pgvector's efficient similarity search.

        Args:
            embedding: Query embedding vector
            limit: Number of results to return
            exclude_video_id: Video ID to exclude from results

        Returns:
            List of (Video, similarity_score) tuples
        """
        # Convert embedding to string format for pgvector
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        # Build query based on whether we have pgvector
        # Using cosine distance: 1 - cosine_similarity
        # So we order by distance ASC (smaller = more similar)
        if exclude_video_id:
            query = text("""
                SELECT *, 1 - (embedding <=> :embedding) as similarity
                FROM videos
                WHERE video_id != :exclude_id AND embedding IS NOT NULL
                ORDER BY embedding <=> :embedding
                LIMIT :limit
            """)
            result = self.session.execute(
                query,
                {"embedding": embedding_str, "exclude_id": exclude_video_id, "limit": limit}
            )
        else:
            query = text("""
                SELECT *, 1 - (embedding <=> :embedding) as similarity
                FROM videos
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :embedding
                LIMIT :limit
            """)
            result = self.session.execute(
                query,
                {"embedding": embedding_str, "limit": limit}
            )

        # Convert results
        videos_with_scores = []
        for row in result:
            video = Video(
                id=row.id,
                video_id=row.video_id,
                title=row.title,
                description=row.description,
                video_path=row.video_path,
                tags=row.tags,
                embedding=row.embedding,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            videos_with_scores.append((video, row.similarity))

        return videos_with_scores

    def find_similar_fallback(
        self,
        embedding: list[float],
        limit: int = 10,
        exclude_video_id: str = None
    ) -> list[tuple[Video, float]]:
        """
        Find similar videos without pgvector (slower, for testing).

        Computes cosine similarity in Python.
        """
        import numpy as np

        query_emb = np.array(embedding)

        all_videos = self.get_all(limit=1000)

        scores = []
        for video in all_videos:
            if video.embedding is None:
                continue
            if exclude_video_id and video.video_id == exclude_video_id:
                continue

            video_emb = np.array(video.embedding)

            # Cosine similarity
            similarity = np.dot(query_emb, video_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(video_emb)
            )

            scores.append((video, float(similarity)))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:limit]

    def get_by_video_ids(self, video_ids: list[str]) -> dict[str, Video]:
        """
        Batch retrieve videos by video_ids in a single query.
        Prevents N+1 query roundtrips.
        """
        if not video_ids:
            return {}
        stmt = select(Video).where(Video.video_id.in_(video_ids))
        videos = self.session.execute(stmt).scalars().all()
        return {v.video_id: v for v in videos}

    def upsert_embedding(
        self,
        video_id: str,
        embedding: list[float],
        title: Optional[str] = None,
        description: Optional[str] = None,
        category_id: Optional[str] = None,
        channel_title: Optional[str] = None,
    ) -> Video:
        """
        Idempotent upsert: updates embedding and metadata if video exists, creates if new.
        """
        safe_title = (title[:500] if title is not None else None)
        video = self.get_by_video_id(video_id)
        if video:
            video.embedding = embedding
            if safe_title is not None:
                video.title = safe_title
            if description is not None:
                video.description = description
            if category_id is not None:
                video.category_id = category_id
            if channel_title is not None:
                video.channel_title = channel_title
            self.session.commit()
            self.session.refresh(video)
            return video
        else:
            video = Video(
                video_id=video_id,
                title=safe_title or f"Video {video_id}",
                description=description,
                category_id=category_id,
                channel_title=channel_title,
                embedding=embedding
            )
            self.session.add(video)
            self.session.commit()
            self.session.refresh(video)
            return video

    def find_similar_with_filters(
        self,
        embedding: list[float],
        limit: int = 10,
        exclude_video_ids: Optional[list[str]] = None,
        category_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        backend: str = "auto",
        ef_search: Optional[int] = None,
        return_backend: bool = False,
    ) -> Union[list[tuple[Video, float]], tuple[list[tuple[Video, float]], str]]:
        """
        Find similar videos using cosine similarity with metadata filtering.
        Supports explicit backend selection:
        - "exact": Deterministic in-memory/exact cosine search.
        - "ann": Enforces pgvector HNSW search; raises RuntimeError if unavailable.
        - "auto": Prefers pgvector HNSW when available; falls back to exact search.
        """
        if backend not in ("auto", "exact", "ann"):
            raise ValueError(f"Invalid backend: '{backend}'. Must be one of 'auto', 'exact', 'ann'.")

        if ef_search is not None and ef_search < 1:
            raise ValueError(f"ef_search must be a positive integer, got {ef_search}")

        exclude_set = set(exclude_video_ids or [])

        # If explicitly exact mode, bypass pgvector directly
        if backend == "exact":
            results = self._find_similar_fallback_filtered(
                embedding=embedding,
                limit=limit,
                exclude_video_ids=exclude_set,
                category_id=category_id,
                channel_id=channel_id,
            )
            return (results, "exact") if return_backend else results

        # Check if database is PostgreSQL and pgvector is imported
        is_postgresql = False
        try:
            bind = self.session.get_bind()
            is_postgresql = bool(bind and bind.dialect.name == "postgresql")
        except Exception:
            pass

        from .models import HAS_PGVECTOR
        can_run_pgvector = is_postgresql and HAS_PGVECTOR

        if backend == "ann" and not can_run_pgvector:
            raise RuntimeError(
                f"ANN retrieval backend requested, but pgvector/PostgreSQL is unavailable "
                f"(is_postgresql={is_postgresql}, HAS_PGVECTOR={HAS_PGVECTOR})."
            )

        if can_run_pgvector:
            try:
                # Optionally configure HNSW search breadth scoped locally to transaction
                if ef_search is not None:
                    self.session.execute(
                        text("SET LOCAL hnsw.ef_search = :ef_search"),
                        {"ef_search": ef_search},
                    )

                # Parameterized pgvector query
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                where_clauses = ["embedding IS NOT NULL"]
                params = {"embedding": embedding_str, "limit": limit}

                if exclude_set:
                    exclude_keys = []
                    for i, ex_id in enumerate(sorted(list(exclude_set))):
                        k = f"ex_{i}"
                        params[k] = ex_id
                        exclude_keys.append(f":{k}")
                    where_clauses.append(f"video_id NOT IN ({', '.join(exclude_keys)})")

                if category_id:
                    params["category_id"] = category_id
                    where_clauses.append("category_id = :category_id")

                if channel_id:
                    params["channel_id"] = channel_id
                    where_clauses.append("channel_id = :channel_id")

                where_sql = " AND ".join(where_clauses)
                query = text(f"""
                    SELECT *, 1 - (embedding <=> :embedding) as similarity
                    FROM videos
                    WHERE {where_sql}
                    ORDER BY embedding <=> :embedding
                    LIMIT :limit
                """)
                result = self.session.execute(query, params)
                videos_with_scores = []
                for row in result:
                    v = Video(
                        id=row.id,
                        video_id=row.video_id,
                        title=row.title,
                        description=row.description,
                        video_path=row.video_path,
                        tags=row.tags,
                        category_id=row.category_id,
                        channel_title=row.channel_title,
                        embedding=row.embedding,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    )
                    videos_with_scores.append((v, float(row.similarity)))

                return (videos_with_scores, "pgvector_hnsw") if return_backend else videos_with_scores

            except Exception as e:
                self.session.rollback()
                if backend == "ann":
                    raise RuntimeError(f"ANN pgvector retrieval query failed: {e}") from e
                logger.warning("pgvector query failed in auto mode (%s). Falling back to exact search.", e)

        # Exact fallback path (for auto mode when pgvector unavailable or failed)
        fallback_results = self._find_similar_fallback_filtered(
            embedding=embedding,
            limit=limit,
            exclude_video_ids=exclude_set,
            category_id=category_id,
            channel_id=channel_id,
        )
        return (fallback_results, "exact_fallback") if return_backend else fallback_results

    def _find_similar_fallback_filtered(
        self,
        embedding: list[float],
        limit: int = 10,
        exclude_video_ids: Optional[set[str]] = None,
        category_id: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> list[tuple[Video, float]]:
        """Exact cosine similarity retrieval with metadata filtering in Python."""
        import numpy as np

        query_emb = np.array(embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query_emb)
        if q_norm == 0 or np.isnan(q_norm):
            return []

        stmt = select(Video).where(Video.embedding.isnot(None))
        if category_id:
            stmt = stmt.where(Video.category_id == category_id)
        if channel_id:
            stmt = stmt.where(Video.channel_id == channel_id)

        all_videos = list(self.session.execute(stmt).scalars().all())
        exclude_set = exclude_video_ids or set()

        scores = []
        for video in all_videos:
            if video.embedding is None or video.video_id in exclude_set:
                continue
            video_emb = np.array(video.embedding, dtype=np.float32)
            v_norm = np.linalg.norm(video_emb)
            if v_norm == 0 or np.isnan(v_norm):
                continue
            sim = float(np.dot(query_emb, video_emb) / (q_norm * v_norm))
            scores.append((video, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]


class InteractionRepository:
    """Repository for user interaction operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        user_id: str,
        video_id: str,
        interaction_type: str = "view",
        value: float = None,
        timestamp: Optional[datetime] = None,
    ) -> UserInteraction:
        """Record a user interaction."""
        interaction = UserInteraction(
            user_id=user_id,
            video_id=video_id,
            interaction_type=interaction_type,
            value=value,
            timestamp=timestamp if timestamp is not None else datetime.utcnow(),
        )

        self.session.add(interaction)
        self.session.commit()
        self.session.refresh(interaction)

        return interaction

    def get_user_history(
        self,
        user_id: str,
        limit: int = 100
    ) -> list[UserInteraction]:
        """Get a user's interaction history."""
        stmt = (
            select(UserInteraction)
            .where(UserInteraction.user_id == user_id)
            .order_by(UserInteraction.timestamp.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_video_interactions(
        self,
        video_id: str,
        limit: int = 100
    ) -> list[UserInteraction]:
        """Get all interactions for a video."""
        stmt = (
            select(UserInteraction)
            .where(UserInteraction.video_id == video_id)
            .order_by(UserInteraction.timestamp.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_user_history_before(
        self,
        user_id: str,
        cutoff_timestamp: Optional[datetime] = None,
        limit: int = 100
    ) -> list[UserInteraction]:
        """
        Get a user's interaction history strictly before cutoff_timestamp.
        Guarantees zero future-event leakage during personalized retrieval or evaluation.
        Results are ordered chronologically ascending.
        """
        stmt = select(UserInteraction).where(UserInteraction.user_id == user_id)
        if cutoff_timestamp is not None:
            stmt = stmt.where(UserInteraction.timestamp < cutoff_timestamp)
        stmt = stmt.order_by(UserInteraction.timestamp.asc()).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def batch_create(
        self,
        interactions_data: list[dict]
    ) -> int:
        """Batch insert interaction records."""
        if not interactions_data:
            return 0
        records = [
            UserInteraction(
                user_id=d["user_id"],
                video_id=d["video_id"],
                interaction_type=d.get("interaction_type", "view"),
                value=d.get("value"),
                timestamp=d.get("timestamp", datetime.utcnow())
            )
            for d in interactions_data
        ]
        self.session.add_all(records)
        self.session.commit()
        return len(records)

    def clear_user_history(self, user_id: str) -> int:
        """Delete all recorded interactions for a user."""
        stmt = delete(UserInteraction).where(UserInteraction.user_id == user_id)
        result = self.session.execute(stmt)
        self.session.commit()
        return result.rowcount or 0


class YouTubeVideoRepository:
    """
    Dedicated repository for YouTube video metadata persistence and ingestion tracking.
    """

    def __init__(self, session: Session):
        self.session = session

    def find_by_youtube_id(self, youtube_video_id: str) -> Optional[Video]:
        """Find video by YouTube video ID."""
        stmt = select(Video).where(Video.youtube_video_id == youtube_video_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def find_by_video_id(self, video_id: str) -> Optional[Video]:
        """Find video by internal video ID."""
        stmt = select(Video).where(Video.video_id == video_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def upsert_video_from_metadata(
        self,
        youtube_video_id: str,
        title: str,
        description: Optional[str] = None,
        channel_id: Optional[str] = None,
        channel_title: Optional[str] = None,
        published_at: Optional[datetime] = None,
        category_id: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
        duration: Optional[int] = None,
        view_count: Optional[int] = None,
        like_count: Optional[int] = None,
        comment_count: Optional[int] = None,
        privacy_status: Optional[str] = None,
        tags: Optional[list[str]] = None,
        video_path: Optional[str] = None,
        ingestion_status: str = IngestionStatus.METADATA_READY.value
    ) -> Video:
        """
        Idempotently create or update a Video record from normalized YouTube metadata.
        """
        video = self.find_by_youtube_id(youtube_video_id)
        if not video:
            video = self.find_by_video_id(f"yt_{youtube_video_id}")

        now = datetime.utcnow()
        if not video:
            video = Video(
                video_id=f"yt_{youtube_video_id}",
                youtube_video_id=youtube_video_id,
                title=title,
                description=description,
                video_path=video_path,
                channel_id=channel_id,
                channel_title=channel_title,
                published_at=published_at,
                category_id=category_id,
                thumbnail_url=thumbnail_url,
                duration=duration,
                view_count=view_count,
                like_count=like_count,
                comment_count=comment_count,
                privacy_status=privacy_status,
                tags=tags or [],
                ingestion_status=ingestion_status,
                last_ingestion_attempt=now
            )
            self.session.add(video)
        else:
            video.youtube_video_id = youtube_video_id
            video.title = title
            if description is not None:
                video.description = description
            if channel_id is not None:
                video.channel_id = channel_id
            if channel_title is not None:
                video.channel_title = channel_title
            if published_at is not None:
                video.published_at = published_at
            if category_id is not None:
                video.category_id = category_id
            if thumbnail_url is not None:
                video.thumbnail_url = thumbnail_url
            if duration is not None:
                video.duration = duration
            if view_count is not None:
                video.view_count = view_count
            if like_count is not None:
                video.like_count = like_count
            if comment_count is not None:
                video.comment_count = comment_count
            if privacy_status is not None:
                video.privacy_status = privacy_status
            if tags is not None:
                video.tags = tags
            if video_path is not None:
                video.video_path = video_path
            video.ingestion_status = ingestion_status
            video.last_ingestion_attempt = now

        self.session.commit()
        self.session.refresh(video)
        return video

    def get_or_create_ingestion(
        self,
        youtube_video_id: str,
        video_id: Optional[str] = None,
        max_attempts: int = 3
    ) -> VideoIngestion:
        """
        Idempotently get or create a VideoIngestion record.
        """
        stmt = select(VideoIngestion).where(VideoIngestion.youtube_video_id == youtube_video_id)
        ingestion = self.session.execute(stmt).scalar_one_or_none()
        if not ingestion:
            ingestion = VideoIngestion(
                youtube_video_id=youtube_video_id,
                video_id=video_id or f"yt_{youtube_video_id}",
                status=IngestionStatus.PENDING.value,
                max_attempts=max_attempts
            )
            self.session.add(ingestion)
            self.session.commit()
            self.session.refresh(ingestion)
        elif video_id and not ingestion.video_id:
            ingestion.video_id = video_id
            self.session.commit()
            self.session.refresh(ingestion)
        return ingestion

    def get_ingestion(self, identifier: str) -> Optional[VideoIngestion]:
        """
        Find VideoIngestion by external YouTube video ID or internal video ID.
        """
        stmt = select(VideoIngestion).where(
            (VideoIngestion.youtube_video_id == identifier) |
            (VideoIngestion.video_id == identifier)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def update_ingestion_state(
        self,
        youtube_video_id: str,
        status: IngestionStatus | str,
        error_message: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        thumbnail_path: Optional[str] = None,
        transcript_text: Optional[str] = None,
        transcript_language: Optional[str] = None,
        transcript_available: Optional[bool] = None,
        transcript_provider: Optional[str] = None,
        frames_path: Optional[str] = None,
        frames_count: Optional[int] = None,
        pipeline_version: Optional[str] = None,
        increment_attempt: bool = False
    ) -> Optional[VideoIngestion]:
        """
        Update ingestion lifecycle status, paths, errors, and attempt counters.
        """
        status_val = status.value if isinstance(status, IngestionStatus) else str(status)
        stmt = select(VideoIngestion).where(VideoIngestion.youtube_video_id == youtube_video_id)
        ingestion = self.session.execute(stmt).scalar_one_or_none()
        if not ingestion:
            return None

        now = datetime.utcnow()
        ingestion.status = status_val
        if increment_attempt:
            ingestion.attempt_count += 1
            ingestion.started_at = now

        if error_message is not None:
            ingestion.error_message = error_message
        if video_path is not None:
            ingestion.video_path = video_path
        if audio_path is not None:
            ingestion.audio_path = audio_path
        if thumbnail_path is not None:
            ingestion.thumbnail_path = thumbnail_path
        if transcript_text is not None:
            ingestion.transcript_text = transcript_text
        if transcript_language is not None:
            ingestion.transcript_language = transcript_language
        if transcript_available is not None:
            ingestion.transcript_available = transcript_available
        if transcript_provider is not None:
            ingestion.transcript_provider = transcript_provider
        if frames_path is not None:
            ingestion.frames_path = frames_path
        if frames_count is not None:
            ingestion.frames_count = frames_count
        if pipeline_version is not None:
            ingestion.pipeline_version = pipeline_version
        if status_val in (IngestionStatus.COMPLETED.value, IngestionStatus.MEDIA_READY.value):
            ingestion.completed_at = now
            ingestion.error_message = None

        # Sync status to Video entity if present
        video = self.find_by_youtube_id(youtube_video_id)
        if video:
            video.ingestion_status = status_val
            video.last_ingestion_attempt = now
            if error_message is not None:
                video.ingestion_error = error_message
            if video_path is not None:
                video.video_path = video_path

        self.session.commit()
        self.session.refresh(ingestion)
        return ingestion


class VideoAssetRepository:
    """
    Repository for managing multimodal video assets (video, audio, thumbnails, frames).
    Handles idempotent upserts, status tracking, checksum validation, and coverage queries.
    """

    def __init__(self, session: Session):
        self.session = session

    def upsert_asset(
        self,
        video_id: str,
        asset_type: Union[str, AssetType],
        storage_path: str,
        youtube_video_id: Optional[str] = None,
        source_url: Optional[str] = None,
        mime_type: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        checksum_sha256: Optional[str] = None,
        status: Union[str, AssetStatus] = AssetStatus.READY,
        error_category: Optional[str] = None,
        error_message: Optional[str] = None,
        pipeline_version: str = "12A",
        extra_metadata: Optional[dict] = None,
    ) -> VideoAsset:
        """Idempotently create or update a VideoAsset record."""
        type_str = asset_type.value if isinstance(asset_type, AssetType) else str(asset_type)
        status_str = status.value if isinstance(status, AssetStatus) else str(status)

        stmt = select(VideoAsset).where(
            VideoAsset.video_id == video_id,
            VideoAsset.asset_type == type_str,
            VideoAsset.storage_path == storage_path,
        )
        asset = self.session.execute(stmt).scalar_one_or_none()
        now = datetime.utcnow()

        if not asset:
            asset = VideoAsset(
                video_id=video_id,
                youtube_video_id=youtube_video_id,
                asset_type=type_str,
                source_url=source_url,
                storage_path=storage_path,
                mime_type=mime_type,
                file_size_bytes=file_size_bytes,
                checksum_sha256=checksum_sha256,
                status=status_str,
                attempt_count=1,
                error_category=error_category,
                error_message=error_message,
                pipeline_version=pipeline_version,
                extra_metadata=extra_metadata or {},
                created_at=now,
                updated_at=now,
            )
            self.session.add(asset)
        else:
            if youtube_video_id is not None:
                asset.youtube_video_id = youtube_video_id
            if source_url is not None:
                asset.source_url = source_url
            if mime_type is not None:
                asset.mime_type = mime_type
            if file_size_bytes is not None:
                asset.file_size_bytes = file_size_bytes
            if checksum_sha256 is not None:
                asset.checksum_sha256 = checksum_sha256
            asset.status = status_str
            asset.attempt_count += 1
            if error_category is not None:
                asset.error_category = error_category
            if error_message is not None:
                asset.error_message = error_message
            if extra_metadata is not None:
                asset.extra_metadata = extra_metadata
            asset.pipeline_version = pipeline_version
            asset.updated_at = now

        self.session.commit()
        self.session.refresh(asset)
        return asset

    def get_assets_for_video(self, video_id: str) -> list[VideoAsset]:
        """Retrieve all registered assets for a given internal video_id."""
        stmt = select(VideoAsset).where(VideoAsset.video_id == video_id).order_by(VideoAsset.created_at.asc())
        return list(self.session.execute(stmt).scalars().all())

    def get_asset_by_type(self, video_id: str, asset_type: Union[str, AssetType]) -> Optional[VideoAsset]:
        """Retrieve primary asset of a specific type for a video."""
        type_str = asset_type.value if isinstance(asset_type, AssetType) else str(asset_type)
        stmt = select(VideoAsset).where(
            VideoAsset.video_id == video_id,
            VideoAsset.asset_type == type_str
        )
        return self.session.execute(stmt).scalars().first()

    def get_by_checksum(self, checksum_sha256: str) -> Optional[VideoAsset]:
        """Look up asset by its SHA-256 content checksum."""
        stmt = select(VideoAsset).where(VideoAsset.checksum_sha256 == checksum_sha256)
        return self.session.execute(stmt).scalar_one_or_none()

    def count_by_type(self) -> dict[str, int]:
        """Return counts of ready assets grouped by asset_type."""
        stmt = select(
            VideoAsset.asset_type,
            func.count(VideoAsset.id)
        ).where(VideoAsset.status == AssetStatus.READY.value).group_by(VideoAsset.asset_type)
        rows = self.session.execute(stmt).all()
        return {row[0]: row[1] for row in rows}

    def count_total(self) -> int:
        """Return total number of asset records."""
        return self.session.scalar(select(func.count(VideoAsset.id))) or 0

    def delete_assets_for_video(self, video_id: str) -> int:
        """Delete all asset records for a specific video."""
        stmt = delete(VideoAsset).where(VideoAsset.video_id == video_id)
        result = self.session.execute(stmt)
        self.session.commit()
        return result.rowcount


class VideoFeatureRepository:
    """
    Repository for VideoFeature operations (Phase 12B).
    Provides idempotent feature persistence, dimension & finiteness validation,
    checksum provenance checking, and stale feature invalidation.
    """

    def __init__(self, session: Session):
        self.session = session

    def validate_embedding(
        self,
        embedding: Union[list[float], tuple[float, ...]],
        expected_dim: int,
    ) -> list[float]:
        """Validate embedding length, finiteness (no NaN/Inf), and type."""
        if not isinstance(embedding, (list, tuple)):
            raise ValueError(f"Embedding must be a list or tuple of floats, got {type(embedding)}")
        if len(embedding) != expected_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}")
        clean_emb = []
        for i, val in enumerate(embedding):
            f_val = float(val)
            if not math.isfinite(f_val):
                raise ValueError(f"Embedding contains non-finite value ({f_val}) at index {i}")
            clean_emb.append(f_val)
        return clean_emb

    def upsert_feature(
        self,
        video_id: str,
        modality: Union[str, FeatureModality],
        feature_type: str,
        encoder_name: str,
        encoder_version: str,
        dimension: int,
        source_checksum: str,
        embedding: list[float],
        youtube_video_id: Optional[str] = None,
        asset_id: Optional[uuid.UUID] = None,
        checkpoint_id: Optional[str] = None,
        preprocessing_config: Optional[dict] = None,
        is_trained: bool = False,
        validation_status: str = "VALID",
        pipeline_version: str = "12B",
    ) -> VideoFeature:
        """Idempotently persist or update a validated VideoFeature record."""
        mod_str = modality.value if isinstance(modality, FeatureModality) else str(modality)
        validated_emb = self.validate_embedding(embedding, expected_dim=dimension)

        stmt = select(VideoFeature).where(
            VideoFeature.video_id == video_id,
            VideoFeature.modality == mod_str,
            VideoFeature.feature_type == feature_type,
            VideoFeature.encoder_name == encoder_name,
        )
        feature = self.session.execute(stmt).scalar_one_or_none()
        now = datetime.utcnow()

        if not feature:
            feature = VideoFeature(
                video_id=video_id,
                youtube_video_id=youtube_video_id,
                asset_id=asset_id,
                modality=mod_str,
                feature_type=feature_type,
                encoder_name=encoder_name,
                encoder_version=encoder_version,
                checkpoint_id=checkpoint_id,
                dimension=dimension,
                preprocessing_config=preprocessing_config or {},
                source_checksum=source_checksum,
                embedding=validated_emb,
                is_valid=True,
                is_stale=False,
                is_trained=is_trained,
                validation_status=validation_status,
                pipeline_version=pipeline_version,
                created_at=now,
                updated_at=now,
            )
            self.session.add(feature)
        else:
            if youtube_video_id is not None:
                feature.youtube_video_id = youtube_video_id
            if asset_id is not None:
                feature.asset_id = asset_id
            feature.encoder_version = encoder_version
            if checkpoint_id is not None:
                feature.checkpoint_id = checkpoint_id
            feature.dimension = dimension
            if preprocessing_config is not None:
                feature.preprocessing_config = preprocessing_config
            feature.source_checksum = source_checksum
            feature.embedding = validated_emb
            feature.is_valid = True
            feature.is_stale = False
            feature.is_trained = is_trained
            feature.validation_status = validation_status
            feature.pipeline_version = pipeline_version
            feature.updated_at = now

        self.session.commit()
        self.session.refresh(feature)
        return feature

    def get_feature(
        self,
        video_id: str,
        modality: Union[str, FeatureModality],
        feature_type: str,
    ) -> Optional[VideoFeature]:
        """Retrieve single feature for a video by modality and feature_type."""
        mod_str = modality.value if isinstance(modality, FeatureModality) else str(modality)
        stmt = select(VideoFeature).where(
            VideoFeature.video_id == video_id,
            VideoFeature.modality == mod_str,
            VideoFeature.feature_type == feature_type,
            VideoFeature.is_valid.is_(True),
            VideoFeature.is_stale.is_(False),
        )
        return self.session.execute(stmt).scalars().first()

    def get_features_for_video(self, video_id: str) -> list[VideoFeature]:
        """Retrieve all active features for a given internal video_id."""
        stmt = select(VideoFeature).where(
            VideoFeature.video_id == video_id,
            VideoFeature.is_valid.is_(True),
            VideoFeature.is_stale.is_(False),
        ).order_by(VideoFeature.created_at.asc())
        return list(self.session.execute(stmt).scalars().all())

    def get_features_by_modality(
        self,
        modality: Union[str, FeatureModality],
        feature_type: Optional[str] = None,
    ) -> list[VideoFeature]:
        """Retrieve all active features for a specific modality."""
        mod_str = modality.value if isinstance(modality, FeatureModality) else str(modality)
        stmt = select(VideoFeature).where(
            VideoFeature.modality == mod_str,
            VideoFeature.is_valid.is_(True),
            VideoFeature.is_stale.is_(False),
        )
        if feature_type:
            stmt = stmt.where(VideoFeature.feature_type == feature_type)
        return list(self.session.execute(stmt).scalars().all())

    def get_features_by_video_ids(
        self,
        video_ids: list[str],
        modality: Union[str, FeatureModality],
        feature_type: str,
    ) -> dict[str, list[float]]:
        """Batch retrieve active features for multiple videos in a single query (Phase 12C optimization)."""
        if not video_ids:
            return {}
        mod_str = modality.value if isinstance(modality, FeatureModality) else str(modality)
        stmt = select(VideoFeature).where(
            VideoFeature.video_id.in_(video_ids),
            VideoFeature.modality == mod_str,
            VideoFeature.feature_type == feature_type,
            VideoFeature.is_valid.is_(True),
            VideoFeature.is_stale.is_(False),
        )
        features = self.session.execute(stmt).scalars().all()
        return {f.video_id: f.embedding for f in features if f.embedding}

    def invalidate_stale_features(
        self,
        video_id: str,
        modality: Union[str, FeatureModality],
        current_source_checksum: str,
    ) -> int:
        """Mark features as stale if source checksum has changed."""
        mod_str = modality.value if isinstance(modality, FeatureModality) else str(modality)
        stmt = select(VideoFeature).where(
            VideoFeature.video_id == video_id,
            VideoFeature.modality == mod_str,
            VideoFeature.source_checksum != current_source_checksum,
            VideoFeature.is_stale.is_(False),
        )
        stale_records = list(self.session.execute(stmt).scalars().all())
        count = 0
        now = datetime.utcnow()
        for rec in stale_records:
            rec.is_stale = True
            rec.validation_status = "STALE"
            rec.updated_at = now
            count += 1
        if count > 0:
            self.session.commit()
        return count

    def count_by_modality(self) -> dict[str, int]:
        """Return counts of valid features grouped by modality."""
        stmt = select(
            VideoFeature.modality,
            func.count(VideoFeature.id)
        ).where(
            VideoFeature.is_valid.is_(True),
            VideoFeature.is_stale.is_(False),
        ).group_by(VideoFeature.modality)
        rows = self.session.execute(stmt).all()
        return {row[0]: row[1] for row in rows}

    def count_total(self) -> int:
        """Return total number of valid, active feature records."""
        stmt = select(func.count(VideoFeature.id)).where(
            VideoFeature.is_valid.is_(True),
            VideoFeature.is_stale.is_(False),
        )
        return self.session.scalar(stmt) or 0

    def delete_features_for_video(self, video_id: str) -> int:
        """Delete all feature records for a video."""
        stmt = delete(VideoFeature).where(VideoFeature.video_id == video_id)
        result = self.session.execute(stmt)
        self.session.commit()
        return result.rowcount


