"""
Vector Recommendation Service.

Experiment / Phase 9 Mode: TEXT_COENGAGEMENT_V2
Coordinates candidate retrieval using the verified Phase 8C text co-engagement
linear projection head and frozen BERT backbone.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from sqlalchemy.orm import Session
from transformers import BertModel, BertTokenizer

from src.db.config import get_settings, Settings
from src.db.models import Video
from src.db.repository import VideoRepository
from src.models.text_projection_model import LinearProjectionModel, load_phase_8c_projection
from src.schemas.recommendations import (
    RecommendationHealthResponse,
    RecommendationItem,
    RecommendationResponse,
    SimilarVideosRequest,
)
from src.services.vector_index_service import VectorIndexService

logger = logging.getLogger(__name__)

RETRIEVAL_MODE = "TEXT_COENGAGEMENT_V2"
MODEL_VERSION = "EXP_MICROLENS_TEXT_COENGAGEMENT_V2"


class RecommendationService:
    """
    Recommendation Service using Phase 8C trained text co-engagement representations.
    Exposes safe text embedding generation, candidate retrieval, metadata hydration,
    and metadata filtering.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        vector_index_service: Optional[VectorIndexService] = None,
    ):
        self.settings = settings or get_settings()
        self.device = device or torch.device("cpu")
        self.checkpoint_path = Path(
            checkpoint_path or self.settings.PHASE_8C_CHECKPOINT_PATH
        )

        self.vector_index_service = vector_index_service or VectorIndexService(
            default_top_k=self.settings.VECTOR_DEFAULT_TOP_K,
            max_top_k=self.settings.VECTOR_MAX_TOP_K,
            backend=self.settings.VECTOR_SEARCH_BACKEND,
            ef_search=self.settings.VECTOR_HNSW_EF_SEARCH,
        )

        from src.services.reranking_service import CandidateReRankingService
        self.reranking_service = CandidateReRankingService(settings=self.settings)

        self.is_checkpoint_loaded = False
        self.checkpoint_sha256: Optional[str] = None
        self.projection_model: Optional[LinearProjectionModel] = None
        self.bert_backbone: Optional[BertModel] = None
        self.tokenizer: Optional[BertTokenizer] = None

        self._init_models()

    def _init_models(self):
        """Initialize frozen BERT backbone and load Phase 8C LinearProjectionModel."""
        # 1. Load Phase 8C projection checkpoint safely
        if self.checkpoint_path.exists():
            try:
                ckpt_bytes = self.checkpoint_path.read_bytes()
                self.checkpoint_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()
                self.projection_model = load_phase_8c_projection(
                    checkpoint_path=str(self.checkpoint_path),
                    device=self.device,
                )
                self.is_checkpoint_loaded = True
                logger.info(
                    "Phase 8C text projection loaded from %s (SHA256: %s)",
                    self.checkpoint_path,
                    self.checkpoint_sha256,
                )
            except Exception as e:
                logger.error("Failed to load Phase 8C checkpoint from %s: %s", self.checkpoint_path, e)
                self.is_checkpoint_loaded = False
        else:
            logger.warning("Phase 8C checkpoint not found at %s", self.checkpoint_path)
            self.is_checkpoint_loaded = False

        # 2. Initialize frozen BERT backbone and tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                self.settings.BERT_MODEL,
                local_files_only=True,
            )
            self.bert_backbone = BertModel.from_pretrained(
                self.settings.BERT_MODEL,
                local_files_only=True,
            ).to(self.device)
        except Exception:
            # Fallback if local_files_only fails in offline dev environment with cached weights
            try:
                self.tokenizer = BertTokenizer.from_pretrained(self.settings.BERT_MODEL)
                self.bert_backbone = BertModel.from_pretrained(self.settings.BERT_MODEL).to(self.device)
            except Exception as e:
                logger.warning("Could not initialize BERT backbone: %s", e)
                self.bert_backbone = None
                self.tokenizer = None

        if self.bert_backbone is not None:
            for p in self.bert_backbone.parameters():
                p.requires_grad = False
            self.bert_backbone.eval()

    def encode_text(self, text: Union[str, dict]) -> List[float]:
        """
        Generate 512-D text embedding using frozen BERT-base CLS + Phase 8C LinearProjection.
        """
        if not self.is_checkpoint_loaded or self.projection_model is None:
            raise RuntimeError(
                f"Recommendation service requires loaded Phase 8C checkpoint at '{self.checkpoint_path}'."
            )
        if self.bert_backbone is None or self.tokenizer is None:
            raise RuntimeError("BERT backbone or tokenizer is uninitialized.")

        # Extract title string
        if isinstance(text, dict):
            raw_title = str(text.get("title", "")).strip()
        else:
            raw_title = str(text).strip()

        if not raw_title:
            raise ValueError("Input text for embedding generation cannot be empty.")

        tokens = self.tokenizer(
            [raw_title],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            bert_out = self.bert_backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls_feat = bert_out.last_hidden_state[:, 0, :]  # (1, 768)
            emb = self.projection_model(cls_feat)  # (1, 512)

        return self.vector_index_service.validate_embedding(emb)

    def get_recommendations_for_video(
        self,
        session: Session,
        video_id: str,
        top_k: int = 10,
        exclude_video_ids: Optional[List[str]] = None,
        category_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        ranking_mode: str = "baseline",
        candidate_pool_size: int = 50,
        weights: Optional[Dict[str, float]] = None,
    ) -> RecommendationResponse:
        """
        Retrieve top-K video recommendations for a query video using Phase 8C embeddings,
        with optional Phase 11B candidate re-ranking.
        """
        repo = VideoRepository(session)
        query_video = repo.get_by_video_id(video_id)

        if not query_video:
            raise ValueError(f"Video '{video_id}' not found in database.")

        # Check or generate query embedding
        if query_video.embedding is None:
            if not query_video.title:
                raise ValueError(f"Video '{video_id}' has no embedding and title is missing.")
            emb = self.encode_text(query_video.title)
            repo.update_embedding(video_id, emb)
            query_embedding = emb
        else:
            query_embedding = self.vector_index_service.validate_embedding(query_video.embedding)

        # Build exclusions (always exclude query item)
        exclusions = set(exclude_video_ids or [])
        exclusions.add(video_id)

        # Stage 1: Candidate Retrieval
        stage1_k = max(candidate_pool_size, top_k) if ranking_mode == "rerank" else top_k
        results, backend_used, _ = self.vector_index_service.search_database(
            session=session,
            query_embedding=query_embedding,
            top_k=stage1_k,
            exclude_video_ids=list(exclusions),
            category_id=category_id,
            channel_id=channel_id,
        )

        # Stage 2: Candidate Re-Ranking (Phase 11B)
        reranking_applied = False
        if ranking_mode == "rerank" and results:
            custom_w = None
            if weights:
                from src.services.reranking_service import ReRankingWeights
                custom_w = ReRankingWeights(**weights)

            results = self.reranking_service.rerank_candidates(
                candidates=results,
                query_text_embedding=query_embedding,
                target_category_id=category_id or query_video.category_id,
                target_channel_id=channel_id or query_video.channel_id,
                weights=custom_w,
                top_k=top_k,
                ranking_mode="rerank",
            )
            reranking_applied = True
        else:
            results = results[:top_k]

        items: List[RecommendationItem] = []
        for rank, (cand_video, score) in enumerate(results, start=1):
            items.append(
                RecommendationItem(
                    video_id=cand_video.video_id,
                    title=cand_video.title or "",
                    score=round(float(score), 4),
                    rank=rank,
                    category_id=cand_video.category_id,
                    channel_title=cand_video.channel_title,
                    thumbnail_url=cand_video.thumbnail_url,
                    duration=cand_video.duration,
                )
            )

        return RecommendationResponse(
            query_video_id=video_id,
            query_title=query_video.title or "",
            recommendations=items,
            count=len(items),
            retrieval_mode=RETRIEVAL_MODE,
            retrieval_backend=backend_used,
            model_version=MODEL_VERSION,
            ranking_mode=ranking_mode,
            reranking_applied=reranking_applied,
        )

    def get_health_status(self, session: Session) -> RecommendationHealthResponse:
        """Report subsystem readiness and diagnostic metrics without leaking secrets."""
        repo = VideoRepository(session)
        try:
            total_indexed = repo.count()
        except Exception:
            total_indexed = 0

        backend = "pgvector" if self.vector_index_service.is_pgvector_available else "exact_fallback"

        return RecommendationHealthResponse(
            status="ok" if self.is_checkpoint_loaded else "uninitialized",
            retrieval_mode=RETRIEVAL_MODE,
            checkpoint_loaded=self.is_checkpoint_loaded,
            checkpoint_sha256=self.checkpoint_sha256,
            vector_backend=backend,
            pgvector_available=self.vector_index_service.is_pgvector_available,
            total_indexed_videos=total_indexed,
        )

    def retrieve_visual_candidates(
        self,
        session: Session,
        video_id: str,
        top_k: int = 5,
        modality: str = "VISUAL_FRAMES",
        feature_type: str = "resnet50_mean_2048",
        expected_dim: int = 2048,
    ) -> List[Dict[str, Any]]:
        """Phase 12C: Direct Visual Video-to-Video Retrieval delegation."""
        return self.reranking_service.retrieve_visual_candidates(
            query_video_id=video_id,
            session=session,
            top_k=top_k,
            modality=modality,
            feature_type=feature_type,
            expected_dim=expected_dim,
        )

