"""
TextEncoder: Extracts 512-dimensional text embeddings using BERT baseline.

Includes support for:
1. Canonical YouTubeTextInput with structured section formatting (title, description, tags, transcript)
2. Token-aware window chunking and hierarchical pooling via TextChunker
3. Deterministic inference with frozen pretrained backbone
4. Experimental abstraction for sentence-transformers/all-MiniLM-L6-v2
5. Numerical stability and diagnostic quality metrics
"""

from typing import Union, Any, Optional
import logging
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from src.features.text_processor import (
    YouTubeTextInput,
    TextProcessor,
    TextQualityDiagnostics,
    clean_transcript,
)
from src.features.text_chunker import TextChunker

logger = logging.getLogger(__name__)


def format_text_input(
    item: Union[str, dict[str, Any]],
    include_metadata: bool = True
) -> str:
    """
    Format multimodal text input into a single string for BERT.
    Supports title, description, tags, and transcript, while remaining
    fully backward compatible with simple title-only strings.

    Args:
        item: Raw text string or dictionary with metadata fields
        include_metadata: Whether to append description, tags, and transcript

    Returns:
        Formatted string representation
    """
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)

    title = str(item.get("title", "")).strip()
    if not include_metadata:
        return title

    parts = []
    if title:
        parts.append(title)

    description = str(item.get("description", "")).strip()
    if description:
        parts.append(description[:200])

    tags = item.get("tags")
    if tags:
        if isinstance(tags, list):
            valid_tags = [str(t).strip() for t in tags if str(t).strip()]
            if valid_tags:
                parts.append("Tags: " + ", ".join(valid_tags))
        elif isinstance(tags, str) and tags.strip():
            parts.append("Tags: " + tags.strip())

    transcript = str(item.get("transcript", "")).strip()
    if transcript:
        parts.append("Transcript: " + transcript[:300])

    return ". ".join(parts) if parts else ""


class TextEncoder(nn.Module):
    """
    Encodes text (titles, descriptions, transcripts) into 512-dimensional embeddings
    using a frozen BERT baseline backbone and trained projection head.

    Args:
        model_name: Which BERT model to use ('bert-base-uncased' by default)
        feature_dim: Final embedding dimension (512 to match video/audio)
        aggregation: Token aggregation method ('cls', 'mean', 'attention')
        freeze_backbone: If True, freeze BERT backbone weights
        max_length: Maximum number of tokens per sequence (default 128)
        pretrained: If True, load pretrained HuggingFace weights
        use_chunking: If True, enable TextChunker for long text/transcripts
        chunk_size: Window size in tokens for chunking (default 128)
        chunk_overlap: Overlap between consecutive chunks (default 32)
        max_total_tokens: Overall maximum token budget across chunks (default 1024)
        pooling_method: How to pool chunk embeddings ('mean', 'decay', 'weighted')
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        feature_dim: int = 512,
        aggregation: str = "cls",
        freeze_backbone: bool = True,
        max_length: int = 128,
        pretrained: bool = True,
        use_chunking: bool = True,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        max_total_tokens: int = 1024,
        pooling_method: str = "mean",
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.aggregation = aggregation
        self.max_length = max_length
        self.pretrained = pretrained
        self.model_name = model_name
        self.use_chunking = use_chunking
        self.pooling_method = pooling_method

        # Load BERT backbone
        if pretrained:
            self.backbone = BertModel.from_pretrained(model_name)
        else:
            from transformers import BertConfig
            config = BertConfig()
            self.backbone = BertModel(config)

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        if "large" in model_name:
            backbone_dim = 1024
        else:
            backbone_dim = 768

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection: Map BERT features (768) to feature_dim (512)
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, feature_dim)
        )

        # Optional: Attention mechanism for weighted token aggregation
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # Text chunker and processor
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_tokens=max_total_tokens
        )
        self.processor = TextProcessor()

    @staticmethod
    def format_input(item: Union[str, dict[str, Any]]) -> str:
        """Helper to format a single title or metadata dictionary."""
        return format_text_input(item)

    def tokenize(self, texts: list[Union[str, dict[str, Any]]]) -> dict:
        """
        Convert list of strings or metadata dicts to token IDs.
        """
        normalized_texts = [format_text_input(t) for t in texts]
        return self.tokenizer(
            normalized_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def encode_single_long_text(
        self,
        text: str
    ) -> tuple[torch.Tensor, int, int]:
        """
        Encode a single long text or structured text segment using TextChunker.

        Returns:
            embedding: (1, feature_dim) tensor
            num_chunks: number of chunks evaluated
            token_count: total tokens in text
        """
        chunk_data = self.chunker.chunk_text(
            text=text,
            tokenizer=self.tokenizer,
            add_special_tokens=True
        )

        input_ids = chunk_data["input_ids"].to(self.backbone.device)
        attention_mask = chunk_data["attention_mask"].to(self.backbone.device)
        num_chunks = chunk_data["num_chunks"]
        total_tokens = chunk_data["total_tokens"]

        # Run BERT over all chunks in a single batched pass
        chunk_embeddings = self._forward_tokens(input_ids, attention_mask)

        # Pool across chunk embeddings
        doc_embedding = self.chunker.pool_chunk_embeddings(
            chunk_embeddings=chunk_embeddings,
            method=self.pooling_method
        )

        return doc_embedding, num_chunks, total_tokens

    def _forward_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Internal helper: execute BERT forward pass on tokenized batch."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state  # (B, seq_len, 768)

        if self.aggregation == "cls":
            features = hidden_states[:, 0, :]      # (B, 768)
        elif self.aggregation == "mean":
            mask = attention_mask.unsqueeze(-1)
            features = (hidden_states * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        elif self.aggregation == "attention":
            projected = self.projection(hidden_states)
            attention_scores = self.attention(projected)
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    attention_mask.unsqueeze(-1) == 0,
                    float("-inf")
                )
            attention_weights = torch.softmax(attention_scores, dim=1)
            text_emb = (projected * attention_weights).sum(dim=1)
            return torch.nan_to_num(text_emb, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            features = hidden_states[:, 0, :]

        text_emb = self.projection(features)
        return torch.nan_to_num(text_emb, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        texts: list[Union[str, dict[str, Any], YouTubeTextInput]] = None
    ) -> torch.Tensor:
        """
        Forward pass producing 512-dimensional text embeddings.

        Args:
            input_ids: Optional pre-tokenized input IDs (B, seq_len)
            attention_mask: Optional attention mask (B, seq_len)
            texts: List of strings or metadata dictionaries / YouTubeTextInput

        Returns:
            embeddings: Tensor of shape (batch_size, feature_dim)
        """
        # Option 1: Direct token tensors provided
        if input_ids is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            input_ids = input_ids.to(self.backbone.device)
            attention_mask = attention_mask.to(self.backbone.device)
            return self._forward_tokens(input_ids, attention_mask)

        # Option 2: List of text inputs
        if texts is not None:
            # Check if any input is a dictionary containing a transcript or structured text
            # requiring long-text chunking
            has_structured_or_long = any(
                isinstance(t, (dict, YouTubeTextInput)) or (isinstance(t, str) and len(t) > 500)
                for t in texts
            )

            if self.use_chunking and has_structured_or_long:
                # Process each item with chunking if it contains structured fields
                embeddings = []
                for item in texts:
                    if isinstance(item, (dict, YouTubeTextInput)):
                        structured_str = self.processor.format_structured_text(item)
                    else:
                        structured_str = str(item)
                    emb, _, _ = self.encode_single_long_text(structured_str)
                    embeddings.append(emb)
                return torch.cat(embeddings, dim=0)

            # Standard batch tokenization for short text / titles
            tokens = self.tokenize(texts)
            input_ids = tokens["input_ids"].to(self.backbone.device)
            attention_mask = tokens["attention_mask"].to(self.backbone.device)
            return self._forward_tokens(input_ids, attention_mask)

        raise ValueError("Either input_ids or texts must be provided to TextEncoder.forward()")


class SentenceTransformerExperimentalEncoder(nn.Module):
    """
    Experimental Model Abstraction (TASK 7).

    Enables offline comparison between the baseline BERT model and
    'sentence-transformers/all-MiniLM-L6-v2'.

    IMPORTANT:
    - This model is NOT used as the default production model.
    - It is NOT downloaded during normal unit tests.
    - Default remains bert-base-uncased.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        feature_dim: int = 512,
        pretrained: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self._is_loaded = False

        # Projection from MiniLM 384-d output to canonical 512-d feature space
        self.projection = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, feature_dim)
        )

        if pretrained:
            self._load_model()

    def _load_model(self):
        """Lazy loader: avoids downloading models during standard initialization."""
        try:
            from transformers import AutoModel, AutoTokenizer
            self.backbone = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            self._is_loaded = True
        except Exception as e:
            logger.warning(
                "Could not load experimental model %s: %s. Encoder remains unloaded.",
                self.model_name, e
            )
            self._is_loaded = False

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Forward pass projecting MiniLM embeddings to 512 dimensions."""
        if not self._is_loaded:
            raise RuntimeError(
                f"Experimental model '{self.model_name}' has not been loaded. "
                "Do not invoke during standard unit tests."
            )

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        tokens = {k: v.to(self.projection[0].weight.device) for k, v in tokens.items()}

        outputs = self.backbone(**tokens)
        # Mean pooling across tokens
        mask = tokens["attention_mask"].unsqueeze(-1)
        mean_pooled = (outputs.last_hidden_state * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        projected = self.projection(mean_pooled)
        return torch.nan_to_num(projected, nan=0.0)


def create_text_encoder(
    model_type: str = "bert",
    feature_dim: int = 512,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory creating either the production BERT encoder or the experimental MiniLM encoder.

    Args:
        model_type: 'bert' (default production baseline) or 'sentence-transformers'
        feature_dim: Output dimension (must be 512)
        pretrained: Whether to load pretrained weights
    """
    if model_type == "bert":
        return TextEncoder(
            feature_dim=feature_dim,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type in ("sentence-transformers", "all-minilm"):
        return SentenceTransformerExperimentalEncoder(
            feature_dim=feature_dim,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown text encoder model_type '{model_type}'. Expected 'bert' or 'sentence-transformers'.")
