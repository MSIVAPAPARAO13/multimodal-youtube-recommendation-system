"""
TextChunker: Token-aware chunking and hierarchical pooling for long texts and transcripts.

Prevents silent truncation of long transcripts by splitting token sequences
into overlapping windows, encoding each window via BERT, and pooling chunk representations
into a cohesive 512-dimensional document embedding.
"""

import logging
from typing import Optional, Union, Any
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 128
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_MAX_TOKENS = 1024


class TextChunker:
    """
    Manages token sequence chunking with configurable window size, overlap,
    and pooling strategies.

    Args:
        chunk_size: Maximum tokens per chunk (default 128)
        chunk_overlap: Number of overlapping tokens between successive chunks (default 32)
        max_tokens: Maximum total tokens to process across all chunks (default 1024)
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.stride = chunk_size - chunk_overlap

    def chunk_token_ids(self, token_ids: list[int]) -> list[list[int]]:
        """
        Split a sequence of token IDs into overlapping chunks.

        Args:
            token_ids: List of integer token IDs

        Returns:
            List of chunked token ID lists
        """
        if not token_ids:
            return []

        # Bound by total max_tokens budget
        bounded_ids = token_ids[:self.max_tokens]

        if len(bounded_ids) <= self.chunk_size:
            return [bounded_ids]

        chunks: list[list[int]] = []
        start = 0
        total_len = len(bounded_ids)

        while start < total_len:
            end = min(start + self.chunk_size, total_len)
            chunk = bounded_ids[start:end]
            chunks.append(chunk)

            if end >= total_len:
                break

            start += self.stride

        return chunks

    def chunk_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pad_token_id: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split 1D or 2D token tensors into a batch of padded chunk tensors.

        Args:
            input_ids: Tensor of shape (seq_len,) or (1, seq_len)
            attention_mask: Optional mask tensor of matching shape
            pad_token_id: Integer padding token ID (default 0)

        Returns:
            chunked_input_ids: Tensor of shape (num_chunks, chunk_size)
            chunked_attention_mask: Tensor of shape (num_chunks, chunk_size)
        """
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask.squeeze(0)

        device = input_ids.device
        seq_len = input_ids.size(0)

        # Truncate to maximum overall token budget
        effective_len = min(seq_len, self.max_tokens)
        active_ids = input_ids[:effective_len]

        if seq_len == 0 or effective_len == 0:
            # Return single empty padded chunk
            c_ids = torch.full((1, self.chunk_size), pad_token_id, dtype=torch.long, device=device)
            c_mask = torch.zeros((1, self.chunk_size), dtype=torch.long, device=device)
            return c_ids, c_mask

        # If fits in one chunk
        if effective_len <= self.chunk_size:
            c_ids = torch.full((1, self.chunk_size), pad_token_id, dtype=torch.long, device=device)
            c_mask = torch.zeros((1, self.chunk_size), dtype=torch.long, device=device)
            c_ids[0, :effective_len] = active_ids
            if attention_mask is not None:
                c_mask[0, :effective_len] = attention_mask[:effective_len]
            else:
                c_mask[0, :effective_len] = 1
            return c_ids, c_mask

        chunk_list_ids: list[torch.Tensor] = []
        chunk_list_masks: list[torch.Tensor] = []

        start = 0
        while start < effective_len:
            end = min(start + self.chunk_size, effective_len)
            length = end - start

            p_ids = torch.full((self.chunk_size,), pad_token_id, dtype=torch.long, device=device)
            p_mask = torch.zeros((self.chunk_size,), dtype=torch.long, device=device)

            p_ids[:length] = active_ids[start:end]
            if attention_mask is not None:
                p_mask[:length] = attention_mask[start:end]
            else:
                p_mask[:length] = 1

            chunk_list_ids.append(p_ids)
            chunk_list_masks.append(p_mask)

            if end >= effective_len:
                break
            start += self.stride

        chunked_input_ids = torch.stack(chunk_list_ids, dim=0)
        chunked_attention_mask = torch.stack(chunk_list_masks, dim=0)

        return chunked_input_ids, chunked_attention_mask

    def chunk_text(
        self,
        text: str,
        tokenizer: Any,
        add_special_tokens: bool = True
    ) -> dict[str, Any]:
        """
        Tokenize raw text without truncation and split into overlapping chunks,
        ensuring proper [CLS] and [SEP] wrapping per chunk.

        Args:
            text: Text string to chunk
            tokenizer: HuggingFace PreTrainedTokenizer (e.g. BertTokenizer)
            add_special_tokens: Whether to insert [CLS] and [SEP] tokens

        Returns:
            Dictionary containing:
                - 'input_ids': Tensor (num_chunks, chunk_size)
                - 'attention_mask': Tensor (num_chunks, chunk_size)
                - 'num_chunks': int
                - 'total_tokens': int
        """
        # Tokenize without truncation
        tokens = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_tensors=None
        )
        raw_ids = tokens.get("input_ids", [])
        if isinstance(raw_ids, torch.Tensor):
            raw_ids = raw_ids.flatten().tolist()
        elif isinstance(raw_ids, list):
            if raw_ids and isinstance(raw_ids[0], (list, torch.Tensor)):
                flattened = []
                for item in raw_ids:
                    if isinstance(item, torch.Tensor):
                        flattened.extend(item.flatten().tolist())
                    elif isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(int(item))
                raw_ids = flattened
            else:
                raw_ids = [int(x) for x in raw_ids]
        total_tokens = len(raw_ids)

        if total_tokens == 0:
            # Handle empty text cleanly
            cls_id = tokenizer.cls_token_id or 101
            sep_id = tokenizer.sep_token_id or 102
            pad_id = tokenizer.pad_token_id or 0

            empty_ids = torch.full((1, self.chunk_size), pad_id, dtype=torch.long)
            empty_mask = torch.zeros((1, self.chunk_size), dtype=torch.long)
            if add_special_tokens:
                empty_ids[0, 0] = cls_id
                empty_ids[0, 1] = sep_id
                empty_mask[0, 0] = 1
                empty_mask[0, 1] = 1

            return {
                "input_ids": empty_ids,
                "attention_mask": empty_mask,
                "num_chunks": 1,
                "total_tokens": 0,
            }

        # Truncate to maximum overall budget
        bounded_ids = raw_ids[:self.max_tokens]

        # Effective content size per chunk (reserving 2 tokens for CLS and SEP if enabled)
        content_capacity = (self.chunk_size - 2) if add_special_tokens else self.chunk_size
        stride = content_capacity - self.chunk_overlap
        if stride <= 0:
            stride = max(1, content_capacity // 2)

        cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 101
        sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 102
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        chunks_ids: list[torch.Tensor] = []
        chunks_masks: list[torch.Tensor] = []

        start = 0
        total_len = len(bounded_ids)

        while start < total_len:
            end = min(start + content_capacity, total_len)
            sub_ids = bounded_ids[start:end]

            chunk_ids = torch.full((self.chunk_size,), pad_id, dtype=torch.long)
            chunk_mask = torch.zeros((self.chunk_size,), dtype=torch.long)

            if add_special_tokens:
                full_seq = [cls_id] + sub_ids + [sep_id]
            else:
                full_seq = sub_ids

            seq_len = len(full_seq)
            chunk_ids[:seq_len] = torch.tensor(full_seq, dtype=torch.long)
            chunk_mask[:seq_len] = 1

            chunks_ids.append(chunk_ids)
            chunks_masks.append(chunk_mask)

            if end >= total_len:
                break
            start += stride

        return {
            "input_ids": torch.stack(chunks_ids, dim=0),
            "attention_mask": torch.stack(chunks_masks, dim=0),
            "num_chunks": len(chunks_ids),
            "total_tokens": total_tokens,
        }

    def pool_chunk_embeddings(
        self,
        chunk_embeddings: torch.Tensor,
        method: str = "mean",
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool chunk-level embeddings (num_chunks, feature_dim) into a single document
        representation (1, feature_dim).

        Args:
            chunk_embeddings: Tensor of shape (num_chunks, feature_dim)
            method: Pooling method ('mean', 'weighted', 'decay')
            weights: Optional tensor of weights (num_chunks,) for weighted pooling

        Returns:
            Document embedding of shape (1, feature_dim)
        """
        if chunk_embeddings.dim() == 1:
            chunk_embeddings = chunk_embeddings.unsqueeze(0)

        num_chunks, dim = chunk_embeddings.shape

        if num_chunks == 1:
            return torch.nan_to_num(chunk_embeddings, nan=0.0)

        if method == "mean":
            pooled = chunk_embeddings.mean(dim=0, keepdim=True)

        elif method == "weighted":
            if weights is None:
                pooled = chunk_embeddings.mean(dim=0, keepdim=True)
            else:
                w = weights.view(num_chunks, 1).to(chunk_embeddings.device).float()
                w_norm = w / (w.sum() + 1e-8)
                pooled = (chunk_embeddings * w_norm).sum(dim=0, keepdim=True)

        elif method == "decay":
            # Exponential decay: earlier chunks (title/description/early transcript) carry higher weight
            indices = torch.arange(num_chunks, dtype=torch.float, device=chunk_embeddings.device)
            decay_weights = torch.exp(-0.15 * indices).unsqueeze(1)
            decay_weights = decay_weights / decay_weights.sum()
            pooled = (chunk_embeddings * decay_weights).sum(dim=0, keepdim=True)

        else:
            pooled = chunk_embeddings.mean(dim=0, keepdim=True)

        return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
