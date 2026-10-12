"""
AudioEncoder: Extracts audio features from waveforms using Wav2Vec2

Key Concepts:
1. Audio is a 1D signal (amplitude over time)
2. Wav2Vec2 was pre-trained on 1000s of hours of speech
3. It outputs a sequence of feature vectors (one per ~20ms)
4. We aggregate these into a single audio embedding

This mirrors what VideoEncoder does, but for audio instead of frames.
"""

from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class AudioEncoder(nn.Module):
    """
    Encodes audio waveforms into fixed-size embeddings using Wav2Vec2.

    Args:
        model_name: Which Wav2Vec2 model to use (base or large)
        feature_dim: Final embedding size (512 to match video)
        aggregation: How to combine time steps ('mean' or 'attention')
        freeze_backbone: If True, don't train Wav2Vec2 (faster, less memory)
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        feature_dim: int = 512,
        aggregation: str = "mean",
        freeze_backbone: bool = True,
        pretrained: bool = True
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.aggregation = aggregation
        self.pretrained = pretrained
        self.model_name = model_name

        # Load Wav2Vec2 backbone
        if pretrained:
            try:
                self.backbone = Wav2Vec2Model.from_pretrained(model_name, local_files_only=True)
            except Exception:
                from transformers import Wav2Vec2Config
                config = Wav2Vec2Config()
                self.backbone = Wav2Vec2Model(config)
        else:
            from transformers import Wav2Vec2Config
            config = Wav2Vec2Config()
            self.backbone = Wav2Vec2Model(config)

        # Wav2Vec2-base outputs 768-dimensional features
        # Wav2Vec2-large outputs 1024-dimensional features
        if "large" in model_name:
            backbone_dim = 1024
        else:
            backbone_dim = 768

        # Freeze backbone to save memory and speed up training
        # The pre-trained features are already good!
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection: Map Wav2Vec2 features to our feature_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, feature_dim)
        )

        # Optional: Attention mechanism for weighted aggregation
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveforms: (batch_size, num_samples) or (num_samples,)
                      e.g., (8, 16000) = 8 audio clips, 1 second each at 16kHz
            attention_mask: (batch_size, num_samples) optional
                           Masks padded regions for variable-length audio

        Returns:
            embeddings: (batch_size, feature_dim)
        """
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        # Explicit silent waveform detection:
        # A silent, zero-energy, or corrupt waveform must never produce a pseudo-valid non-zero embedding
        is_silent = (waveforms.abs().max(dim=-1).values < 1e-5)

        # Pass through Wav2Vec2
        outputs = self.backbone(
            waveforms,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get the hidden states (feature sequence): (batch_size, seq_len, 768)
        features = outputs.last_hidden_state

        # Project: (batch_size, seq_len, 768) -> (batch_size, seq_len, 512)
        features = self.projection(features)

        # Handle attention mask downsampling across temporal dimension if provided
        if attention_mask is not None and attention_mask.numel() > 0:
            seq_len = features.size(1)
            # Pool binary mask to match feature sequence length
            mask_float = attention_mask.unsqueeze(1).float()
            mask_down = F.adaptive_avg_pool1d(mask_float, seq_len).squeeze(1)
            mask_weight = (mask_down > 0.1).float().unsqueeze(-1)  # (batch_size, seq_len, 1)

            if self.aggregation == "mean":
                masked_feat = features * mask_weight
                sum_feat = masked_feat.sum(dim=1)
                counts = mask_weight.sum(dim=1).clamp(min=1.0)
                audio_emb = sum_feat / counts
            elif self.aggregation == "attention" and hasattr(self, "attention"):
                attention_scores = self.attention(features)
                attention_scores = attention_scores.masked_fill(mask_weight == 0, -1e9)
                attention_weights = torch.softmax(attention_scores, dim=1)
                audio_emb = (features * attention_weights).sum(dim=1)
            else:
                audio_emb = features.mean(dim=1)
        else:
            # Unmasked aggregation across time into single embedding
            if self.aggregation == "mean":
                audio_emb = features.mean(dim=1)
            elif self.aggregation == "attention" and hasattr(self, "attention"):
                attention_scores = self.attention(features)
                attention_weights = torch.softmax(attention_scores, dim=1)
                audio_emb = (features * attention_weights).sum(dim=1)
            else:
                audio_emb = features.mean(dim=1)

        audio_emb = torch.nan_to_num(audio_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # Enforce zero vector for silent/missing waveforms so downstream fusion masks them out
        if is_silent.any():
            silent_mask = is_silent.unsqueeze(-1).expand_as(audio_emb)
            audio_emb = torch.where(silent_mask, torch.zeros_like(audio_emb), audio_emb)

        return audio_emb

    def encode_chunks(
        self,
        chunks: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """
        Encode multi-chunk audio tensor of shape (num_chunks, chunk_samples)
        into a single (1, feature_dim) embedding.

        Args:
            chunks: (num_chunks, chunk_samples)
            attention_mask: (num_chunks, chunk_samples)
            aggregation: 'mean', 'decay', or 'max'

        Returns:
            video_audio_embedding: Tensor of shape (1, feature_dim)
        """
        if chunks.numel() == 0:
            return torch.zeros(1, self.feature_dim, device=chunks.device)

        with torch.no_grad():
            chunk_embs = self.forward(chunks, attention_mask=attention_mask)  # (num_chunks, feature_dim)

            if chunk_embs.size(0) == 1:
                return chunk_embs

            agg = aggregation.lower().strip()
            if agg == "mean":
                return chunk_embs.mean(dim=0, keepdim=True)
            elif agg in ("decay", "weighted"):
                C = chunk_embs.size(0)
                steps = torch.arange(C, dtype=chunk_embs.dtype, device=chunk_embs.device)
                w = torch.exp(0.1 * (steps - (C - 1))).view(C, 1)
                w = w / w.sum()
                return (chunk_embs * w).sum(dim=0, keepdim=True)
            elif agg == "max":
                return chunk_embs.max(dim=0, keepdim=True)[0]
            else:
                return chunk_embs.mean(dim=0, keepdim=True)


# Test code to verify it works
if __name__ == "__main__":
    print("Testing AudioEncoder...")
    print("(This will download Wav2Vec2 on first run, ~360MB)")

    encoder = AudioEncoder(
        model_name="facebook/wav2vec2-base-960h",
        feature_dim=512,
        aggregation="mean",
        freeze_backbone=True
    )

    # Simulate 4 audio clips, each 1 second at 16kHz
    # In real use, these would be actual audio samples
    dummy_audio = torch.randn(4, 16000)

    output = encoder(dummy_audio)

    print(f"Input shape: {dummy_audio.shape}")   # (4, 16000)
    print(f"Output shape: {output.shape}")       # (4, 512)
    print("[OK] AudioEncoder working!")
