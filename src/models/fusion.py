"""
Multimodal Fusion: Combine video, audio, and text embeddings into one.

This is where the "multimodal" magic happens - we take separate understanding
of what a video looks like, sounds like, and is about, and merge them into
a single representation.
"""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion.

    Concatenates all modalities and projects to output dimension.
    Simple but effective baseline.

    Args:
        input_dim: Dimension of each modality (512)
        output_dim: Final embedding dimension (512)
        num_modalities: Number of modalities to fuse (3: video, audio, text)
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        num_modalities: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        concat_dim = input_dim * num_modalities  # 512 * 3 = 1536

        # MLP to project concatenated features
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(concat_dim, output_dim)
        )

    def forward(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_emb: (batch, 512) from VideoEncoder
            audio_emb: (batch, 512) from AudioEncoder
            text_emb:  (batch, 512) from TextEncoder

        Returns:
            fused: (batch, 512) combined embedding
        """
        # Concatenate: (batch, 1536)
        concat = torch.cat([video_emb, audio_emb, text_emb], dim=1)
        concat = torch.nan_to_num(concat, nan=0.0)

        # Project: (batch, 512)
        fused = self.fusion(concat)

        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion learns importance weights for each modality.

    For each video, it learns:
    - How important is the visual content?
    - How important is the audio?
    - How important is the text metadata?

    A music video might weight audio highly.
    A tutorial might weight text highly.

    Args:
        input_dim: Dimension of each modality (512)
        output_dim: Final embedding dimension (512)
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim

        # Gate network: learns importance of each modality
        # Input: all three modalities concatenated
        # Output: 3 weights (one per modality)
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # 3 weights for 3 modalities
        )

        # Optional: project to output dimension if different
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_emb: (batch, 512)
            audio_emb: (batch, 512)
            text_emb:  (batch, 512)

        Returns:
            fused: (batch, output_dim)
        """
        # Clean any accidental NaNs
        video_emb = torch.nan_to_num(video_emb, nan=0.0)
        audio_emb = torch.nan_to_num(audio_emb, nan=0.0)
        text_emb = torch.nan_to_num(text_emb, nan=0.0)

        # Stack modalities: (batch, 3, 512)
        stacked = torch.stack([video_emb, audio_emb, text_emb], dim=1)

        # Detect missing modalities (norm < 1e-6)
        modality_norms = torch.norm(stacked, p=2, dim=-1)
        is_missing = modality_norms < 1e-6

        # Concatenate for gate input: (batch, 1536)
        concat = torch.cat([video_emb, audio_emb, text_emb], dim=1)

        # Compute gate logits: (batch, 3)
        gate_logits = self.gate(concat)

        # If some (but not all) modalities are missing, mask out missing ones
        any_present = (~is_missing).any(dim=1, keepdim=True)
        masked_logits = torch.where(
            is_missing & any_present,
            torch.tensor(-1e9, device=gate_logits.device, dtype=gate_logits.dtype),
            gate_logits
        )

        gate_weights = torch.softmax(masked_logits, dim=1)  # Sum to 1

        # Expand for broadcasting: (batch, 3, 1)
        gate_weights = gate_weights.unsqueeze(-1)

        # Weighted sum: (batch, 3, 512) * (batch, 3, 1) -> sum -> (batch, 512)
        fused = (stacked * gate_weights).sum(dim=1)

        # Project to output dimension
        fused = self.projection(fused)

        return fused

    def get_gate_weights(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the learned importance weights (useful for interpretability).

        Returns:
            weights: (batch, 3) - [video_weight, audio_weight, text_weight]
        """
        video_emb = torch.nan_to_num(video_emb, nan=0.0)
        audio_emb = torch.nan_to_num(audio_emb, nan=0.0)
        text_emb = torch.nan_to_num(text_emb, nan=0.0)

        stacked = torch.stack([video_emb, audio_emb, text_emb], dim=1)
        modality_norms = torch.norm(stacked, p=2, dim=-1)
        is_missing = modality_norms < 1e-6

        concat = torch.cat([video_emb, audio_emb, text_emb], dim=1)
        gate_logits = self.gate(concat)

        any_present = (~is_missing).any(dim=1, keepdim=True)
        masked_logits = torch.where(
            is_missing & any_present,
            torch.tensor(-1e9, device=gate_logits.device, dtype=gate_logits.dtype),
            gate_logits
        )

        return torch.softmax(masked_logits, dim=1)


class AttentionFusion(nn.Module):
    """
    Cross-modal attention fusion.

    Each modality attends to the others to capture relationships:
    - Video attends to audio: "What sounds match these visuals?"
    - Audio attends to text: "What words describe these sounds?"

    Most powerful but most complex.

    Args:
        input_dim: Dimension of each modality (512)
        output_dim: Final embedding dimension (512)
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention across modalities
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(input_dim)

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_emb: (batch, 512)
            audio_emb: (batch, 512)
            text_emb:  (batch, 512)

        Returns:
            fused: (batch, output_dim)
        """
        video_emb = torch.nan_to_num(video_emb, nan=0.0)
        audio_emb = torch.nan_to_num(audio_emb, nan=0.0)
        text_emb = torch.nan_to_num(text_emb, nan=0.0)

        # Stack as sequence: (batch, 3, 512) - 3 "tokens" (one per modality)
        stacked = torch.stack([video_emb, audio_emb, text_emb], dim=1)

        modality_norms = torch.norm(stacked, p=2, dim=-1)
        is_missing = modality_norms < 1e-6
        all_missing = is_missing.all(dim=1, keepdim=True)
        key_padding_mask = is_missing & (~all_missing)

        # Self-attention: each modality attends to others
        attended, _ = self.attention(
            stacked, stacked, stacked,
            key_padding_mask=key_padding_mask if key_padding_mask.any() else None
        )

        # Residual connection + norm
        attended = self.norm(stacked + attended)

        # Flatten and project: (batch, 1536) -> (batch, 512)
        flat = attended.view(attended.size(0), -1)
        fused = self.projection(flat)

        return fused


# Test code
if __name__ == "__main__":
    print("Testing Fusion modules...")

    batch_size = 4
    dim = 512

    # Dummy embeddings (as if from encoders)
    video_emb = torch.randn(batch_size, dim)
    audio_emb = torch.randn(batch_size, dim)
    text_emb = torch.randn(batch_size, dim)

    # Test ConcatFusion
    print("\n1. ConcatFusion:")
    concat_fusion = ConcatFusion(input_dim=dim, output_dim=dim)
    output = concat_fusion(video_emb, audio_emb, text_emb)
    print(f"   Input: 3 × (4, 512)")
    print(f"   Output: {output.shape}")  # (4, 512)

    # Test GatedFusion
    print("\n2. GatedFusion:")
    gated_fusion = GatedFusion(input_dim=dim, output_dim=dim)
    output = gated_fusion(video_emb, audio_emb, text_emb)
    weights = gated_fusion.get_gate_weights(video_emb, audio_emb, text_emb)
    print(f"   Input: 3 × (4, 512)")
    print(f"   Output: {output.shape}")  # (4, 512)
    print(f"   Gate weights (first sample): {weights[0].tolist()}")
    print(f"   [video, audio, text] weights sum to: {weights[0].sum():.2f}")

    # Test AttentionFusion
    print("\n3. AttentionFusion:")
    attn_fusion = AttentionFusion(input_dim=dim, output_dim=dim)
    output = attn_fusion(video_emb, audio_emb, text_emb)
    print(f"   Input: 3 × (4, 512)")
    print(f"   Output: {output.shape}")  # (4, 512)

    print("\n[OK] All fusion modules working!")
