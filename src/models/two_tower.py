"""
Two-Tower Recommendation Model

The two-tower architecture is standard for large-scale recommendations.
Each tower produces an embedding, and similarity between embeddings
determines recommendations.

For content-based recommendations:
- Both towers are identical (shared weights)
- We learn embeddings where similar videos are close together

For user-video recommendations:
- User tower: encodes user preferences
- Video tower: encodes video content
- We learn embeddings where users are close to videos they'd like
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from features.video_encoder import VideoEncoder
from features.audio_encoder import AudioEncoder
from features.text_encoder import TextEncoder
from models.fusion import GatedFusion


class VideoTower(nn.Module):
    """
    Encodes a video into a single embedding using all modalities.

    This is the "tower" that processes video content:
    Video frames → VideoEncoder ─┐
    Audio waveform → AudioEncoder ├→ Fusion → Final Embedding
    Text metadata → TextEncoder ──┘

    Args:
        feature_dim: Dimension of final embedding (512)
        freeze_encoders: Freeze pre-trained backbones (recommended)
        fusion_type: Which fusion strategy ('gated', 'concat', 'attention')
        pretrained: Use pretrained weights for encoders
    """

    def __init__(
        self,
        feature_dim: int = 512,
        freeze_encoders: bool = True,
        fusion_type: str = "gated",
        pretrained: bool = True,
        video_backbone: str = "resnet50",
        audio_model: str = "facebook/wav2vec2-base-960h",
        text_model: str = "bert-base-uncased"
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.pretrained = pretrained

        # Initialize encoders
        self.video_encoder = VideoEncoder(
            backbone=video_backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_encoders
        )

        self.audio_encoder = AudioEncoder(
            model_name=audio_model,
            feature_dim=feature_dim,
            freeze_backbone=freeze_encoders,
            pretrained=pretrained
        )

        self.text_encoder = TextEncoder(
            model_name=text_model,
            feature_dim=feature_dim,
            freeze_backbone=freeze_encoders,
            pretrained=pretrained
        )

        # Fusion layer
        if fusion_type == "gated":
            self.fusion = GatedFusion(
                input_dim=feature_dim,
                output_dim=feature_dim
            )
        elif fusion_type == "concat":
            from models.fusion import ConcatFusion
            self.fusion = ConcatFusion(
                input_dim=feature_dim,
                output_dim=feature_dim
            )
        elif fusion_type == "attention":
            from models.fusion import AttentionFusion
            self.fusion = AttentionFusion(
                input_dim=feature_dim,
                output_dim=feature_dim
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Final normalization (important for similarity computation)
        self.normalize = True

    def forward(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        text: list[str] = None,
        text_ids: torch.Tensor = None,
        text_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Encode video into embedding.

        Args:
            frames: (batch, num_frames, 3, H, W) video frames
            audio: (batch, num_samples) audio waveform
            text: List of strings (titles) OR
            text_ids: (batch, seq_len) pre-tokenized text
            text_mask: (batch, seq_len) attention mask

        Returns:
            embedding: (batch, feature_dim) normalized video embedding
        """
        # Encode each modality
        video_emb = self.video_encoder(frames)        # (batch, 512)
        audio_emb = self.audio_encoder(audio)         # (batch, 512)

        # Text can be raw strings or pre-tokenized
        if text is not None:
            text_emb = self.text_encoder(texts=text)  # (batch, 512)
        else:
            text_emb = self.text_encoder(
                input_ids=text_ids,
                attention_mask=text_mask
            )

        # Fuse modalities
        fused = self.fusion(video_emb, audio_emb, text_emb)  # (batch, 512)

        # L2 normalize for cosine similarity
        if self.normalize:
            fused = F.normalize(fused, p=2, dim=1)

        return fused


class TwoTowerModel(nn.Module):
    """
    Two-Tower model for video recommendations.

    For content-based (video-to-video) recommendations:
    - Both towers share weights
    - Similar videos should have similar embeddings

    For user-to-video recommendations:
    - User tower encodes user preferences
    - Video tower encodes video content
    - (User tower not implemented here - would need user data)

    Args:
        feature_dim: Embedding dimension (512)
        share_towers: Whether towers share weights (True for content-based)
        temperature: Scaling factor for similarity scores
    """

    def __init__(
        self,
        feature_dim: int = 512,
        share_towers: bool = True,
        temperature: float = 0.07,
        freeze_encoders: bool = True,
        fusion_type: str = "gated",
        pretrained: bool = True,
        video_backbone: str = "resnet50",
        audio_model: str = "facebook/wav2vec2-base-960h",
        text_model: str = "bert-base-uncased"
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.temperature = temperature
        self.share_towers = share_towers
        self.fusion_type = fusion_type
        self.pretrained = pretrained

        # Primary tower (video content encoder)
        self.video_tower = VideoTower(
            feature_dim=feature_dim,
            freeze_encoders=freeze_encoders,
            fusion_type=fusion_type,
            pretrained=pretrained,
            video_backbone=video_backbone,
            audio_model=audio_model,
            text_model=text_model
        )

        # For shared towers, reuse the same network
        # For separate towers, create another (e.g., for user-video)
        if share_towers:
            self.query_tower = self.video_tower
        else:
            self.query_tower = VideoTower(
                feature_dim=feature_dim,
                freeze_encoders=freeze_encoders,
                fusion_type=fusion_type,
                pretrained=pretrained,
                video_backbone=video_backbone,
                audio_model=audio_model,
                text_model=text_model
            )

    def encode_video(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        text: list[str] = None
    ) -> torch.Tensor:
        """
        Encode a video into an embedding.

        Use this for:
        - Building a video index
        - Getting embeddings for storage in database
        """
        return self.video_tower(frames, audio, text)

    def forward(
        self,
        # Anchor video (the video we're finding recommendations for)
        anchor_frames: torch.Tensor,
        anchor_audio: torch.Tensor,
        anchor_text: list[str],
        # Positive video (similar/related video)
        positive_frames: torch.Tensor,
        positive_audio: torch.Tensor,
        positive_text: list[str],
        # Negative videos (random/unrelated videos) - optional
        negative_frames: torch.Tensor = None,
        negative_audio: torch.Tensor = None,
        negative_text: list[str] = None
    ) -> dict:
        """
        Forward pass for training.

        Args:
            anchor_*: The query video
            positive_*: A similar video (should be close)
            negative_*: Dissimilar videos (should be far)

        Returns:
            Dictionary with embeddings and similarity scores
        """
        # Encode anchor
        anchor_emb = self.video_tower(
            anchor_frames, anchor_audio, anchor_text
        )

        # Encode positive
        positive_emb = self.query_tower(
            positive_frames, positive_audio, positive_text
        )

        # Compute positive similarity (should be high)
        # Dot product of normalized vectors = cosine similarity
        positive_sim = (anchor_emb * positive_emb).sum(dim=1) / self.temperature

        result = {
            "anchor_emb": anchor_emb,
            "positive_emb": positive_emb,
            "positive_sim": positive_sim
        }

        # Encode negatives if provided
        if negative_frames is not None:
            negative_emb = self.query_tower(
                negative_frames, negative_audio, negative_text
            )
            # Negative similarity (should be low)
            negative_sim = (anchor_emb * negative_emb).sum(dim=1) / self.temperature

            result["negative_emb"] = negative_emb
            result["negative_sim"] = negative_sim

        return result

    def get_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between two embeddings.

        Args:
            emb1: (batch, 512) or (512,)
            emb2: (batch, 512) or (512,)

        Returns:
            similarity: Cosine similarity score(s)
        """
        # Ensure normalized
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)

        # Dot product = cosine similarity for normalized vectors
        return (emb1 * emb2).sum(dim=-1)

    def find_similar(
        self,
        query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        top_k: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find most similar videos to a query.

        Args:
            query_emb: (512,) query video embedding
            candidate_embs: (N, 512) all candidate video embeddings
            top_k: Number of results to return

        Returns:
            indices: (top_k,) indices of most similar videos
            scores: (top_k,) similarity scores
        """
        # Compute all similarities at once
        # (1, 512) @ (512, N) = (1, N)
        similarities = query_emb @ candidate_embs.T

        # Get top-k
        scores, indices = torch.topk(similarities, k=top_k)

        return indices, scores


# Test code
if __name__ == "__main__":
    print("Testing Two-Tower Model...")
    print("(This will download models on first run)")

    # Create model
    model = TwoTowerModel(
        feature_dim=512,
        share_towers=True,
        temperature=0.07
    )

    # Dummy data
    batch_size = 2
    frames = torch.randn(batch_size, 8, 3, 224, 224)
    audio = torch.randn(batch_size, 16000)
    text = ["Cooking tutorial pasta", "Funny cat compilation"]

    # Test encoding
    print("\n1. Testing video encoding:")
    embeddings = model.encode_video(frames, audio, text)
    print(f"   Input: frames {frames.shape}, audio {audio.shape}, text {len(text)} strings")
    print(f"   Output: {embeddings.shape}")  # (2, 512)
    print(f"   Normalized: {embeddings.norm(dim=1)}")  # Should be ~1.0

    # Test similarity
    print("\n2. Testing similarity:")
    sim = model.get_similarity(embeddings[0], embeddings[1])
    print(f"   Similarity between video 0 and 1: {sim.item():.4f}")

    # Test find_similar
    print("\n3. Testing find_similar:")
    query = embeddings[0]
    candidates = torch.randn(100, 512)  # 100 random videos
    candidates = F.normalize(candidates, p=2, dim=1)
    indices, scores = model.find_similar(query, candidates, top_k=5)
    print(f"   Top 5 similar videos: {indices.tolist()}")
    print(f"   Scores: {scores.tolist()}")

    print("\n[OK] Two-Tower Model working!")
