"""
VideoEncoder: Extracts visual features from video frames using CNNs

Key Concepts:
1. We sample N frames from the video (e.g., 8 frames)
2. Each frame goes through a pre-trained CNN (ResNet, EfficientNet)
3. We get a feature vector for each frame
4. Aggregate all frame features into one video embedding (mean/max/attention)

This is standard for video understanding - treating video as a sequence of images.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VideoEncoder(nn.Module):
    """
    Encodes video into a fixed-size embedding using CNN + aggregation.

    Args:
        backbone: Which CNN to use ('resnet50', 'resnet18', etc.)
        pretrained: Use ImageNet pre-trained weights (usually better)
        feature_dim: Final embedding size (512 is common)
        aggregation: How to combine frame features ('mean' is simplest, 'attention' is best)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        feature_dim: int = 512,
        aggregation: str = 'mean',
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.aggregation = aggregation

        # Load CNN backbone (removes final classification layer)
        # Using new weights API (PyTorch 2.0+)
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet50(weights=weights)
            # ResNet outputs 2048-d features before the FC layer
            backbone_dim = 2048
            # Remove the final FC layer - we only want features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet18(weights=weights)
            backbone_dim = 512
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        else:
            raise ValueError(f"Backbone {backbone} not supported")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection: Map CNN features (2048-d for ResNet50) to our feature_dim
        # We add a hidden layer for more capacity
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, feature_dim)
        )

        # Optional: Attention mechanism to weight important frames
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (batch_size, num_frames, 3, H, W)
                   e.g., (32, 8, 3, 224, 224) = 32 videos, 8 frames each

        Returns:
            embeddings: (batch_size, feature_dim)
        """
        B, T, C, H, W = frames.shape  # B=batch, T=num_frames

        # Reshape to process all frames together
        # (B, T, C, H, W) -> (B*T, C, H, W)
        frames_flat = frames.view(B * T, C, H, W)

        # Pass through CNN
        # Output: (B*T, 2048, 1, 1) for ResNet50
        features = self.backbone(frames_flat)

        # Flatten: (B*T, 2048, 1, 1) -> (B*T, 2048)
        features = features.view(B * T, -1)

        # Project: (B*T, 2048) -> (B*T, feature_dim)
        features = self.projection(features)

        # Reshape back to separate frames: (B*T, feature_dim) -> (B, T, feature_dim)
        features = features.view(B, T, self.feature_dim)

        # Aggregate frames into single video embedding
        if self.aggregation == 'mean':
            # Simple average - works well in practice
            video_emb = features.mean(dim=1)  # (B, feature_dim)

        elif self.aggregation == 'max':
            # Max pooling - captures most prominent features
            video_emb = features.max(dim=1)[0]  # (B, feature_dim)

        elif self.aggregation == 'attention':
            # Learn which frames are important
            # attention_scores: (B, T, 1)
            attention_scores = self.attention(features)
            attention_weights = torch.softmax(attention_scores, dim=1)
            # Weighted sum: (B, T, feature_dim) * (B, T, 1) -> sum -> (B, feature_dim)
            video_emb = (features * attention_weights).sum(dim=1)

        return video_emb


# Test code to verify it works
if __name__ == "__main__":
    print("Testing VideoEncoder...")

    encoder = VideoEncoder(backbone="resnet50", feature_dim=512, aggregation='mean')

    # Simulate 4 videos, each with 8 frames of 224x224
    dummy_videos = torch.randn(4, 8, 3, 224, 224)

    output = encoder(dummy_videos)

    print(f"Input shape: {dummy_videos.shape}")  # (4, 8, 3, 224, 224)
    print(f"Output shape: {output.shape}")  # (4, 512)
    print("[OK] VideoEncoder working!")
