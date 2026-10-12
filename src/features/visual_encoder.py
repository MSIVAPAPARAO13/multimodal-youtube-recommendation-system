"""
VisualEncoder: Modular Visual Representation Architecture.

Supports:
1. ResNet50VisualEncoder: The frozen 2048-D -> 512-D production baseline.
2. CLIPExperimentalEncoder: Opt-in experimental visual transformer.
3. Multi-frame aggregation: Mean pooling, decay/weighted pooling, max pooling.
4. Factory abstraction: get_visual_encoder()
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


class BaseVisualEncoder(nn.Module, ABC):
    """Abstract base class for visual encoders."""

    def __init__(self, feature_dim: int = 512, aggregation: str = "mean"):
        super().__init__()
        self.feature_dim = feature_dim
        self.aggregation = aggregation.lower()

    @abstractmethod
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames.
        Input shape: (B, T, 3, H, W) or (T, 3, H, W)
        Output shape: (B, feature_dim) or (1, feature_dim)
        """
        pass

    @abstractmethod
    def encode_single_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image or thumbnail.
        Input shape: (1, 3, H, W) or (B, 3, H, W)
        Output shape: (1, feature_dim) or (B, feature_dim)
        """
        pass

    def aggregate_frames(
        self,
        frame_features: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Aggregate frame embeddings into a video-level representation.

        Args:
            frame_features: Tensor of shape (B, T, feature_dim)
            method: 'mean', 'decay', 'weighted', or 'max'

        Returns:
            video_embedding: Tensor of shape (B, feature_dim)
        """
        pooling = (method or self.aggregation).lower()
        B, T, D = frame_features.shape

        if T == 1:
            return frame_features.squeeze(1)

        if pooling == "mean":
            return frame_features.mean(dim=1)

        elif pooling in ("decay", "weighted"):
            # Temporal exponential decay weighting: more recent frames receive higher weights
            # w_t = exp(0.2 * (t - (T - 1)))
            steps = torch.arange(T, dtype=frame_features.dtype, device=frame_features.device)
            decay_weights = torch.exp(0.2 * (steps - (T - 1)))
            decay_weights = decay_weights / decay_weights.sum()  # Normalize sum to 1.0
            decay_weights = decay_weights.view(1, T, 1)  # (1, T, 1)
            return (frame_features * decay_weights).sum(dim=1)

        elif pooling == "max":
            return frame_features.max(dim=1)[0]

        else:
            logger.warning("Unknown pooling method '%s'; defaulting to mean pooling.", pooling)
            return frame_features.mean(dim=1)


class ResNet50VisualEncoder(BaseVisualEncoder):
    """
    Production Visual Baseline using torchvision ResNet-50.

    Pipeline:
        Sampled RGB Frames (B, T, 3, 224, 224)
            ↓
        ResNet-50 Backbone (Frozen ImageNet-1K)
            ↓
        2048-D Feature Map
            ↓
        512-D Non-linear Projection
            ↓
        Multi-Frame Aggregation (Mean / Weighted Decay)
            ↓
        512-D Video-level Visual Embedding
    """

    def __init__(
        self,
        feature_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        aggregation: str = "mean",
    ):
        super().__init__(feature_dim=feature_dim, aggregation=aggregation)

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet50(weights=weights)

        # 2048-d feature extractor before the classification FC layer
        self.backbone_dim = 2048
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection: Map 2048-d CNN features to 512-d shared multimodal space
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.backbone_dim, feature_dim),
        )

        self.eval()

    def extract_backbone_features(self, frames_flat: torch.Tensor) -> torch.Tensor:
        """
        Extract raw 2048-D features from ResNet-50 backbone.
        Input: (N, 3, H, W)
        Output: (N, 2048)
        """
        feat = self.backbone(frames_flat)  # (N, 2048, 1, 1)
        return feat.view(feat.size(0), -1)

    def encode_single_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Pass a single image / thumbnail through ResNet-50 and project to 512-D.
        Input: (1, 3, H, W) or (B, 3, H, W)
        Output: (B, feature_dim)
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            feat_2048 = self.extract_backbone_features(image_tensor)
            proj_512 = self.projection(feat_2048)
            return proj_512

    def forward(
        self,
        frames: torch.Tensor,
        aggregation_method: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W) or (T, 3, H, W)
            aggregation_method: Optional override of pooling method
        Returns:
            (B, feature_dim) video embedding
        """
        if frames.dim() == 4:
            # (T, 3, H, W) -> (1, T, 3, H, W)
            frames = frames.unsqueeze(0)

        B, T, C, H, W = frames.shape

        frames_flat = frames.view(B * T, C, H, W)
        features_2048 = self.extract_backbone_features(frames_flat)
        features_512 = self.projection(features_2048)
        features_512 = features_512.view(B, T, self.feature_dim)

        video_emb = self.aggregate_frames(features_512, method=aggregation_method)
        return video_emb


class CLIPExperimentalEncoder(BaseVisualEncoder):
    """
    Experimental Visual Transformer Encoder using OpenAI CLIP Vision.
    Opt-in only; ResNet-50 remains production baseline.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        feature_dim: int = 512,
        pretrained: bool = False,
        freeze_backbone: bool = True,
        aggregation: str = "mean",
    ):
        super().__init__(feature_dim=feature_dim, aggregation=aggregation)
        self.model_name = model_name
        self.pretrained = pretrained

        try:
            from transformers import CLIPVisionConfig, CLIPVisionModel
            if pretrained:
                self.clip_model = CLIPVisionModel.from_pretrained(model_name)
            else:
                # Deterministic lightweight config for offline testing
                config = CLIPVisionConfig(
                    hidden_size=768,
                    intermediate_size=3072,
                    num_hidden_layers=2,
                    num_attention_heads=12,
                    image_size=224,
                    patch_size=32,
                )
                self.clip_model = CLIPVisionModel(config)

            self.clip_dim = getattr(self.clip_model.config, "hidden_size", 768)

            if freeze_backbone:
                for param in self.clip_model.parameters():
                    param.requires_grad = False

        except ImportError:
            logger.warning("transformers library not available for CLIPExperimentalEncoder.")
            self.clip_model = None
            self.clip_dim = 768

        # Projection from CLIP hidden dimension to 512-D
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, self.clip_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.clip_dim, feature_dim),
        )

        self.eval()

    def encode_single_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            if self.clip_model is not None:
                outputs = self.clip_model(pixel_values=image_tensor)
                feat = outputs.pooler_output  # (B, 768)
            else:
                # Fallback dummy representation
                feat = torch.zeros(image_tensor.size(0), self.clip_dim, device=image_tensor.device)

            return self.projection(feat)

    def forward(
        self,
        frames: torch.Tensor,
        aggregation_method: Optional[str] = None,
    ) -> torch.Tensor:
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)

        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)

        if self.clip_model is not None:
            outputs = self.clip_model(pixel_values=frames_flat)
            features = outputs.pooler_output
        else:
            features = torch.zeros(B * T, self.clip_dim, device=frames.device)

        features_512 = self.projection(features)
        features_512 = features_512.view(B, T, self.feature_dim)
        return self.aggregate_frames(features_512, method=aggregation_method)


# Registry of visual encoders
ENCODER_REGISTRY = {
    "resnet50": ResNet50VisualEncoder,
    "clip": CLIPExperimentalEncoder,
}


def get_visual_encoder(
    name: str = "resnet50",
    feature_dim: int = 512,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    aggregation: str = "mean",
    **kwargs: Any,
) -> BaseVisualEncoder:
    """
    Factory function to instantiate visual encoders.

    Default: resnet50 (Production Baseline)
    """
    normalized_name = name.lower().strip()
    if normalized_name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unsupported visual encoder '{name}'. Available: {list(ENCODER_REGISTRY.keys())}"
        )

    encoder_cls = ENCODER_REGISTRY[normalized_name]
    return encoder_cls(
        feature_dim=feature_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        aggregation=aggregation,
        **kwargs,
    )


class CrossModalLinearProjector:
    """
    Phase 12C Experiment B: Cross-Modal Visual-to-Text Linear Projection.

    Maps 2048-D raw visual representations (ResNet-50 mean pooled frame vectors)
    into the 512-D text co-engagement embedding space (Phase 8C).

    Design Principles:
    - Closed-form Ridge Regression for exact, convex, deterministic fitting:
      W = (X^T X + alpha * I)^(-1) X^T Y
    - Validation-only hyperparameter tuning across regularization alphas.
    - Strict L2 unit normalization on projected outputs.
    - Finiteness and dimension validation.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 512,
        alpha: float = 1.0,
        weights: Optional[np.ndarray] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = float(alpha)
        if weights is not None:
            w_arr = np.asarray(weights, dtype=np.float32)
            if w_arr.shape != (input_dim, output_dim):
                raise ValueError(
                    f"Weight shape mismatch: expected ({input_dim}, {output_dim}), got {w_arr.shape}"
                )
            self.weights = w_arr
        else:
            self.weights = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: Optional[float] = None,
    ) -> "CrossModalLinearProjector":
        """
        Fit projection weights using closed-form Ridge Regression on training data only.
        X: shape (N, input_dim)
        Y: shape (N, output_dim)
        """
        import numpy as np

        X_arr = np.asarray(X, dtype=np.float32)
        Y_arr = np.asarray(Y, dtype=np.float32)
        if X_arr.ndim != 2 or Y_arr.ndim != 2:
            raise ValueError("X and Y must be 2D matrices")
        N, d_in = X_arr.shape
        N_y, d_out = Y_arr.shape
        if N != N_y:
            raise ValueError(f"Row count mismatch: X has {N}, Y has {N_y}")
        if d_in != self.input_dim or d_out != self.output_dim:
            raise ValueError(
                f"Dimension mismatch: expected ({self.input_dim}, {self.output_dim}), got ({d_in}, {d_out})"
            )
        if not np.all(np.isfinite(X_arr)) or not np.all(np.isfinite(Y_arr)):
            raise ValueError("Training data contains NaN or infinite values")

        reg_alpha = float(alpha if alpha is not None else self.alpha)
        # Ridge closed form: (X^T X + alpha * I)^(-1) X^T Y
        reg_matrix = reg_alpha * np.eye(self.input_dim, dtype=np.float32)
        normal_eq = X_arr.T @ X_arr + reg_matrix
        rhs = X_arr.T @ Y_arr
        self.weights = np.linalg.solve(normal_eq, rhs).astype(np.float32)
        self.alpha = reg_alpha
        return self

    def project(self, X: np.ndarray) -> np.ndarray:
        """
        Project visual features into text embedding space with L2 unit normalization.
        """
        import numpy as np

        if self.weights is None:
            raise RuntimeError("Projector weights have not been fitted.")
        X_arr = np.asarray(X, dtype=np.float32)
        single = (X_arr.ndim == 1)
        if single:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {X_arr.shape[1]}"
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Input contains NaN or infinite values")

        proj = X_arr @ self.weights  # (N, output_dim)
        norms = np.linalg.norm(proj, axis=1, keepdims=True)
        norms = np.where(norms > 1e-7, norms, 1.0)
        proj_norm = proj / norms
        return proj_norm.squeeze(0) if single else proj_norm

    def evaluate_cosine_similarity(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Evaluate mean cosine similarity between projected X and ground-truth Y."""
        import numpy as np

        proj = self.project(X)
        Y_arr = np.asarray(Y, dtype=np.float32)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(1, -1)
        y_norms = np.linalg.norm(Y_arr, axis=1, keepdims=True)
        y_norms = np.where(y_norms > 1e-7, y_norms, 1.0)
        y_unit = Y_arr / y_norms
        sims = np.sum(proj * y_unit, axis=1)
        return float(np.mean(sims))

    @classmethod
    def tune_and_fit(
        cls,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        alphas: Optional[list] = None,
    ) -> tuple:
        """
        Select hyperparameter alpha using VALIDATION set only, then return fitted projector.
        Guarantees validation-only hyperparameter tuning without test set contamination.
        """
        if alphas is None:
            alphas = [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]

        best_alpha = alphas[0]
        best_val_sim = -float("inf")
        tuning_history = {}

        for a in alphas:
            proj = cls(input_dim=X_train.shape[1], output_dim=Y_train.shape[1], alpha=a)
            proj.fit(X_train, Y_train)
            val_sim = proj.evaluate_cosine_similarity(X_val, Y_val)
            tuning_history[str(a)] = val_sim
            if val_sim > best_val_sim:
                best_val_sim = val_sim
                best_alpha = a

        # Final projector fitted with best alpha on train split
        final_proj = cls(input_dim=X_train.shape[1], output_dim=Y_train.shape[1], alpha=best_alpha)
        final_proj.fit(X_train, Y_train)
        telemetry = {
            "best_alpha": best_alpha,
            "best_val_cosine_sim": best_val_sim,
            "tuning_history": tuning_history,
        }
        return final_proj, telemetry

