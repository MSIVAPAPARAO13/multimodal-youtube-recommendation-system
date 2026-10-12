"""
Text Projection Head for Metadata Co-Engagement Representation Learning.

Experiment: EXP_MICROLENS_TEXT_COENGAGEMENT_V1
Maps frozen BERT-base 768-D representations into a 512-D metric space.
Maintains 1:1 structural equivalence with TextEncoder.projection in src/features/text_encoder.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextProjectionModel(nn.Module):
    """
    Trainable text projection head mapping 768-D BERT features to 512-D unit sphere.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        output_dim: int = 512,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize

        # Standard Phase 5 / Phase 8 architecture
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to 512-D normalized embeddings.
        Args:
            x: (B, 768) raw or frozen BERT representations
        Returns:
            (B, 512) normalized embeddings
        """
        proj = self.projection(x)
        if self.normalize:
            proj = F.normalize(proj, p=2, dim=-1)
        return proj


class LinearProjectionModel(nn.Module):
    """
    Direct single-layer linear projection mapping 768-D BERT features to 512-D unit sphere.
    Validated in Phase 8C (EXP_MICROLENS_TEXT_COENGAGEMENT_V2) to preserve continuous
    semantic geometry without ReLU non-linear clipping.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 512,
        normalize: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to 512-D normalized embeddings.
        Args:
            x: (B, 768) raw or frozen BERT representations
        Returns:
            (B, 512) normalized embeddings on unit sphere
        """
        out = self.linear(x)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


def load_phase_8c_projection(
    checkpoint_path: str = "data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt",
    device: torch.device = None,
) -> LinearProjectionModel:
    """
    Safely load the verified Phase 8C LinearProjectionModel checkpoint.

    Validates:
    - Checkpoint file exists
    - Compatible state dictionary keys and tensor dimensions
    - Sets model to eval mode
    """
    from pathlib import Path
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Phase 8C checkpoint not found at '{checkpoint_path}'. "
            "Ensure EXP_MICROLENS_TEXT_COENGAGEMENT_V2 training has completed."
        )

    if device is None:
        device = torch.device("cpu")

    model = LinearProjectionModel(input_dim=768, output_dim=512, normalize=True).to(device)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model

