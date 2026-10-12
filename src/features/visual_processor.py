"""
VisualProcessor: Standardized ImageNet Preprocessing for Video Frames and YouTube Thumbnails.

Ensures strict consistency between sampled video frames and YouTube thumbnails:
- RGB conversion
- Deterministic resize and aspect-ratio handling / center crop to (224, 224)
- Float tensor conversion [0.0, 1.0]
- Canonical ImageNet normalization:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
"""

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VisualProcessor:
    """
    Standardizes image and video-frame preprocessing for ResNet-50 and visual encoders.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        use_center_crop: bool = False,
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.use_center_crop = use_center_crop

        # Build torchvision transforms pipeline
        transform_steps = []
        if use_center_crop:
            transform_steps.extend([
                transforms.Resize(256),
                transforms.CenterCrop(self.target_size),
            ])
        else:
            transform_steps.append(transforms.Resize(self.target_size))

        transform_steps.append(transforms.ToTensor())  # Converts PIL / ndarray [0, 255] to [0.0, 1.0]

        if normalize:
            transform_steps.append(
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            )

        self.transform = transforms.Compose(transform_steps)

    def _to_pil_rgb(self, image_input: Union[str, Path, np.ndarray, Image.Image, torch.Tensor]) -> Image.Image:
        """Convert arbitrary visual input into a PIL Image in RGB format."""
        if isinstance(image_input, (str, Path)):
            path_obj = Path(image_input)
            if not path_obj.exists() or not path_obj.is_file():
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = Image.open(path_obj)
            return img.convert("RGB")

        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, np.ndarray):
            # If floating point in [0, 1], scale to [0, 255] uint8
            arr = image_input
            if arr.dtype in (np.float32, np.float64):
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)

            # Check channels: if (C, H, W) permute to (H, W, C)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))

            if arr.ndim == 2:
                # Grayscale
                return Image.fromarray(arr).convert("RGB")
            elif arr.ndim == 3:
                if arr.shape[2] == 1:
                    return Image.fromarray(arr.squeeze(-1)).convert("RGB")
                elif arr.shape[2] == 4:
                    return Image.fromarray(arr, mode="RGBA").convert("RGB")
                return Image.fromarray(arr, mode="RGB")
            else:
                raise ValueError(f"Unsupported numpy image array dimensions: {arr.shape}")

        if isinstance(image_input, torch.Tensor):
            t = image_input.detach().cpu()
            if t.ndim == 3:
                # Shape (C, H, W)
                if t.shape[0] in (1, 3):
                    t = t.permute(1, 2, 0)
                arr = t.numpy()
                return self._to_pil_rgb(arr)
            elif t.ndim == 2:
                return self._to_pil_rgb(t.numpy())
            else:
                raise ValueError(f"Unsupported torch tensor shape: {t.shape}")

        raise TypeError(f"Unsupported visual input type: {type(image_input)}")

    def preprocess_frame(
        self,
        frame: Union[str, Path, np.ndarray, Image.Image, torch.Tensor],
    ) -> torch.Tensor:
        """
        Preprocess a single frame or image into a (3, H, W) normalized tensor.
        """
        pil_img = self._to_pil_rgb(frame)
        tensor = self.transform(pil_img)  # Shape: (3, H, W)
        return tensor

    def preprocess_frames(
        self,
        frames: Sequence[Union[str, Path, np.ndarray, Image.Image, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Preprocess a sequence of N frames into an (N, 3, H, W) tensor.
        """
        if not frames:
            return torch.zeros(0, 3, self.target_size[0], self.target_size[1], dtype=torch.float32)

        tensors = [self.preprocess_frame(f) for f in frames]
        return torch.stack(tensors, dim=0)

    def preprocess_thumbnail(
        self,
        thumbnail_input: Union[str, Path, np.ndarray, Image.Image, torch.Tensor],
    ) -> torch.Tensor:
        """
        Preprocess YouTube thumbnail into a (1, 3, H, W) normalized tensor.
        Consistent with video frame preprocessing.
        """
        frame_tensor = self.preprocess_frame(thumbnail_input)
        return frame_tensor.unsqueeze(0)  # Shape: (1, 3, H, W)
