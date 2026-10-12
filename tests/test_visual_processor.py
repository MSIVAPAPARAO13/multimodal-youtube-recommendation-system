"""
Tests for VisualProcessor (Phase 6):
- RGB conversion across input formats (PIL, numpy, BGR, RGBA, grayscale)
- ImageNet normalization shape and parameters
- Consistency between video frame preprocessing and thumbnail preprocessing
- Batch frame stacking
- Error handling
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.features.visual_processor import (
    VisualProcessor,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


@pytest.fixture
def processor():
    return VisualProcessor(target_size=(224, 224), normalize=True)


def test_rgb_conversion_from_various_formats(processor):
    """Test preprocessing accepts PIL images, uint8 arrays, floats, and converts to RGB."""
    # 1. PIL Image (RGB)
    pil_rgb = Image.new("RGB", (100, 100), color=(255, 128, 0))
    t1 = processor.preprocess_frame(pil_rgb)
    assert t1.shape == (3, 224, 224)
    assert t1.dtype == torch.float32

    # 2. PIL Image (RGBA)
    pil_rgba = Image.new("RGBA", (100, 100), color=(0, 255, 128, 200))
    t2 = processor.preprocess_frame(pil_rgba)
    assert t2.shape == (3, 224, 224)

    # 3. PIL Image (Grayscale L)
    pil_gray = Image.new("L", (100, 100), color=128)
    t3 = processor.preprocess_frame(pil_gray)
    assert t3.shape == (3, 224, 224)

    # 4. Numpy array (H, W, 3) uint8
    arr_uint8 = np.ones((80, 80, 3), dtype=np.uint8) * 200
    t4 = processor.preprocess_frame(arr_uint8)
    assert t4.shape == (3, 224, 224)

    # 5. Numpy array (H, W, 3) float in [0, 1]
    arr_float = np.ones((80, 80, 3), dtype=np.float32) * 0.5
    t5 = processor.preprocess_frame(arr_float)
    assert t5.shape == (3, 224, 224)


def test_imagenet_normalization_values(processor):
    """Verify ImageNet normalization math: (x - mean) / std."""
    # Black image (all zeros)
    black_img = np.zeros((224, 224, 3), dtype=np.uint8)
    t_black = processor.preprocess_frame(black_img)

    # (0.0 - mean) / std for each channel
    expected_c0 = (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    expected_c1 = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    expected_c2 = (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    assert pytest.approx(t_black[0, 0, 0].item(), abs=1e-3) == expected_c0
    assert pytest.approx(t_black[1, 0, 0].item(), abs=1e-3) == expected_c1
    assert pytest.approx(t_black[2, 0, 0].item(), abs=1e-3) == expected_c2


def test_thumbnail_preprocessing_shape_and_consistency(processor, tmp_path):
    """Thumbnail and video frame preprocessing apply identical transforms."""
    test_img = Image.new("RGB", (320, 180), color=(100, 150, 200))
    thumb_path = tmp_path / "thumb.jpg"
    test_img.save(thumb_path)

    # Preprocess via preprocess_thumbnail
    t_thumb = processor.preprocess_thumbnail(thumb_path)
    assert t_thumb.shape == (1, 3, 224, 224)

    # Preprocess same image via preprocess_frame
    t_frame = processor.preprocess_frame(thumb_path)
    assert t_frame.shape == (3, 224, 224)

    # Values must be bit-exact identical
    torch.testing.assert_close(t_thumb[0], t_frame)


def test_batch_frame_stacking(processor):
    """Sequence of 8 frames preprocesses into an (8, 3, 224, 224) tensor."""
    frames = [np.full((64, 64, 3), i * 30, dtype=np.uint8) for i in range(8)]
    tensor_stack = processor.preprocess_frames(frames)

    assert tensor_stack.shape == (8, 3, 224, 224)
    assert tensor_stack.dtype == torch.float32
    assert not torch.isnan(tensor_stack).any()
    assert not torch.isinf(tensor_stack).any()


def test_missing_file_raises_error(processor, tmp_path):
    """Non-existent image file raises FileNotFoundError."""
    missing = tmp_path / "missing_img.jpg"
    with pytest.raises(FileNotFoundError):
        processor.preprocess_frame(missing)
