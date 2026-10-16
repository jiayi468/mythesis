import numpy as np
import torch
from PIL import Image

from clip_anomaly.preprocessing import CLIP_MEAN, CLIP_STD, preprocess_image, downsample_mask


def test_preprocess_image_returns_normalized_rgb_chw() -> None:
    image = Image.new("RGB", (20, 10), color=(255, 0, 0))

    tensor = preprocess_image(image, size=224)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    expected_red = (1.0 - CLIP_MEAN[0]) / CLIP_STD[0]
    assert torch.isclose(tensor[0, 100, 100], torch.tensor(expected_red), atol=1e-5)


def test_downsample_mask_preserves_single_defect_pixel() -> None:
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[17, 19] = 255

    patch_mask = downsample_mask(mask, grid_size=14)

    assert patch_mask.shape == (14, 14)
    assert patch_mask.dtype == torch.float32
    assert patch_mask.sum().item() >= 1.0


def test_downsample_empty_mask_stays_empty() -> None:
    patch_mask = downsample_mask(np.zeros((224, 224), dtype=np.uint8), grid_size=14)

    assert patch_mask.sum().item() == 0.0
