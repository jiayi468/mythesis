from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_image(image: Image.Image, size: int = 224) -> torch.Tensor:
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(CLIP_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def downsample_mask(mask: np.ndarray | Image.Image, grid_size: int = 14) -> torch.Tensor:
    array = np.asarray(mask.convert("L") if isinstance(mask, Image.Image) else mask)
    binary = torch.from_numpy((array > 0).astype(np.float32))[None, None]
    pooled = F.adaptive_max_pool2d(binary, (grid_size, grid_size))
    return pooled[0, 0]
