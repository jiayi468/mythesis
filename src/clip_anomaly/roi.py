from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


BBox = tuple[int, int, int, int]


def mask_bbox(mask: np.ndarray) -> BBox:
    array = np.asarray(mask)
    locations = np.argwhere(array > 0)
    if locations.size == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask")
    y0, x0 = locations.min(axis=0)
    y1, x1 = locations.max(axis=0) + 1
    return int(x0), int(y0), int(x1), int(y1)


def expand_bbox(box: BBox, image_width: int, image_height: int, ratio: float = 0.25) -> BBox:
    if ratio < 0:
        raise ValueError("ratio must be non-negative")
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    expanded = (
        max(0, math.floor(x0 - width * ratio)),
        max(0, math.floor(y0 - height * ratio)),
        min(image_width, math.ceil(x1 + width * ratio)),
        min(image_height, math.ceil(y1 + height * ratio)),
    )
    if expanded[0] >= expanded[2] or expanded[1] >= expanded[3]:
        raise ValueError("expanded bounding box is empty")
    return expanded


def paste_roi_map(canvas: np.ndarray, roi_map: np.ndarray, box: BBox) -> np.ndarray:
    output = np.asarray(canvas, dtype=np.float32).copy()
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= output.shape[1] and 0 <= y0 < y1 <= output.shape[0]):
        raise ValueError("box must lie within the canvas")
    tensor = torch.as_tensor(np.asarray(roi_map), dtype=torch.float32)[None, None]
    resized = F.interpolate(tensor, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0, 0]
    output[y0:y1, x0:x1] = resized.numpy()
    return output


def fuse_maps(full_map: np.ndarray, projected_roi_map: np.ndarray) -> np.ndarray:
    full = np.asarray(full_map, dtype=np.float32)
    roi = np.asarray(projected_roi_map, dtype=np.float32)
    if full.shape != roi.shape:
        raise ValueError("full and ROI maps must share a shape")
    return np.maximum(full, roi)
