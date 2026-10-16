from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np


@dataclass(frozen=True)
class Candidate:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    method: str
    rank: int


@dataclass(frozen=True)
class RecallResult:
    recalled_instances: int
    total_instances: int
    recall: float
    coverages: tuple[float, ...]


def _iou(left: Candidate, right: Candidate) -> float:
    width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    if intersection == 0:
        return 0.0
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    return intersection / (left_area + right_area - intersection)


def _centered_box(cx: float, cy: float, width: int, height: int, image_width: int, image_height: int):
    x1 = min(max(0, int(round(cx - width / 2))), image_width - width)
    y1 = min(max(0, int(round(cy - height / 2))), image_height - height)
    return x1, y1, x1 + width, y1 + height


def generate_candidates(
    anomaly_map: np.ndarray,
    image_size: tuple[int, int],
    k: int,
    config: dict[str, object],
) -> list[Candidate]:
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("anomaly_map must be two-dimensional")
    if k < 0:
        raise ValueError("k must be non-negative")
    if str(config.get("method", "peak_nms")) != "peak_nms":
        raise ValueError("only the frozen peak_nms method is supported")
    image_width, image_height = map(int, image_size)
    fraction = float(config.get("window_fraction", 0.35))
    max_area = float(config.get("max_area_fraction", 0.4))
    nms_iou = float(config.get("nms_iou", 0.3))
    if not 0 < fraction <= 1 or not 0 < max_area <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("candidate configuration is outside valid bounds")
    side = max(1, round(min(image_width, image_height) * fraction))
    side = min(side, image_width, image_height)
    while k * side * side > max_area * image_width * image_height and side > 1:
        side -= 1

    order = np.argsort(-values, axis=None, kind="stable")
    selected: list[Candidate] = []
    map_height, map_width = values.shape
    for flat_index in order:
        row, column = np.unravel_index(flat_index, values.shape)
        cx = (column + 0.5) * image_width / map_width
        cy = (row + 0.5) * image_height / map_height
        box = _centered_box(cx, cy, side, side, image_width, image_height)
        candidate = Candidate(*box, float(values[row, column]), "peak_nms", len(selected) + 1)
        if all(_iou(candidate, existing) <= nms_iou for existing in selected):
            selected.append(candidate)
            if len(selected) == k:
                break
    return selected


def candidate_union_area_ratio(candidates: list[Candidate], image_size: tuple[int, int]) -> float:
    width, height = map(int, image_size)
    canvas = np.zeros((height, width), dtype=bool)
    for item in candidates:
        canvas[item.y1:item.y2, item.x1:item.x2] = True
    return float(canvas.mean())


def generate_random_candidates(
    image_size: tuple[int, int], k: int, config: dict[str, object], seed: int
) -> list[Candidate]:
    width, height = map(int, image_size)
    fraction = float(config.get("window_fraction", 0.35))
    max_area = float(config.get("max_area_fraction", 0.4))
    nms_iou = float(config.get("nms_iou", 0.3))
    side = max(1, min(width, height, round(min(width, height) * fraction)))
    while k * side * side > max_area * width * height and side > 1:
        side -= 1
    rng = np.random.default_rng(seed)
    selected: list[Candidate] = []
    attempts = 0
    while len(selected) < k and attempts < 1000:
        attempts += 1
        x1 = int(rng.integers(0, width - side + 1))
        y1 = int(rng.integers(0, height - side + 1))
        candidate = Candidate(x1, y1, x1 + side, y1 + side, 0.0, "random", len(selected) + 1)
        if all(_iou(candidate, existing) <= nms_iou for existing in selected):
            selected.append(candidate)
    return selected


def generate_oracle_candidates(
    gt_mask: np.ndarray, image_size: tuple[int, int], k: int, config: dict[str, object]
) -> list[Candidate]:
    mask = np.asarray(gt_mask) > 0
    if mask.shape != (image_size[1], image_size[0]):
        raise ValueError("gt_mask shape must match image_size")
    width, height = image_size
    fraction = float(config.get("window_fraction", 0.35))
    max_area = float(config.get("max_area_fraction", 0.4))
    side = max(1, min(width, height, round(min(width, height) * fraction)))
    while k * side * side > max_area * width * height and side > 1:
        side -= 1
    ranked = sorted(connected_components(mask), key=lambda item: int(item.sum()), reverse=True)
    output = []
    for rank, component in enumerate(ranked[:k], start=1):
        locations = np.argwhere(component)
        cy, cx = locations.mean(axis=0)
        box = _centered_box(float(cx), float(cy), side, side, width, height)
        output.append(Candidate(*box, float(component.sum()), "oracle", rank))
    return output


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    binary = np.asarray(mask) > 0
    visited = np.zeros(binary.shape, dtype=bool)
    output: list[np.ndarray] = []
    height, width = binary.shape
    for start_y, start_x in np.argwhere(binary):
        if visited[start_y, start_x]:
            continue
        component = np.zeros(binary.shape, dtype=bool)
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            component[y, x] = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (
                        (dy or dx) and 0 <= ny < height and 0 <= nx < width
                        and binary[ny, nx] and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        output.append(component)
    return output


def region_recall_components(
    candidates: list[Candidate], components: list[np.ndarray], shape: tuple[int, int], min_coverage: float = 0.5
) -> RecallResult:
    union = np.zeros(shape, dtype=bool)
    for item in candidates:
        union[item.y1:item.y2, item.x1:item.x2] = True
    coverages = tuple(float((component & union).sum() / component.sum()) for component in components)
    recalled = sum(value >= min_coverage for value in coverages)
    return RecallResult(recalled, len(components), recalled / len(components) if components else 0.0, coverages)


def region_recall(
    candidates: list[Candidate], gt_mask: np.ndarray, min_coverage: float = 0.5
) -> RecallResult:
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between zero and one")
    mask = np.asarray(gt_mask)
    if mask.ndim != 2:
        raise ValueError("gt_mask must be two-dimensional")
    return region_recall_components(candidates, connected_components(mask), mask.shape, min_coverage)
