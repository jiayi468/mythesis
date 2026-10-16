from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _validate_binary(name: str, values: np.ndarray) -> None:
    unique = np.unique(values)
    if not np.array_equal(unique, np.array([0, 1])):
        raise ValueError(f"{name} must contain both binary classes 0 and 1; got {unique.tolist()}")


def compute_detection_metrics(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    pixel_labels: np.ndarray,
    pixel_scores: np.ndarray,
) -> dict[str, float]:
    image_labels = np.asarray(image_labels).astype(np.uint8).reshape(-1)
    image_scores = np.asarray(image_scores).reshape(-1)
    pixel_labels = np.asarray(pixel_labels).astype(np.uint8).reshape(-1)
    pixel_scores = np.asarray(pixel_scores).reshape(-1)
    _validate_binary("image_labels", image_labels)
    _validate_binary("pixel_labels", pixel_labels)
    return {
        "image_auroc": float(roc_auc_score(image_labels, image_scores)),
        "image_ap": float(average_precision_score(image_labels, image_scores)),
        "pixel_auroc": float(roc_auc_score(pixel_labels, pixel_scores)),
        "pixel_ap": float(average_precision_score(pixel_labels, pixel_scores)),
    }
