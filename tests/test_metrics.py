import numpy as np
import pytest

from clip_anomaly.metrics import compute_detection_metrics


def test_metrics_are_one_for_perfect_ranking() -> None:
    image_labels = np.array([0, 0, 1, 1])
    image_scores = np.array([0.05, 0.1, 0.8, 0.9])
    pixel_labels = np.array([0, 0, 0, 1, 1, 1])
    pixel_scores = np.array([0.01, 0.02, 0.03, 0.8, 0.9, 0.95])

    result = compute_detection_metrics(image_labels, image_scores, pixel_labels, pixel_scores)

    assert result == {
        "image_auroc": 1.0,
        "image_ap": 1.0,
        "pixel_auroc": 1.0,
        "pixel_ap": 1.0,
    }


def test_metrics_reject_single_class_image_labels() -> None:
    with pytest.raises(ValueError, match="image_labels"):
        compute_detection_metrics(
            np.zeros(3), np.array([0.1, 0.2, 0.3]), np.array([0, 1]), np.array([0.1, 0.9])
        )
