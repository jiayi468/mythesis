import numpy as np

from clip_anomaly.candidates import (
    Candidate, candidate_union_area_ratio, generate_candidates, generate_oracle_candidates,
    generate_random_candidates, region_recall,
)


def test_peak_nms_candidates_are_clipped_ranked_and_respect_area_budget() -> None:
    anomaly = np.zeros((8, 8), dtype=np.float32)
    anomaly[0, 0] = 1.0
    anomaly[7, 7] = 0.9

    candidates = generate_candidates(
        anomaly,
        image_size=(80, 40),
        k=2,
        config={"method": "peak_nms", "window_fraction": 0.25, "nms_iou": 0.1, "max_area_fraction": 0.4},
    )

    assert [candidate.rank for candidate in candidates] == [1, 2]
    assert np.allclose([candidate.score for candidate in candidates], [1.0, 0.9])
    assert all(0 <= item.x1 < item.x2 <= 80 and 0 <= item.y1 < item.y2 <= 40 for item in candidates)
    assert candidate_union_area_ratio(candidates, (80, 40)) <= 0.4


def test_region_recall_counts_connected_instances_not_images() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:5, 1:5] = 1
    mask[12:16, 12:16] = 1
    candidates = [Candidate(0, 0, 6, 6, 1.0, "peak_nms", 1)]

    result = region_recall(candidates, mask, min_coverage=0.5)

    assert result.total_instances == 2
    assert result.recalled_instances == 1
    assert result.recall == 0.5


def test_region_recall_handles_empty_mask_explicitly() -> None:
    result = region_recall([], np.zeros((10, 10), dtype=np.uint8))

    assert result.total_instances == 0
    assert result.recalled_instances == 0
    assert result.recall == 0.0


def test_random_candidates_are_deterministic_and_match_auto_area() -> None:
    config = {"window_fraction": 0.25, "nms_iou": 0.1, "max_area_fraction": 0.4}

    first = generate_random_candidates((80, 40), 2, config, seed=11)
    second = generate_random_candidates((80, 40), 2, config, seed=11)
    auto = generate_candidates(np.eye(8, dtype=np.float32), (80, 40), 2, {"method": "peak_nms", **config})

    assert first == second
    assert sum((item.x2 - item.x1) * (item.y2 - item.y1) for item in first) == sum(
        (item.x2 - item.x1) * (item.y2 - item.y1) for item in auto
    )


def test_oracle_candidates_rank_components_and_use_same_budget() -> None:
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    mask[25:28, 60:63] = 1
    config = {"window_fraction": 0.25, "max_area_fraction": 0.4}

    oracle = generate_oracle_candidates(mask, (80, 40), 2, config)
    random = generate_random_candidates((80, 40), 2, config, 11)

    assert [item.score for item in oracle] == [36.0, 9.0]
    assert [(item.x2 - item.x1, item.y2 - item.y1) for item in oracle] == [
        (item.x2 - item.x1, item.y2 - item.y1) for item in random
    ]
