import torch

from clip_anomaly.model import aggregate_image_scores


def test_score_aggregations_are_exact() -> None:
    values = torch.arange(196, dtype=torch.float32).reshape(1, 14, 14)

    assert aggregate_image_scores(values, "max").item() == 195.0
    assert aggregate_image_scores(values, "topk_mean", top_k=5).item() == 193.0
    assert torch.allclose(
        aggregate_image_scores(values, "quantile", quantile=0.995),
        torch.quantile(values.flatten(1), 0.995, dim=1),
    )


def test_score_aggregation_rejects_unknown_method() -> None:
    values = torch.zeros(1, 14, 14)

    try:
        aggregate_image_scores(values, "median")
    except ValueError as error:
        assert "median" in str(error)
    else:
        raise AssertionError("unknown score aggregation must fail")
