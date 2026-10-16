import torch

from clip_anomaly.model import PatchAnomalyHead, image_scores_from_map


def test_patch_head_returns_probabilities_and_expected_shapes() -> None:
    prototypes = torch.eye(2, 4)
    model = PatchAnomalyHead(input_dim=6, embedding_dim=4, hidden_dim=8, prototypes=prototypes, dropout=0.0)
    features = torch.randn(3, 196, 6)

    anomaly_map, image_score = model(features)

    assert anomaly_map.shape == (3, 14, 14)
    assert image_score.shape == (3,)
    assert torch.all((0.0 <= anomaly_map) & (anomaly_map <= 1.0))


def test_image_score_is_995_percentile() -> None:
    anomaly_map = torch.arange(196, dtype=torch.float32).reshape(1, 14, 14) / 195

    actual = image_scores_from_map(anomaly_map)
    expected = torch.quantile(anomaly_map.flatten(1), 0.995, dim=1)

    assert torch.allclose(actual, expected)


def test_text_prototypes_are_fixed_buffers() -> None:
    model = PatchAnomalyHead(6, 4, 8, torch.eye(2, 4), dropout=0.0)

    assert "text_prototypes" in dict(model.named_buffers())
    assert "text_prototypes" not in dict(model.named_parameters())
