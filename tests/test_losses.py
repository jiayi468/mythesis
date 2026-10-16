import torch

from clip_anomaly.losses import DiceLoss, FocalLoss


def test_perfect_prediction_has_near_zero_dice_loss() -> None:
    target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    prediction = target.clone()

    assert DiceLoss()(prediction, target).item() < 1e-5


def test_wrong_prediction_has_higher_combined_loss() -> None:
    target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    good = target * 0.98 + (1 - target) * 0.02
    bad = 1 - good
    criterion = lambda pred: FocalLoss(gamma=2.0)(pred, target) + DiceLoss()(pred, target)

    assert criterion(bad) > criterion(good)
