import numpy as np

from scripts.compare_browser_preprocessing import compare_tensors


def test_compare_tensors_reports_max_and_mean_absolute_error() -> None:
    actual = compare_tensors(np.array([0.0, 1.0]), np.array([0.0, 1.25]))

    assert actual == {"max_abs_error": 0.25, "mean_abs_error": 0.125}


def test_compare_tensors_rejects_shape_mismatch() -> None:
    try:
        compare_tensors(np.zeros(2), np.zeros(3))
    except ValueError as error:
        assert "shape" in str(error).lower()
    else:
        raise AssertionError("shape mismatch must fail")
