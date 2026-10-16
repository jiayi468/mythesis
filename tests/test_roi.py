import numpy as np

from clip_anomaly.roi import expand_bbox, fuse_maps, mask_bbox, paste_roi_map


def test_mask_bbox_uses_exclusive_maximum_coordinates() -> None:
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 7:11] = 1

    assert mask_bbox(mask) == (7, 2, 11, 5)


def test_bbox_expands_and_clips_at_image_boundary() -> None:
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[:10, 110:] = 1

    box = expand_bbox(mask_bbox(mask), image_width=120, image_height=100, ratio=0.25)

    assert 0 <= box[0] < box[2] <= 120
    assert 0 <= box[1] < box[3] <= 100
    assert box[2] == 120


def test_paste_and_fuse_roi_map() -> None:
    canvas = np.zeros((20, 30), dtype=np.float32)
    projected = paste_roi_map(canvas, np.ones((2, 2), dtype=np.float32), (10, 5, 20, 15))

    assert projected.shape == canvas.shape
    assert np.isclose(projected[5:15, 10:20].min(), 1.0)
    assert projected[:5].max() == 0.0
    assert np.array_equal(fuse_maps(np.full_like(canvas, 0.25), projected), np.maximum(0.25, projected))


def test_empty_mask_has_no_bbox() -> None:
    try:
        mask_bbox(np.zeros((10, 10), dtype=np.uint8))
    except ValueError as error:
        assert "empty" in str(error).lower()
    else:
        raise AssertionError("empty masks must be rejected")
