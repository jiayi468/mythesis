from pathlib import Path

from PIL import Image

from clip_anomaly.data import decisive_split_records, scan_mvtec_category


def _png(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=value).save(path)


def _category(root: Path, name: str) -> Path:
    category = root / name
    for index in range(4):
        _png(category / "train/good" / f"{index:03d}.png")
        _png(category / "test/good" / f"{index:03d}.png")
    for defect in ("crack", "color"):
        for index in range(5):
            _png(category / "test" / defect / f"{index:03d}.png", 120)
            _png(category / "ground_truth" / defect / f"{index:03d}_mask.png", 255)
    return category


def test_decisive_split_is_deterministic_disjoint_and_keeps_holdout_isolated(tmp_path: Path) -> None:
    records = []
    for name in ("pill", "screw"):
        records.extend(scan_mvtec_category(_category(tmp_path, name)))

    first = decisive_split_records(records, seed=42, abnormal_train_fraction=0.6, dev_fraction=0.2)
    second = decisive_split_records(records, seed=42, abnormal_train_fraction=0.6, dev_fraction=0.2)

    assert first == second
    by_split = {
        split: {record.image_path for record in first if record.split == split}
        for split in ("train", "dev", "source_holdout")
    }
    assert by_split["train"].isdisjoint(by_split["dev"])
    assert by_split["train"].isdisjoint(by_split["source_holdout"])
    assert by_split["dev"].isdisjoint(by_split["source_holdout"])
    assert all(record.split == "train" for record in first if record.source_partition == "train")
    for category in ("pill", "screw"):
        for defect in ("crack", "color"):
            group = [record for record in first if record.category == category and record.defect_type == defect]
            assert {record.split for record in group} == {"train", "dev", "source_holdout"}
    test_good = [record for record in first if record.source_partition == "test" and record.label == 0]
    assert {record.split for record in test_good} == {"dev", "source_holdout"}


def test_decisive_split_rejects_fractions_that_leave_no_holdout(tmp_path: Path) -> None:
    records = scan_mvtec_category(_category(tmp_path, "pill"))

    try:
        decisive_split_records(records, seed=42, abnormal_train_fraction=0.8, dev_fraction=0.2)
    except ValueError as error:
        assert "holdout" in str(error).lower()
    else:
        raise AssertionError("fractions without holdout must fail")
