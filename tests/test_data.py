from pathlib import Path

from PIL import Image

from clip_anomaly.data import discover_category_root, scan_mvtec_category, split_records


def _png(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=value).save(path)


def _category(tmp_path: Path) -> Path:
    outer = tmp_path / "pill"
    root = outer / "pill"
    _png(root / "train/good/000.png")
    _png(root / "train/good/001.png")
    _png(root / "test/good/010.png")
    for defect, names in {"crack": ["020", "021", "022"], "color": ["030", "031"]}.items():
        for name in names:
            _png(root / "test" / defect / f"{name}.png", 120)
            _png(root / "ground_truth" / defect / f"{name}_mask.png", 255)
    return outer


def test_discover_category_root_accepts_double_nested_directory(tmp_path: Path) -> None:
    outer = _category(tmp_path)

    assert discover_category_root(outer) == outer / "pill"


def test_scan_pairs_anomalies_and_assigns_zero_mask_to_good(tmp_path: Path) -> None:
    records = scan_mvtec_category(_category(tmp_path))

    assert len(records) == 8
    abnormal = [record for record in records if record.label == 1]
    normal = [record for record in records if record.label == 0]
    assert all(record.mask_path is not None and record.mask_path.exists() for record in abnormal)
    assert all(record.mask_path is None for record in normal)


def test_split_is_deterministic_stratified_and_non_overlapping(tmp_path: Path) -> None:
    records = scan_mvtec_category(_category(tmp_path))

    first = split_records(records, seed=42, abnormal_train_fraction=0.7)
    second = split_records(records, seed=42, abnormal_train_fraction=0.7)

    assert first == second
    train_paths = {record.image_path for record in first if record.split == "train"}
    val_paths = {record.image_path for record in first if record.split == "val"}
    assert train_paths.isdisjoint(val_paths)
    assert {record.defect_type for record in first if record.split == "val" and record.label == 1} == {"crack", "color"}
