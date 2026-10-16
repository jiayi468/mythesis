from pathlib import Path

import pytest

from clip_anomaly.config import load_config


def test_load_config_resolves_paths_from_project_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project_root: .\n"
        "dataset_roots: [dataset/pill]\n"
        "output_root: midterm\n"
        "seed: 42\n"
        "split:\n  abnormal_train_fraction: 0.7\n"
        "model:\n  input_size: 224\n  patch_grid: 14\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project_root == tmp_path.resolve()
    assert config.dataset_roots == ((tmp_path / "dataset/pill").resolve(),)
    assert config.output_root == (tmp_path / "midterm").resolve()


def test_load_config_rejects_invalid_split_fraction(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project_root: .\n"
        "dataset_roots: [dataset/pill]\n"
        "output_root: midterm\n"
        "seed: 42\n"
        "split:\n  abnormal_train_fraction: 1.0\n"
        "model:\n  input_size: 224\n  patch_grid: 14\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="abnormal_train_fraction"):
        load_config(config_path)
