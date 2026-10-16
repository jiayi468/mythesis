from pathlib import Path

from clip_anomaly.experiment import build_run_matrix, supplement_output


def test_run_matrix_has_one_raw_and_three_trainable_runs() -> None:
    assert build_run_matrix([42, 123, 2026]) == [
        ("b0_raw_clip", None),
        ("b1_classifier", 42),
        ("b1_classifier", 123),
        ("b1_classifier", 2026),
        ("b2_text_prototype", 42),
        ("b2_text_prototype", 123),
        ("b2_text_prototype", 2026),
    ]


def test_supplement_output_separates_head_and_seed(tmp_path: Path) -> None:
    paths = {
        supplement_output(tmp_path, head, seed)
        for head in ("b1_classifier", "b2_text_prototype")
        for seed in (42, 123, 2026)
    }

    assert len(paths) == 6
    assert tmp_path / "experiments" / "supplement" / "b1_classifier" / "seed_42" in paths
