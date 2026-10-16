import json
from pathlib import Path

from clip_anomaly.decisive import build_result, write_result


def test_result_status_is_incomplete_when_any_planned_item_is_missing(tmp_path: Path) -> None:
    result = build_result(
        criteria={"metric": ">=0.2"},
        observed={"metric": 0.3},
        artifact_paths=["raw.csv"],
        protocol_hash="abc",
        git_commit="def",
        started_at="2026-10-16T10:00:00+08:00",
        completed_at="2026-10-16T10:01:00+08:00",
        planned=10,
        succeeded=9,
        failed=1,
        skipped=0,
        failure_reasons=["one backend failed"],
        criteria_passed=True,
    )

    assert result["status"] == "INCOMPLETE"
    output = write_result(tmp_path / "result.json", result)
    assert json.loads(output.read_text(encoding="utf-8"))["planned"] == 10


def test_result_status_is_fail_only_for_complete_numeric_evidence() -> None:
    result = build_result(
        criteria={}, observed={}, artifact_paths=[], protocol_hash="abc", git_commit="def",
        started_at="s", completed_at="e", planned=4, succeeded=4, failed=0, skipped=0,
        failure_reasons=[], criteria_passed=False,
    )

    assert result["status"] == "FAIL"
