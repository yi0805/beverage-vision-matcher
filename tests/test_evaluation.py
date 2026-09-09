"""Tests for portable evaluation discovery, summaries, and CSV output."""

import csv
from pathlib import Path

import pytest

from beverage_matcher.evaluation import (
    EvaluationRecord,
    discover_query_images,
    parse_query_metadata,
    summarize_by_condition,
    summarize_by_product,
    summarize_records,
    write_evaluation_csv,
)


def _record(
    query_image: str,
    expected_product: str,
    condition: str,
    predicted_product: str,
    accepted: bool,
) -> EvaluationRecord:
    return EvaluationRecord(
        query_image=query_image,
        expected_product=expected_product,
        condition=condition,
        predicted_product=predicted_product,
        best_candidate="goodbuzz",
        reference_image="goodbuzz/reference.jpg",
        good_matches=12,
        ransac_inliers=9,
        inlier_ratio=0.75,
        score=6.75,
        accepted=accepted,
        correct=predicted_product == expected_product,
    )


def test_parse_query_metadata_uses_parent_label_and_filename_condition() -> None:
    path = Path("queries/goodbuzz/goodbuzz__low_light.jpg")

    assert parse_query_metadata(path) == ("goodbuzz", "low_light")


def test_parse_query_metadata_rejects_malformed_filename() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_query_metadata(Path("queries/goodbuzz/goodbuzz_angle.jpg"))


def test_parse_query_metadata_rejects_parent_label_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        parse_query_metadata(Path("queries/goodbuzz/redbull__angle.jpg"))


def test_query_discovery_is_deterministic_and_ignores_hidden_files(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "b" / "b__normal.png").touch()
    (tmp_path / "a" / "a__glare.jpg").touch()
    (tmp_path / ".hidden" / "hidden__angle.jpg").touch()
    (tmp_path / "a" / ".ignored.jpg").touch()

    discovered = discover_query_images(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in discovered] == [
        "a/a__glare.jpg",
        "b/b__normal.png",
    ]


def test_summary_distinguishes_correct_unknown_and_accepted_wrong() -> None:
    records = [
        _record("goodbuzz/goodbuzz__normal.jpg", "goodbuzz", "normal", "goodbuzz", True),
        _record("mo/mo__glare.jpg", "mo", "glare", "UNKNOWN", False),
        _record("redbull/redbull__angle.jpg", "redbull", "angle", "goodbuzz", True),
    ]

    summary = summarize_records(records)

    assert summary == {
        "total_queries": 3,
        "correct_predictions": 1,
        "incorrect_predictions": 2,
        "unknown_predictions": 1,
        "accepted_predictions": 2,
        "accepted_but_wrong_predictions": 1,
        "overall_accuracy": 1 / 3,
    }


def test_condition_and_product_summaries_reconcile() -> None:
    records = [
        _record("goodbuzz/goodbuzz__normal.jpg", "goodbuzz", "normal", "goodbuzz", True),
        _record("mo/mo__normal.jpg", "mo", "normal", "UNKNOWN", False),
        _record("goodbuzz/goodbuzz__glare.jpg", "goodbuzz", "glare", "goodbuzz", True),
    ]

    by_condition = summarize_by_condition(records)
    by_product = summarize_by_product(records)

    assert by_condition["normal"]["total_queries"] == 2
    assert by_condition["normal"]["correct_predictions"] == 1
    assert by_product["goodbuzz"]["total_queries"] == 2
    assert by_product["mo"]["unknown_predictions"] == 1


def test_csv_uses_relative_portable_paths_and_stable_numeric_format(tmp_path: Path) -> None:
    record = _record("goodbuzz/goodbuzz__normal.jpg", "goodbuzz", "normal", "goodbuzz", True)
    output_path = write_evaluation_csv([record], tmp_path / "evaluation.csv")

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))

    assert row["query_image"] == "goodbuzz/goodbuzz__normal.jpg"
    assert row["reference_image"] == "goodbuzz/reference.jpg"
    assert row["inlier_ratio"] == "0.750"
    assert row["score"] == "6.750"
    assert ":/" not in row["query_image"]
