"""Repeatable batch evaluation for the baseline beverage reference matcher."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .matcher import MatcherConfig, best_accepted_match, match_query_against_references

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_CSV_FIELDS = (
    "query_image",
    "expected_product",
    "condition",
    "predicted_product",
    "best_candidate",
    "reference_image",
    "good_matches",
    "ransac_inliers",
    "inlier_ratio",
    "score",
    "accepted",
    "correct",
)


@dataclass(frozen=True)
class EvaluationRecord:
    """Measured result for one labelled query image."""

    query_image: str
    expected_product: str
    condition: str
    predicted_product: str
    best_candidate: str
    reference_image: str
    good_matches: int
    ransac_inliers: int
    inlier_ratio: float
    score: float
    accepted: bool
    correct: bool


def discover_query_images(query_directory: Path) -> list[Path]:
    """Return visible supported query images in deterministic path order."""
    directory = Path(query_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Query directory does not exist: {directory}")

    query_images = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in _IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    ]
    if not query_images:
        raise ValueError(f"No query images found in: {directory}")
    return query_images


def parse_query_metadata(query_path: Path) -> tuple[str, str]:
    """Extract and validate ``(expected_product, condition)`` from a query path."""
    path = Path(query_path)
    expected_product = path.parent.name
    filename_parts = path.stem.split("__")
    if len(filename_parts) != 2 or not all(filename_parts):
        raise ValueError(
            f"Malformed query filename (expected '<product>__<condition>'): {path}"
        )
    filename_product, condition = filename_parts
    if filename_product != expected_product:
        raise ValueError(
            f"Query filename product '{filename_product}' does not match parent label "
            f"'{expected_product}': {path}"
        )
    return expected_product, condition


def evaluate_queries(
    reference_directory: Path,
    query_directory: Path,
    config: MatcherConfig | None = None,
) -> list[EvaluationRecord]:
    """Evaluate all labelled queries using the existing unchanged matching pipeline."""
    active_config = config or MatcherConfig()
    references_root = Path(reference_directory)
    queries_root = Path(query_directory)
    records: list[EvaluationRecord] = []
    for query_path in discover_query_images(queries_root):
        expected_product, condition = parse_query_metadata(query_path)
        candidates = match_query_against_references(query_path, references_root, active_config)
        strongest_candidate = candidates[0]
        accepted_candidate = best_accepted_match(candidates)
        evidence_candidate = accepted_candidate or strongest_candidate
        prediction = accepted_candidate.product_label if accepted_candidate else "UNKNOWN"
        records.append(
            EvaluationRecord(
                query_image=_relative_path(query_path, queries_root),
                expected_product=expected_product,
                condition=condition,
                predicted_product=prediction,
                best_candidate=strongest_candidate.product_label,
                reference_image=_relative_path(evidence_candidate.reference_path, references_root),
                good_matches=evidence_candidate.good_match_count,
                ransac_inliers=evidence_candidate.inlier_count,
                inlier_ratio=evidence_candidate.inlier_ratio,
                score=evidence_candidate.score,
                accepted=accepted_candidate is not None,
                correct=prediction == expected_product,
            )
        )
    return records


def summarize_records(records: list[EvaluationRecord]) -> dict[str, int | float]:
    """Calculate overall counts, distinguishing UNKNOWN from accepted mistakes."""
    total_queries = len(records)
    correct_predictions = sum(record.correct for record in records)
    unknown_predictions = sum(record.predicted_product == "UNKNOWN" for record in records)
    accepted_predictions = sum(record.accepted for record in records)
    accepted_but_wrong_predictions = sum(record.accepted and not record.correct for record in records)
    return {
        "total_queries": total_queries,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": total_queries - correct_predictions,
        "unknown_predictions": unknown_predictions,
        "accepted_predictions": accepted_predictions,
        "accepted_but_wrong_predictions": accepted_but_wrong_predictions,
        "overall_accuracy": correct_predictions / total_queries if total_queries else 0.0,
    }


def summarize_by_condition(records: list[EvaluationRecord]) -> dict[str, dict[str, int | float]]:
    """Summarize records by their filename-derived image condition."""
    return _summarize_by(records, lambda record: record.condition)


def summarize_by_product(records: list[EvaluationRecord]) -> dict[str, dict[str, int | float]]:
    """Summarize records by their parent-directory ground-truth label."""
    return _summarize_by(records, lambda record: record.expected_product)


def write_evaluation_csv(records: list[EvaluationRecord], output_path: Path) -> Path:
    """Write portable derived measurements with stable numeric formatting."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "query_image": record.query_image,
                    "expected_product": record.expected_product,
                    "condition": record.condition,
                    "predicted_product": record.predicted_product,
                    "best_candidate": record.best_candidate,
                    "reference_image": record.reference_image,
                    "good_matches": record.good_matches,
                    "ransac_inliers": record.ransac_inliers,
                    "inlier_ratio": f"{record.inlier_ratio:.3f}",
                    "score": f"{record.score:.3f}",
                    "accepted": str(record.accepted).lower(),
                    "correct": str(record.correct).lower(),
                }
            )
    return destination


def _summarize_by(
    records: list[EvaluationRecord], key: Callable[[EvaluationRecord], str]
) -> dict[str, dict[str, int | float]]:
    grouped_records: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        grouped_records.setdefault(key(record), []).append(record)

    return {group: summarize_records(group_records) for group, group_records in sorted(grouped_records.items())}


def _relative_path(path: Path, root: Path) -> str:
    return Path(path).relative_to(root).as_posix()
