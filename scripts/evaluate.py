"""Evaluate labelled beverage query images with the unchanged baseline matcher."""

from __future__ import annotations

import argparse
from pathlib import Path

from beverage_matcher.evaluation import (
    evaluate_queries,
    summarize_by_condition,
    summarize_by_product,
    summarize_records,
    write_evaluation_csv,
)
from beverage_matcher.matcher import MatcherConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally small batch-evaluation command-line interface."""
    parser = argparse.ArgumentParser(description="Evaluate labelled beverage query images.")
    parser.add_argument("--references", type=Path, required=True, help="Reference image directory")
    parser.add_argument("--queries", type=Path, required=True, help="Labelled query image directory")
    parser.add_argument(
        "--output", type=Path, default=Path("results/evaluation.csv"), help="CSV output path"
    )
    parser.add_argument("--max-features", type=int, default=500, help="Maximum ORB features per image")
    parser.add_argument("--ratio-threshold", type=float, default=0.75, help="KNN ratio-test threshold")
    parser.add_argument(
        "--ransac-threshold", type=float, default=5.0, help="RANSAC reprojection threshold in pixels"
    )
    parser.add_argument("--min-inliers", type=int, default=8, help="Minimum RANSAC inliers")
    parser.add_argument(
        "--min-inlier-ratio", type=float, default=0.30, help="Minimum RANSAC inlier ratio"
    )
    return parser


def main() -> int:
    """Run the baseline evaluation once and print its measured summaries."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = MatcherConfig(
            max_features=args.max_features,
            ratio_threshold=args.ratio_threshold,
            ransac_reprojection_threshold=args.ransac_threshold,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
        )
        records = evaluate_queries(args.references, args.queries, config)
        output_path = write_evaluation_csv(records, args.output)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    overall = summarize_records(records)
    by_condition = summarize_by_condition(records)
    by_product = summarize_by_product(records)
    print("Dataset\n-------")
    print(f"Products: {len(by_product)}")
    print(f"Queries: {overall['total_queries']}")
    print("\nBaseline parameters\n-------------------")
    print(f"ORB max features: {config.max_features}")
    print(f"Ratio threshold: {config.ratio_threshold}")
    print(f"RANSAC threshold: {config.ransac_reprojection_threshold:.1f} px")
    print(f"Minimum inliers: {config.min_inliers}")
    print(f"Minimum inlier ratio: {config.min_inlier_ratio:.2f}")
    print("\nOverall results\n---------------")
    print(f"Correct: {overall['correct_predictions']}")
    print(f"Wrong accepted: {overall['accepted_but_wrong_predictions']}")
    print(f"Unknown: {overall['unknown_predictions']}")
    print(f"Accuracy: {overall['overall_accuracy']:.1%}")
    _print_breakdown("By condition", by_condition)
    _print_breakdown("By product", by_product)
    failures = [record for record in records if not record.correct]
    print("\nFailures\n--------")
    if not failures:
        print("None")
    for record in failures:
        print(
            f"{record.query_image}: expected={record.expected_product}, "
            f"predicted={record.predicted_product}, best={record.best_candidate}, "
            f"good={record.good_matches}, inliers={record.ransac_inliers}, "
            f"ratio={record.inlier_ratio:.3f}, score={record.score:.3f}"
        )
    print(f"\nResults written to:\n{output_path}")
    return 0


def _print_breakdown(
    title: str, summary: dict[str, dict[str, int | float]]
) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(f"{'Group':<16} {'N':>3} {'Correct':>8} {'Wrong':>6} {'Unknown':>8} {'Accuracy':>9}")
    for group, values in summary.items():
        wrong = int(values["incorrect_predictions"]) - int(values["unknown_predictions"])
        print(
            f"{group:<16} {values['total_queries']:>3} {values['correct_predictions']:>8} "
            f"{wrong:>6} {values['unknown_predictions']:>8} {values['overall_accuracy']:>8.1%}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
