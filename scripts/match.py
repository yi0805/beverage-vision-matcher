"""Command-line entry point for one-query reference-image matching."""

from __future__ import annotations

import argparse
from pathlib import Path

from beverage_matcher.features import load_image
from beverage_matcher.matcher import MatcherConfig, best_accepted_match, match_query_against_references
from beverage_matcher.visualization import save_match_visualization


def build_parser() -> argparse.ArgumentParser:
    """Create the small command-line interface for the baseline experiment."""
    parser = argparse.ArgumentParser(description="Match one image against beverage references.")
    parser.add_argument("--references", type=Path, required=True, help="Reference image directory")
    parser.add_argument("--query", type=Path, required=True, help="Query image path")
    parser.add_argument("--visualize", action="store_true", help="Save verified matches under results/")
    parser.add_argument("--max-features", type=int, default=500, help="Maximum ORB features per image")
    parser.add_argument("--ratio-threshold", type=float, default=0.75, help="KNN ratio-test threshold")
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold in pixels",
    )
    parser.add_argument("--min-inliers", type=int, default=8, help="Minimum RANSAC inliers")
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.30,
        help="Minimum RANSAC inlier ratio",
    )
    return parser


def main() -> int:
    """Run matching and print only statistics measured for the supplied images."""
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
        candidates = match_query_against_references(args.query, args.references, config)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    best_candidate = candidates[0]
    accepted = best_accepted_match(candidates)
    print(f"Query: {args.query.name}")
    print(f"Prediction: {accepted.product_label if accepted else 'UNKNOWN'}")
    if accepted:
        print(f"Reference: {accepted.reference_path}")
        result_to_report = accepted
    else:
        print(f"Best candidate: {best_candidate.product_label}")
        result_to_report = best_candidate
    print(f"Good matches: {result_to_report.good_match_count}")
    print(f"RANSAC inliers: {result_to_report.inlier_count}")
    print(f"Inlier ratio: {result_to_report.inlier_ratio:.3f}")
    print(f"Score: {result_to_report.score:.3f}")

    if not accepted:
        print("Reason: insufficient geometric evidence")
    elif args.visualize:
        output_path = Path("results") / f"{args.query.stem}__{accepted.product_label}_matches.jpg"
        saved_path = save_match_visualization(
            load_image(args.query), load_image(accepted.reference_path), accepted, output_path
        )
        print(f"Visualisation: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
