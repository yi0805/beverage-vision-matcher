"""Saved visualisations for geometrically verified feature correspondences."""

from pathlib import Path

import cv2
import numpy as np

from .matcher import MatchResult


def save_match_visualization(
    query_image: np.ndarray,
    reference_image: np.ndarray,
    result: MatchResult,
    output_path: Path,
) -> Path:
    """Save a feature-match image, preferring RANSAC inliers when available."""
    selected_matches = list(result.good_matches)
    if result.homography is not None and result.inlier_mask:
        selected_matches = [
            match for match, is_inlier in zip(result.good_matches, result.inlier_mask) if is_inlier
        ]
    if not selected_matches:
        raise ValueError("No feature matches are available to visualise")

    visualization = cv2.drawMatches(
        query_image,
        list(result.query_keypoints),
        reference_image,
        list(result.reference_keypoints),
        selected_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), visualization):
        raise ValueError(f"Could not save match visualisation: {destination}")
    return destination
