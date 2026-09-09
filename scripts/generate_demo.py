"""Generate a public synthetic feature-matching example for the README."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from beverage_matcher.features import extract_orb_features
from beverage_matcher.matcher import (
    MatcherConfig,
    MatchResult,
    filter_ratio_matches,
    knn_match_descriptors,
    verify_geometric_matches,
)
from beverage_matcher.scoring import calculate_score, passes_confidence_threshold
from beverage_matcher.visualization import save_match_visualization


def generate_demo(output_path: Path = Path("docs/example_match.jpg")) -> Path:
    """Create a deterministic transformed-image match visualisation for the README."""
    # This display-only limit keeps individual verified correspondences legible.
    config = MatcherConfig(max_features=80)
    reference_image = _structured_image()
    query_image = _transform_image(reference_image)
    query_keypoints, query_descriptors = extract_orb_features(query_image, config.max_features)
    reference_keypoints, reference_descriptors = extract_orb_features(
        reference_image, config.max_features
    )
    good_matches = filter_ratio_matches(
        knn_match_descriptors(query_descriptors, reference_descriptors), config.ratio_threshold
    )
    homography, inlier_mask, inlier_count, inlier_ratio = verify_geometric_matches(
        query_keypoints,
        reference_keypoints,
        good_matches,
        config.ransac_reprojection_threshold,
    )
    if homography is None or inlier_count < 4:
        raise RuntimeError("Synthetic demo did not produce enough geometric inliers")

    result = MatchResult(
        product_label="synthetic_reference",
        reference_path=Path("synthetic_reference.png"),
        good_matches=tuple(good_matches),
        query_keypoints=tuple(query_keypoints),
        reference_keypoints=tuple(reference_keypoints),
        homography=homography,
        inlier_mask=inlier_mask,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        score=calculate_score(len(good_matches), inlier_count),
        is_accepted=passes_confidence_threshold(
            inlier_count,
            inlier_ratio,
            config.min_inliers,
            config.min_inlier_ratio,
        ),
    )
    return save_match_visualization(query_image, reference_image, result, output_path)


def _structured_image() -> np.ndarray:
    """Build a deterministic high-contrast image with varied local structure."""
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(45):
        center = tuple(int(value) for value in rng.integers((25, 25), (455, 335)))
        radius = int(rng.integers(3, 11))
        cv2.circle(image, center, radius, (255, 255, 255), 1)
    cv2.rectangle(image, (35, 45), (165, 165), (255, 255, 255), 3)
    cv2.rectangle(image, (285, 190), (430, 320), (255, 255, 255), 2)
    cv2.line(image, (55, 310), (420, 55), (255, 255, 255), 2)
    cv2.putText(image, "ORB", (190, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(image, "SYNTHETIC", (115, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return image


def _transform_image(image: np.ndarray) -> np.ndarray:
    """Apply a modest deterministic rotation and translation to the synthetic image."""
    transform = cv2.getRotationMatrix2D((240, 180), 7, 1.0)
    transform[:, 2] += (14, -10)
    return cv2.warpAffine(image, transform, (480, 360))


if __name__ == "__main__":
    saved_path = generate_demo()
    print(f"Synthetic demo saved to: {saved_path}")
