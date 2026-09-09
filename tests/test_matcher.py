"""Unit and synthetic integration tests for geometric ORB reference matching."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from beverage_matcher.features import extract_orb_features
from beverage_matcher.matcher import (
    MatcherConfig,
    best_accepted_match,
    filter_ratio_matches,
    knn_match_descriptors,
    match_query_against_references,
    verify_geometric_matches,
)
from beverage_matcher.visualization import save_match_visualization


def _match(query_index: int, train_index: int, distance: float) -> cv2.DMatch:
    return cv2.DMatch(_queryIdx=query_index, _trainIdx=train_index, _imgIdx=0, _distance=distance)


def _structured_image(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    for _ in range(30):
        center = tuple(rng.integers(20, 300, size=2))
        radius = int(rng.integers(3, 12))
        cv2.circle(image, center, radius, (255, 255, 255), 1)
    cv2.rectangle(image, (35, 35), (125, 115), (255, 255, 255), 3)
    cv2.putText(image, "MATCH", (65, 245), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image


def _unrelated_image() -> np.ndarray:
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    for row in range(0, 320, 32):
        for column in range(0, 320, 32):
            if (row // 32 + column // 32) % 2 == 0:
                image[row : row + 32, column : column + 32] = 255
    return image


def test_ratio_filter_keeps_clear_first_neighbour() -> None:
    matches = [[_match(0, 0, 10), _match(0, 1, 20)]]

    assert filter_ratio_matches(matches) == [matches[0][0]]


def test_ratio_filter_rejects_ambiguous_neighbours() -> None:
    matches = [[_match(0, 0, 15), _match(0, 1, 20)]]

    assert filter_ratio_matches(matches) == []


def test_ratio_filter_ignores_single_neighbour() -> None:
    assert filter_ratio_matches([[_match(0, 0, 10)]]) == []


@pytest.mark.parametrize("ratio_threshold", [0.0, 1.0, -0.1])
def test_ratio_filter_rejects_invalid_threshold(ratio_threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        filter_ratio_matches([], ratio_threshold)


def test_geometric_verification_detects_inliers_after_known_transform() -> None:
    source = _structured_image()
    transform = cv2.getRotationMatrix2D((160, 160), 7, 1.0)
    transform[:, 2] += (12, -8)
    transformed = cv2.warpAffine(source, transform, (320, 320))
    source_keypoints, source_descriptors = extract_orb_features(source, max_features=1000)
    transformed_keypoints, transformed_descriptors = extract_orb_features(transformed, max_features=1000)

    good_matches = filter_ratio_matches(
        knn_match_descriptors(source_descriptors, transformed_descriptors)
    )
    homography, _, inlier_count, _ = verify_geometric_matches(
        source_keypoints, transformed_keypoints, good_matches
    )

    assert len(good_matches) >= 4
    assert homography is not None
    assert inlier_count >= 4


def test_unrelated_images_are_not_confidently_accepted(tmp_path: Path) -> None:
    references = tmp_path / "references" / "first_product"
    references.mkdir(parents=True)
    query_path = tmp_path / "query.png"
    reference_path = references / "reference.png"
    assert cv2.imwrite(str(query_path), _structured_image(seed=3))
    assert cv2.imwrite(str(reference_path), _unrelated_image())

    candidates = match_query_against_references(query_path, tmp_path / "references", MatcherConfig())

    assert len(candidates) == 1
    assert best_accepted_match(candidates) is None


def test_verified_matches_can_be_saved_without_a_gui(tmp_path: Path) -> None:
    reference_image = _structured_image()
    transform = cv2.getRotationMatrix2D((160, 160), 5, 1.0)
    transform[:, 2] += (8, -5)
    query_image = cv2.warpAffine(reference_image, transform, (320, 320))
    reference_directory = tmp_path / "references" / "synthetic_product"
    reference_directory.mkdir(parents=True)
    query_path = tmp_path / "query.png"
    reference_path = reference_directory / "reference.png"
    assert cv2.imwrite(str(query_path), query_image)
    assert cv2.imwrite(str(reference_path), reference_image)

    candidates = match_query_against_references(
        query_path,
        tmp_path / "references",
        MatcherConfig(max_features=1000, min_inliers=4, min_inlier_ratio=0.2),
    )
    accepted = best_accepted_match(candidates)

    assert accepted is not None
    output_path = save_match_visualization(
        query_image, reference_image, accepted, tmp_path / "verified_matches.jpg"
    )
    assert output_path.is_file()
    assert cv2.imread(str(output_path)) is not None
