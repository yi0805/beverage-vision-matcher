"""ORB descriptor matching and geometric verification for reference images."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .features import extract_orb_features, load_image
from .scoring import calculate_inlier_ratio, calculate_score, passes_confidence_threshold

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class MatcherConfig:
    """Visible, provisional baseline parameters for the matching experiment."""

    max_features: int = 500
    ratio_threshold: float = 0.75
    ransac_reprojection_threshold: float = 5.0
    min_inliers: int = 8
    min_inlier_ratio: float = 0.30

    def __post_init__(self) -> None:
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        _validate_ratio_threshold(self.ratio_threshold)
        if self.ransac_reprojection_threshold <= 0:
            raise ValueError("ransac_reprojection_threshold must be positive")
        if self.min_inliers < 0:
            raise ValueError("min_inliers must not be negative")
        if not 0.0 <= self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be between 0 and 1")


@dataclass(frozen=True)
class MatchResult:
    """The descriptor and geometric evidence for one reference image candidate."""

    product_label: str
    reference_path: Path
    good_matches: tuple[cv2.DMatch, ...]
    query_keypoints: tuple[cv2.KeyPoint, ...]
    reference_keypoints: tuple[cv2.KeyPoint, ...]
    homography: np.ndarray | None
    inlier_mask: tuple[bool, ...]
    inlier_count: int
    inlier_ratio: float
    score: float
    is_accepted: bool

    @property
    def good_match_count(self) -> int:
        """Return the count of ratio-filtered descriptor matches."""
        return len(self.good_matches)


def discover_reference_images(reference_directory: Path) -> list[tuple[str, Path]]:
    """Discover supported images, using each image's parent directory as its label."""
    directory = Path(reference_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {directory}")

    references = [
        (image_path.parent.name, image_path)
        for image_path in sorted(directory.rglob("*"))
        if image_path.is_file() and image_path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not references:
        raise ValueError(f"No reference images found in: {directory}")
    return references


def knn_match_descriptors(
    query_descriptors: np.ndarray | None, reference_descriptors: np.ndarray | None
) -> list[list[cv2.DMatch]]:
    """Match ORB descriptors with a Hamming BFMatcher, retaining two neighbours."""
    if query_descriptors is None or reference_descriptors is None:
        return []
    if len(query_descriptors) == 0 or len(reference_descriptors) == 0:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return [list(neighbours) for neighbours in matcher.knnMatch(query_descriptors, reference_descriptors, k=2)]


def filter_ratio_matches(
    knn_matches: list[list[cv2.DMatch]], ratio_threshold: float = 0.75
) -> list[cv2.DMatch]:
    """Keep unambiguous first-neighbour matches using Lowe's ratio test."""
    _validate_ratio_threshold(ratio_threshold)
    good_matches: list[cv2.DMatch] = []
    for neighbours in knn_matches:
        if len(neighbours) < 2:
            continue
        best_match, second_match = neighbours[0], neighbours[1]
        if best_match.distance < ratio_threshold * second_match.distance:
            good_matches.append(best_match)
    return good_matches


def verify_geometric_matches(
    query_keypoints: list[cv2.KeyPoint],
    reference_keypoints: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    ransac_reprojection_threshold: float = 5.0,
) -> tuple[np.ndarray | None, tuple[bool, ...], int, float]:
    """Estimate a query-to-reference homography and return its RANSAC inliers."""
    if ransac_reprojection_threshold <= 0:
        raise ValueError("ransac_reprojection_threshold must be positive")
    good_match_count = len(good_matches)
    if good_match_count < 4:
        return None, (), 0, 0.0

    query_points = np.float32([query_keypoints[match.queryIdx].pt for match in good_matches])
    reference_points = np.float32([reference_keypoints[match.trainIdx].pt for match in good_matches])
    try:
        homography, mask = cv2.findHomography(
            query_points,
            reference_points,
            cv2.RANSAC,
            ransac_reprojection_threshold,
        )
    except cv2.error:
        return None, (), 0, 0.0

    if homography is None or mask is None:
        return None, (), 0, 0.0

    inlier_mask = tuple(bool(value) for value in mask.ravel())
    inlier_count = sum(inlier_mask)
    return (
        homography,
        inlier_mask,
        inlier_count,
        calculate_inlier_ratio(good_match_count, inlier_count),
    )


def match_query_against_references(
    query_path: Path, reference_directory: Path, config: MatcherConfig | None = None
) -> list[MatchResult]:
    """Rank reference-image candidates for one query, extracting query features once."""
    if config is None:
        config = MatcherConfig()
    query_image = load_image(query_path)
    query_keypoints, query_descriptors = extract_orb_features(query_image, config.max_features)
    candidates = [
        _match_reference(
            product_label,
            reference_path,
            query_keypoints,
            query_descriptors,
            config,
        )
        for product_label, reference_path in discover_reference_images(reference_directory)
    ]
    return sorted(candidates, key=lambda result: result.score, reverse=True)


def best_accepted_match(candidates: list[MatchResult]) -> MatchResult | None:
    """Return the strongest accepted candidate, or ``None`` to represent UNKNOWN."""
    return next((candidate for candidate in candidates if candidate.is_accepted), None)


def _match_reference(
    product_label: str,
    reference_path: Path,
    query_keypoints: list[cv2.KeyPoint],
    query_descriptors: np.ndarray | None,
    config: MatcherConfig,
) -> MatchResult:
    reference_image = load_image(reference_path)
    reference_keypoints, reference_descriptors = extract_orb_features(
        reference_image, config.max_features
    )
    knn_matches = knn_match_descriptors(query_descriptors, reference_descriptors)
    good_matches = filter_ratio_matches(knn_matches, config.ratio_threshold)
    homography, inlier_mask, inlier_count, inlier_ratio = verify_geometric_matches(
        query_keypoints,
        reference_keypoints,
        good_matches,
        config.ransac_reprojection_threshold,
    )
    score = calculate_score(len(good_matches), inlier_count)
    accepted = passes_confidence_threshold(
        inlier_count,
        inlier_ratio,
        config.min_inliers,
        config.min_inlier_ratio,
    )
    return MatchResult(
        product_label=product_label,
        reference_path=reference_path,
        good_matches=tuple(good_matches),
        query_keypoints=tuple(query_keypoints),
        reference_keypoints=tuple(reference_keypoints),
        homography=homography,
        inlier_mask=inlier_mask,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        score=score,
        is_accepted=accepted,
    )


def _validate_ratio_threshold(ratio_threshold: float) -> None:
    if not 0.0 < ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be between 0 and 1")
