"""Explainable scoring and acceptance checks for geometric feature matches."""


def calculate_inlier_ratio(good_match_count: int, inlier_count: int) -> float:
    """Return the fraction of descriptor matches supported by RANSAC geometry."""
    _validate_match_counts(good_match_count, inlier_count)
    if good_match_count == 0:
        return 0.0
    return inlier_count / good_match_count


def calculate_score(good_match_count: int, inlier_count: int) -> float:
    """Score a candidate as ``inlier_count * inlier_ratio``.

    The rule rewards more geometrically verified matches while reducing the value
    of candidates whose descriptor matches are mostly geometrically inconsistent.
    """
    return inlier_count * calculate_inlier_ratio(good_match_count, inlier_count)


def passes_confidence_threshold(
    inlier_count: int,
    inlier_ratio: float,
    min_inliers: int = 8,
    min_inlier_ratio: float = 0.30,
) -> bool:
    """Return whether a candidate satisfies both baseline acceptance thresholds."""
    if inlier_count < 0:
        raise ValueError("inlier_count must not be negative")
    if min_inliers < 0:
        raise ValueError("min_inliers must not be negative")
    if not 0.0 <= inlier_ratio <= 1.0:
        raise ValueError("inlier_ratio must be between 0 and 1")
    if not 0.0 <= min_inlier_ratio <= 1.0:
        raise ValueError("min_inlier_ratio must be between 0 and 1")
    return inlier_count >= min_inliers and inlier_ratio >= min_inlier_ratio


def _validate_match_counts(good_match_count: int, inlier_count: int) -> None:
    if good_match_count < 0 or inlier_count < 0:
        raise ValueError("match counts must not be negative")
    if inlier_count > good_match_count:
        raise ValueError("inlier_count cannot exceed good_match_count")
