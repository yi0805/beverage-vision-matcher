"""Tests for simple geometric match scoring and UNKNOWN acceptance logic."""

import pytest

from beverage_matcher.scoring import (
    calculate_inlier_ratio,
    calculate_score,
    passes_confidence_threshold,
)


def test_zero_matches_have_zero_ratio_and_score() -> None:
    assert calculate_inlier_ratio(0, 0) == 0.0
    assert calculate_score(0, 0) == 0.0


def test_more_inliers_increase_score_at_same_ratio() -> None:
    assert calculate_score(10, 5) < calculate_score(20, 10)


def test_inlier_ratio_is_calculated_from_good_matches() -> None:
    assert calculate_inlier_ratio(25, 20) == 0.8


def test_impossible_match_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        calculate_score(3, 4)


def test_confidence_requires_both_thresholds() -> None:
    assert passes_confidence_threshold(8, 0.30)
    assert not passes_confidence_threshold(8, 0.29)
    assert not passes_confidence_threshold(7, 0.90)
