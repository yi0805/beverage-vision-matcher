"""Focused tests for the Task 001 image and ORB feature foundation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from beverage_matcher.features import extract_orb_features, load_image


def _structured_image() -> np.ndarray:
    image = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (100, 100), (255, 255, 255), 3)
    cv2.circle(image, (170, 70), 35, (255, 255, 255), 3)
    cv2.line(image, (25, 200), (210, 135), (255, 255, 255), 3)
    cv2.putText(image, "ORB", (65, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image


def test_load_image_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_image(tmp_path / "missing.jpg")


def test_load_image_rejects_undecodable_file(tmp_path: Path) -> None:
    invalid_file = tmp_path / "not-an-image.jpg"
    invalid_file.write_text("not image data", encoding="utf-8")

    with pytest.raises(ValueError, match="could not decode"):
        load_image(invalid_file)


def test_load_image_returns_numpy_array(tmp_path: Path) -> None:
    image_path = tmp_path / "structured.png"
    assert cv2.imwrite(str(image_path), _structured_image())

    loaded = load_image(image_path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (240, 240, 3)


def test_orb_detects_features_in_structured_colour_image() -> None:
    keypoints, descriptors = extract_orb_features(_structured_image())

    assert keypoints
    assert descriptors is not None
    assert descriptors.dtype == np.uint8


def test_featureless_image_is_a_valid_outcome() -> None:
    blank = np.zeros((120, 120), dtype=np.uint8)

    keypoints, descriptors = extract_orb_features(blank)

    assert keypoints == []
    assert descriptors is None


def test_orb_supports_grayscale_input() -> None:
    grayscale = cv2.cvtColor(_structured_image(), cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = extract_orb_features(grayscale)

    assert keypoints
    assert descriptors is not None


@pytest.mark.parametrize("max_features", [0, -1])
def test_orb_rejects_non_positive_max_features(max_features: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        extract_orb_features(_structured_image(), max_features=max_features)


@pytest.mark.parametrize(
    "invalid_image",
    [None, np.array([], dtype=np.uint8), np.zeros((10, 10, 2), dtype=np.uint8)],
)
def test_orb_rejects_invalid_image_input(invalid_image: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        extract_orb_features(invalid_image)  # type: ignore[arg-type]
