"""Image loading and ORB feature extraction utilities."""

from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load an image from *path* or raise a clear error when it is unavailable."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode image file: {image_path}")
    return image


def extract_orb_features(
    image: np.ndarray, max_features: int = 500
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Return ORB keypoints and binary descriptors for a grayscale or colour image.

    A valid but featureless image returns an empty keypoint list and ``None`` for
    descriptors. The input array is not modified.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale or 3D colour array")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("colour image must have 3 (BGR) or 4 (BGRA) channels")
    if isinstance(max_features, bool) or not isinstance(max_features, int):
        raise TypeError("max_features must be an integer")
    if max_features <= 0:
        raise ValueError("max_features must be positive")

    grayscale = _to_grayscale(image)
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(grayscale, None)
    return list(keypoints), descriptors


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a supported OpenCV image to grayscale without changing its caller."""
    if image.ndim == 2:
        return image
    conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, conversion)
