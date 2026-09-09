# beverage-vision-matcher

A small computer-vision experiment exploring local-feature-based beverage product
matching under changes in viewpoint, lighting, and occlusion. It is intended as a
clear, testable portfolio exercise in Python and classical computer vision rather
than a production recognition system.

## Current status

Task 002 implements the baseline reference-image matching experiment: validated
image loading, ORB features, Hamming KNN descriptor matching, ratio filtering,
RANSAC homography verification, geometric inlier scoring, `UNKNOWN` rejection,
and an optional saved match visualisation. Real beverage evaluation has not been
performed, so there is no accuracy number and the thresholds below are provisional.

## Planned direction

The pipeline is: ORB features → Hamming KNN matches → ratio filtering → RANSAC
homography → geometric inliers → score/rank → accepted product or `UNKNOWN`.
ORB produces binary local descriptors that work directly with Hamming-distance
matching. Ratio filtering removes ambiguous correspondences, while RANSAC rejects
matches that do not support a shared geometric transform.

The baseline score is `inlier_count * inlier_ratio`; a candidate must have at
least 8 RANSAC inliers and an inlier ratio of at least 0.30. These are starting
parameters, not calibrated claims. Use `python scripts/match.py --help` for the
single-query CLI. Later evaluation will use real, documented images before making
any performance conclusion.

## Current limitations

Curved containers do not perfectly obey one planar homography. Glare, severe
viewpoint changes, low-texture packaging, visually similar designs, and strong
occlusion can reduce useful geometric evidence. Feature detection, correspondence,
and geometric consistency are also concepts used in broader multi-view geometry
pipelines such as structure from motion; this project does not implement SfM or
3D reconstruction.
