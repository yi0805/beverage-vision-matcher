# beverage-vision-matcher

A small computer-vision experiment exploring local-feature-based beverage product
matching under changes in viewpoint, lighting, and occlusion. It is intended as a
clear, testable portfolio exercise in Python and classical computer vision rather
than a production recognition system.

## Current status

Task 001 implements only the foundation: validated image loading and ORB feature
extraction for colour and grayscale images. Product matching, descriptor matching,
geometric verification (RANSAC), scoring, visualisation, and measured evaluation
are not implemented yet.

## Planned direction

Later tasks will investigate ORB binary descriptors, Hamming-distance matching,
ratio filtering, and geometric verification to compare query images with known
product references. Any performance or robustness conclusions will be reported
only after tests using a real, documented image set.
