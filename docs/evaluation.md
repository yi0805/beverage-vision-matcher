# Evaluation

## Dataset

The local experiment contained 3 beverage products: `goodbuzz`, `mo`, and
`redbull`. Each had one reference photograph and five query photographs, for 3
references and 15 queries total. Query conditions were `normal`,
`angle`, `low_light`, `occlusion`, and `glare`.

Raw photographs are local and intentionally excluded from the public repository.
The accompanying [`results/evaluation.csv`](../results/evaluation.csv) contains
only derived matching measurements. The query filenames were normalised locally to
the documented `<product>__<condition>.<extension>` convention before evaluation.

## Baseline configuration

- ORB maximum features: 500
- KNN ratio threshold: 0.75
- RANSAC reprojection threshold: 5.0 pixels
- Minimum inliers: 8
- Minimum inlier ratio: 0.30
- Score: `inlier_count * inlier_ratio`

These pre-existing Task 002 baseline values were used unchanged. They were not
tuned using this dataset.

## Method

Each query is processed as:

```text
query → ORB → Hamming KNN matching → ratio filtering → RANSAC → score → thresholded prediction
```

`UNKNOWN` is recorded when no candidate passes both minimum geometric-evidence
thresholds. For known-product queries, `UNKNOWN` counts as incorrect. The CSV's
`evidence_candidate` is the accepted candidate when a prediction is accepted;
otherwise it is the strongest candidate, with all diagnostic metrics referring to
that same candidate.

## Results

Overall, the baseline correctly identified 11 of 15 queries (73.3%). There was
one accepted wrong-product result and three `UNKNOWN` results.

| Condition | N | Correct | Wrong accepted | Unknown | Accuracy |
| --- | --: | --: | --: | --: | --: |
| angle | 3 | 3 | 0 | 0 | 100.0% |
| glare | 3 | 1 | 0 | 2 | 33.3% |
| low_light | 3 | 3 | 0 | 0 | 100.0% |
| normal | 3 | 3 | 0 | 0 | 100.0% |
| occlusion | 3 | 1 | 1 | 1 | 33.3% |

| Product | N | Correct | Wrong accepted | Unknown | Accuracy |
| --- | --: | --: | --: | --: | --: |
| goodbuzz | 5 | 4 | 0 | 1 | 80.0% |
| mo | 5 | 3 | 1 | 1 | 60.0% |
| redbull | 5 | 4 | 0 | 1 | 80.0% |

## Failure cases

| Query | Expected | Prediction | Evidence candidate | Good matches | Inliers | Ratio | Score |
| --- | --- | --- | --- | --: | --: | --: | --: |
| `goodbuzz__glare.jpg` | goodbuzz | UNKNOWN | goodbuzz | 10 | 5 | 0.500 | 2.500 |
| `mo__glare.jpg` | mo | UNKNOWN | mo | 20 | 6 | 0.300 | 1.800 |
| `mo__occlusion.jpg` | mo | redbull | redbull | 12 | 9 | 0.750 | 6.750 |
| `redbull__occlusion.jpg` | redbull | UNKNOWN | mo | 6 | 4 | 0.667 | 2.667 |

The two glare failures and one occlusion failure had fewer than the required eight
inliers, consistent with insufficient geometric evidence. The `mo` occlusion image
was an accepted wrong-product result with strong measured geometry for `redbull`.
This suggests that occlusion can leave locally consistent but misleading evidence;
the small dataset alone cannot establish its cause.

## Observations and limitations

This experiment suggests the baseline was more successful on the collected normal,
angle, and low-light images than on the collected glare and occlusion images. The
dataset is very small, includes only three products and one reference per product,
and was not designed for statistical significance. These numbers must not be read
as general product-recognition accuracy.

Planar homography is only an approximation for curved cans and bottles. Package
appearance, glare, viewpoint, occlusion, and keypoint texture may affect ORB
correspondence quality. Potential future work includes multiple references per
product, a larger independent test set, stronger unknown-product tests, separate
calibration and test sets, and a controlled SIFT comparison.
