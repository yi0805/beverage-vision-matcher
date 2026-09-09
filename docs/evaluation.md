# Evaluation template

Evaluation has not been implemented or measured in Task 001. A later experiment
should record one row per query image with these fields:

| Query image | Expected product | Predicted product | Condition | Good descriptor matches | RANSAC inliers | Inlier ratio | Score | Outcome |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| _to be collected_ |  |  |  |  |  |  |  |  |

Planned conditions are `normal`, `viewpoint change`, `low light`, `partial
occlusion`, and `glare / reflection`. Later v1 evaluation should infer conditions
from a simple filename or folder convention; CSV metadata support is not part of
Task 001. No measurements or conclusions have been collected yet.
