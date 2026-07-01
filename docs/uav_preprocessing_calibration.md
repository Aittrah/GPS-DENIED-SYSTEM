# UAV Query Calibration

The FAISS/UAV preprocessing path now supports optional lens undistortion for
desktop and offline workflows. This path is disabled by default and does not
change the packaged runtime or ROS launch behavior.

## Calibration File Format

```yaml
camera_matrix:
  - [554.25, 0.0, 320.0]
  - [0.0, 554.25, 240.0]
  - [0.0, 0.0, 1.0]
distortion_coefficients: [0.10, -0.02, 0.0, 0.0, 0.0]
image_size: [640, 480]
```

`image_size` is optional, but it should be included when the calibration was
produced at a different resolution than the query frames you plan to use.

## Programmatic Use

```python
from vns.database.faiss_database import FAISSDatabase

db = FAISSDatabase(
    query_calibration_path="camera_calibration.yaml",
    undistort_queries=True,
)
```

If `undistort_queries` is left at its default `False`, the query path behaves
exactly as before.
