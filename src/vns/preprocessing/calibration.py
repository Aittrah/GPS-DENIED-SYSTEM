from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraCalibration:
    """Camera intrinsics and distortion parameters for query undistortion."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    calibration_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        camera_matrix = np.asarray(self.camera_matrix, dtype=np.float32)
        if camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3).")

        distortion = np.asarray(
            self.distortion_coefficients,
            dtype=np.float32,
        ).reshape(-1)
        if distortion.size not in {4, 5, 8}:
            raise ValueError(
                "distortion_coefficients must contain 4, 5, or 8 values."
            )

        size = self.calibration_size
        if size is not None:
            width, height = int(size[0]), int(size[1])
            if width <= 0 or height <= 0:
                raise ValueError("calibration_size must contain positive integers.")
            size = (width, height)

        object.__setattr__(self, "camera_matrix", camera_matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)
        object.__setattr__(self, "calibration_size", size)


def _reshape_camera_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape == (3, 3):
        return matrix
    if matrix.size == 9:
        return matrix.reshape(3, 3)
    raise ValueError("camera_matrix must be a 3x3 matrix or a flat 9-value list.")


def _calibration_size_from_mapping(data: dict[str, Any]) -> tuple[int, int] | None:
    image_size = data.get("image_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])

    calibration_size = data.get("calibration_size")
    if isinstance(calibration_size, (list, tuple)) and len(calibration_size) == 2:
        return int(calibration_size[0]), int(calibration_size[1])

    width = data.get("width")
    height = data.get("height")
    if width is not None and height is not None:
        return int(width), int(height)

    return None


def calibration_from_mapping(data: dict[str, Any]) -> CameraCalibration:
    """Build a :class:`CameraCalibration` from a mapping."""
    if "camera_matrix" in data:
        camera_matrix = _reshape_camera_matrix(data["camera_matrix"])
    elif "camera" in data and isinstance(data["camera"], dict):
        camera = data["camera"]
        camera_matrix = np.array(
            [
                [float(camera["fx"]), 0.0, float(camera["cx"])],
                [0.0, float(camera["fy"]), float(camera["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError("calibration mapping must define camera_matrix or camera fx/fy/cx/cy values.")

    distortion = data.get("distortion_coefficients")
    if distortion is None:
        distortion = data.get("distortion")
    if distortion is None and "camera" in data and isinstance(data["camera"], dict):
        distortion = data["camera"].get("distortion")
    if distortion is None:
        raise ValueError(
            "calibration mapping must define distortion_coefficients or distortion."
        )

    return CameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        calibration_size=_calibration_size_from_mapping(data),
    )


def load_camera_calibration(path_like: str | Path) -> CameraCalibration:
    """Load camera calibration data from a YAML or JSON file."""
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Calibration path is not a file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle)

    if not isinstance(raw_data, dict):
        raise ValueError("Calibration file must contain a mapping.")

    return calibration_from_mapping(raw_data)
