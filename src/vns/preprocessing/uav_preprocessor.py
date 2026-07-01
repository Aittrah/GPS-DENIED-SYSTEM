from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .calibration import CameraCalibration
from .satellite_preprocessor import MAX_SHORT_SIDE

logger = logging.getLogger("vns.preprocessing.uav")


class UAVPreprocessor:
    """
    Pre-processes a single UAV camera frame before patch generation.
    Mirrors the satellite pipeline so descriptors are comparable.
    """

    TARGET_SIZE = (MAX_SHORT_SIDE, MAX_SHORT_SIDE)

    def __init__(
        self,
        *,
        calibration: CameraCalibration | None = None,
        undistort: bool = False,
    ) -> None:
        self._calibration = calibration
        self._undistort = bool(undistort)

    def preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return None
        if self._undistort:
            frame = self._maybe_undistort(frame)
        # Resize to same scale as satellite patches
        h, w = frame.shape[:2]
        short = min(h, w)
        if short > MAX_SHORT_SIDE:
            scale = MAX_SHORT_SIDE / short
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        frame = self._apply_clahe(frame)
        return frame

    def _maybe_undistort(self, frame: np.ndarray) -> np.ndarray:
        calibration = self._calibration
        if calibration is None:
            return frame
        if not np.any(calibration.distortion_coefficients):
            return frame

        camera_matrix = self._scaled_camera_matrix(
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )
        try:
            return cv2.undistort(
                frame,
                camera_matrix,
                calibration.distortion_coefficients,
            )
        except cv2.error as exc:
            logger.warning("Failed to undistort UAV frame: %s", exc)
            return frame

    def _scaled_camera_matrix(
        self,
        *,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        calibration = self._calibration
        if calibration is None:
            raise ValueError("Calibration is required to scale the camera matrix.")

        camera_matrix = calibration.camera_matrix.copy()
        if calibration.calibration_size is None:
            return camera_matrix

        calib_width, calib_height = calibration.calibration_size
        if calib_width <= 0 or calib_height <= 0:
            return camera_matrix
        if (frame_width, frame_height) == calibration.calibration_size:
            return camera_matrix

        scale_x = frame_width / calib_width
        scale_y = frame_height / calib_height
        camera_matrix[0, 0] *= scale_x
        camera_matrix[1, 1] *= scale_y
        camera_matrix[0, 2] *= scale_x
        camera_matrix[1, 2] *= scale_y
        return camera_matrix

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
