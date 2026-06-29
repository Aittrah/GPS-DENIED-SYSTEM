"""Camera frame preprocessing: resize, grayscale, CLAHE."""

import cv2
import numpy as np


class Preprocessor:
    """Resize -> grayscale -> CLAHE contrast enhancement."""

    def __init__(
        self,
        target_width: int = 640,
        target_height: int = 480,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        *,
        camera_matrix: np.ndarray | None = None,
        distortion_coefficients: np.ndarray | None = None,
        calibration_size: tuple[int, int] | None = None,
        undistort: bool = True,
    ) -> None:
        self._target_size = (target_width, target_height)
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_grid_size, clahe_grid_size),
        )
        self._camera_matrix = (
            None if camera_matrix is None else np.asarray(camera_matrix, dtype=np.float32)
        )
        self._distortion_coefficients = (
            None
            if distortion_coefficients is None
            else np.asarray(distortion_coefficients, dtype=np.float32)
        )
        self._calibration_size = calibration_size
        self._undistort = undistort

    def _scaled_camera_matrix(
        self,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray | None:
        if self._camera_matrix is None:
            return None
        if self._calibration_size is None:
            return self._camera_matrix
        calib_width, calib_height = self._calibration_size
        if calib_width <= 0 or calib_height <= 0:
            return self._camera_matrix
        if (frame_width, frame_height) == self._calibration_size:
            return self._camera_matrix

        scale_x = frame_width / calib_width
        scale_y = frame_height / calib_height
        scaled = self._camera_matrix.copy()
        scaled[0, 0] *= scale_x
        scaled[1, 1] *= scale_y
        scaled[0, 2] *= scale_x
        scaled[1, 2] *= scale_y
        return scaled

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a camera frame for feature extraction.

        Returns a grayscale, contrast-enhanced image at the target resolution.
        """
        if (
            self._undistort
            and self._distortion_coefficients is not None
            and np.any(self._distortion_coefficients)
        ):
            camera_matrix = self._scaled_camera_matrix(frame.shape[1], frame.shape[0])
            if camera_matrix is not None:
                frame = cv2.undistort(
                    frame,
                    camera_matrix,
                    self._distortion_coefficients,
                )

        if frame.shape[1] != self._target_size[0] or frame.shape[0] != self._target_size[1]:
            frame = cv2.resize(
                frame, self._target_size, interpolation=cv2.INTER_AREA,
            )

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return self._clahe.apply(frame)
