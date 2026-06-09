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
    ) -> None:
        self._target_size = (target_width, target_height)
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_grid_size, clahe_grid_size),
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a camera frame for feature extraction.

        Returns a grayscale, contrast-enhanced image at the target resolution.
        """
        if frame.shape[1] != self._target_size[0] or frame.shape[0] != self._target_size[1]:
            frame = cv2.resize(
                frame, self._target_size, interpolation=cv2.INTER_AREA,
            )

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return self._clahe.apply(frame)
