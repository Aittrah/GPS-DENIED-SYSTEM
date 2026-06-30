from typing import Optional
import cv2
import numpy as np
from .satellite_preprocessor import MAX_SHORT_SIDE


class UAVPreprocessor:
    """
    Pre-processes a single UAV camera frame before patch generation.
    Mirrors the satellite pipeline so descriptors are comparable.
    """

    def preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            print('[WARNING] Empty frame received')
            return None
        # Resize to same scale as satellite patches
        h, w = frame.shape[:2]
        short = min(h, w)
        if short > MAX_SHORT_SIDE:
            scale = MAX_SHORT_SIDE / short
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        frame = self._apply_clahe(frame)
        return frame

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
