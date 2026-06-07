from typing import Optional
import cv2
import numpy as np
from .satellite_preprocessor import TARGET_SIZE

class UAVPreprocessor:

    TARGET_SIZE = TARGET_SIZE

    def preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            print('[WARNING] Empty frame received')
            return None
        frame = cv2.resize(frame, TARGET_SIZE)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        frame = cv2.filter2D(frame, -1, kernel)
        frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        frame = self._apply_clahe(frame)
        return frame

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
