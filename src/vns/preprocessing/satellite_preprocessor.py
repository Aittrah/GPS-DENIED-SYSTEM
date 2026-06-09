from typing import Optional
import cv2
import numpy as np

TARGET_SIZE = (512, 512)

class SatellitePreprocessor:

    def load_and_preprocess(self, image_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            print(f'[WARNING] Could not load: {image_path}')
            return None
        return self.load_and_preprocess_array(img)

    def load_and_preprocess_array(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, TARGET_SIZE)
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        img = self._apply_clahe(img)
        return img

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
