import cv2
import numpy as np
from typing import Tuple

class FeatureExtractor:
    """Extracts local features (ORB) from images."""

    def __init__(
        self,
        algorithm: str = "ORB",
        max_features: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8
    ) -> None:
        self.algorithm = algorithm
        self.max_features = max_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        
        if self.algorithm.upper() == "ORB":
            self._detector = cv2.ORB_create(
                nfeatures=self.max_features,
                scaleFactor=self.scale_factor,
                nlevels=self.n_levels
            )
        else:
            raise ValueError(f"Unsupported feature algorithm: {self.algorithm}")

    def detect_and_compute(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect keypoints and compute descriptors from an image.
        Returns:
            keypoints: np.ndarray of shape (N, 2)
            descriptors: np.ndarray of shape (N, 32)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        kp, des = self._detector.detectAndCompute(gray, None)
        
        if kp is None or len(kp) == 0:
            kp_array = np.array([]).reshape(0, 2)
            des_array = np.array([]).reshape(0, 32)
        else:
            kp_array = np.array([[k.pt[0], k.pt[1]] for k in kp])
            des_array = des
            
        if des_array is None:
            des_array = np.array([]).reshape(0, 32)
            
        return kp_array, des_array
