from __future__ import annotations

from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class FeatureExtractorProtocol(Protocol):
    """Common interface for local feature extractors."""

    def detect_and_compute(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


class OrbFeatureExtractor:
    """Extract local ORB features from images."""

    def __init__(
        self,
        algorithm: str = "ORB",
        max_features: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
    ) -> None:
        self.algorithm = algorithm
        self.max_features = max_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels

        if self.algorithm.upper() != "ORB":
            raise ValueError(f"Unsupported feature algorithm: {self.algorithm}")

        self._detector = cv2.ORB_create(
            nfeatures=self.max_features,
            scaleFactor=self.scale_factor,
            nlevels=self.n_levels,
        )

    def detect_and_compute(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect keypoints and compute descriptors from an image.

        Returns:
            keypoints: float32 array of shape (N, 2)
            descriptors: uint8 array of shape (N, 32)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        if keypoints is None or len(keypoints) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 32), dtype=np.uint8),
            )

        keypoint_array = np.array(
            [[keypoint.pt[0], keypoint.pt[1]] for keypoint in keypoints],
            dtype=np.float32,
        )
        descriptor_array = (
            np.empty((0, 32), dtype=np.uint8)
            if descriptors is None
            else np.asarray(descriptors, dtype=np.uint8)
        )
        return keypoint_array, descriptor_array


class FeatureExtractor(OrbFeatureExtractor):
    """Backward-compatible default extractor used across the repo."""
