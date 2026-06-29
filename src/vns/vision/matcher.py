import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from vns.database.reference_db import DatabaseEntry
from vns.vision.pose_recovery import displacement_to_pose

logger = logging.getLogger("vns.vision")

class FeatureMatcher:
    """Matches query image features against database reference entries and estimates relative pose."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        ratio_test_threshold: float = 0.75,
        min_matches: int = 10
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.ratio_test_threshold = ratio_test_threshold
        self.min_matches = min_matches
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(
        self,
        query_kp: np.ndarray,
        query_des: np.ndarray,
        entry: DatabaseEntry
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Match query features with a database entry.
        Returns:
            inliers_query: np.ndarray of matched inliers in query image (N, 2)
            inliers_ref: np.ndarray of matched inliers in reference image (N, 2)
            confidence: Float matching confidence score between 0 and 1
        """
        if len(query_kp) == 0 or len(entry.keypoints) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0.0
            
        try:
            # KNN match for ratio test
            matches = self._bf.knnMatch(query_des, entry.descriptors, k=2)
        except Exception as e:
            logger.warning("Feature matching error: %s", e)
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0.0
            
        good_matches = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < self.ratio_test_threshold * n.distance:
                    good_matches.append(m)
                    
        if len(good_matches) < self.min_matches:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0.0
            
        # Get matched point coordinates
        pts_query = np.float32([query_kp[m.queryIdx] for m in good_matches])
        pts_ref = np.float32([entry.keypoints[m.trainIdx] for m in good_matches])
        
        # Estimate Homography using RANSAC to remove outliers
        try:
            H, mask = cv2.findHomography(pts_ref, pts_query, cv2.RANSAC, 5.0)
        except Exception as e:
            logger.warning("Homography estimation error: %s", e)
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0.0
            
        if H is None or mask is None:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0.0
            
        inliers_mask = mask.ravel() == 1
        inliers_query = pts_query[inliers_mask]
        inliers_ref = pts_ref[inliers_mask]
        
        num_inliers = len(inliers_query)
        if num_inliers < self.min_matches:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0.0
            
        # Confidence score based on count of inliers
        confidence = min(1.0, num_inliers / 30.0)
        
        return inliers_query, inliers_ref, confidence

    def estimate_relative_pose(
        self,
        query_kp: np.ndarray,
        query_des: np.ndarray,
        entry: DatabaseEntry,
        camera_config: Dict[str, Any],
        drone_altitude: float,
        drone_heading: float,
        geo_origin: Dict[str, float]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Estimate absolute lat, lon, alt from visual match.
        Returns:
            lat, lon, alt if successful, or None if matching failed or confidence is too low.
        """
        inliers_query, inliers_ref, confidence = self.match(query_kp, query_des, entry)

        if confidence < self.confidence_threshold or len(inliers_query) < self.min_matches:
            return None

        # Geometry lives in one place (shared with PoseRecovery) so the legacy
        # and current pipelines can never drift apart.
        geodetic, _ned, _yaw = displacement_to_pose(
            inliers_query,
            inliers_ref,
            entry,
            camera_config,
            drone_altitude,
            drone_heading,
            geo_origin,
        )
        return geodetic
