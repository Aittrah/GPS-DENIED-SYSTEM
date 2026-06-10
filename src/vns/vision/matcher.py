import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from vns.database.reference_db import DatabaseEntry
from vns.utils.coordinates import enu_to_geodetic, geodetic_to_enu

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
            
        # Extrinsics and Intrinsics
        fx = camera_config.get("fx", 554.25)
        fy = camera_config.get("fy", 554.25)
        f = (fx + fy) / 2.0
        
        # Median pixel displacement
        du = np.median(inliers_query[:, 0] - inliers_ref[:, 0])
        dv = np.median(inliers_query[:, 1] - inliers_ref[:, 1])
        
        # Altitude above terrain: use drone altitude relative to the reference point altitude
        # Wait, if drone_altitude is absolute altitude MSL, the altitude relative to the entry terrain is:
        height = max(1.0, drone_altitude - entry.altitude)
        
        # Pixel translation in body frame
        # Camera is facing straight down, pitch=90 deg.
        # body_x: right, body_y: forward
        dx_body = -du * (height / f)
        dy_body = -dv * (height / f)
        
        # Rotate body coordinates to ENU frame using drone's yaw (heading)
        # yaw (heading) is clockwise from North (deg)
        yaw_rad = np.radians(drone_heading)
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        
        # Forward is Y, Right is X. So at heading 0, E = body_x, N = body_y.
        de = dx_body * cos_y + dy_body * sin_y
        dn = -dx_body * sin_y + dy_body * cos_y
        
        # Get Reference point ENU coordinates relative to geo_origin
        origin_lat = geo_origin["origin_latitude"]
        origin_lon = geo_origin["origin_longitude"]
        origin_alt = geo_origin["origin_altitude"]
        
        ref_e, ref_n, ref_u = geodetic_to_enu(
            entry.latitude, entry.longitude, entry.altitude,
            origin_lat, origin_lon, origin_alt
        )
        
        # Add relative offset to get Estimated ENU coordinates
        est_e = ref_e + de
        est_n = ref_n + dn
        est_u = drone_altitude - origin_alt
        
        # Convert back to Geodetic coordinates
        est_lat, est_lon, est_alt = enu_to_geodetic(
            est_e, est_n, est_u,
            origin_lat, origin_lon, origin_alt
        )
        
        return est_lat, est_lon, drone_altitude
