"""Geometric verification: ratio-test matching + homography RANSAC per candidate."""

import logging
from typing import Optional

import cv2
import numpy as np

from vns.database.reference_db import DatabaseEntry
from vns.vision.types import VerificationResult

logger = logging.getLogger("vns.vision.verification")


class GeometricVerifier:
    """Match query descriptors against a candidate, estimate homography, count inliers."""

    def __init__(
        self,
        ratio_test_threshold: float = 0.75,
        min_matches: int = 10,
        ransac_reproj_threshold: float = 5.0,
    ) -> None:
        self._ratio_threshold = ratio_test_threshold
        self._min_matches = min_matches
        self._ransac_threshold = ransac_reproj_threshold
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def verify(
        self,
        query_kp: np.ndarray,
        query_des: np.ndarray,
        entry: DatabaseEntry,
    ) -> Optional[VerificationResult]:
        """Run ratio test + homography RANSAC on one candidate.

        Returns a :class:`VerificationResult` when enough inliers survive,
        ``None`` otherwise.
        """
        if (
            len(query_kp) == 0
            or query_des is None
            or len(query_des) == 0
            or entry.descriptors is None
            or len(entry.descriptors) == 0
            or len(entry.keypoints) == 0
        ):
            return None

        try:
            matches = self._matcher.knnMatch(
                query_des.astype(np.uint8),
                entry.descriptors.astype(np.uint8),
                k=2,
            )
        except cv2.error as exc:
            logger.warning("Matching failed for %s: %s", entry.id, exc)
            return None

        good = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < self._ratio_threshold * n.distance:
                    good.append(m)

        if len(good) < self._min_matches:
            return None

        pts_query = np.float32([query_kp[m.queryIdx] for m in good])
        pts_ref = np.float32([entry.keypoints[m.trainIdx] for m in good])

        try:
            H, mask = cv2.findHomography(
                pts_ref, pts_query, cv2.RANSAC, self._ransac_threshold,
            )
        except cv2.error as exc:
            logger.warning("Homography failed for %s: %s", entry.id, exc)
            return None

        if H is None or mask is None:
            return None

        inlier_mask = mask.ravel() == 1
        inlier_count = int(inlier_mask.sum())

        if inlier_count < self._min_matches:
            return None

        confidence = min(1.0, inlier_count / 30.0)

        return VerificationResult(
            entry_id=entry.id,
            inlier_count=inlier_count,
            confidence=confidence,
            homography=H,
            inliers_query=pts_query[inlier_mask],
            inliers_ref=pts_ref[inlier_mask],
        )
