"""Unit tests for the geometric verification stage."""

import cv2
import numpy as np
import pytest

from vns.database.reference_db import DatabaseEntry
from vns.vision.verification import GeometricVerifier


def _extract_orb(image: np.ndarray):
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
    kp, des = orb.detectAndCompute(image, None)
    if kp is None or len(kp) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 32), dtype=np.uint8)
    kp_arr = np.array([[k.pt[0], k.pt[1]] for k in kp], dtype=np.float32)
    if des is None:
        des = np.empty((0, 32), dtype=np.uint8)
    return kp_arr, des


class TestGeometricVerifier:

    def test_matching_identical_image(self, textured_image):
        kp, des = _extract_orb(textured_image)
        entry = DatabaseEntry(
            id="ref",
            source_path="",
            latitude=33.747,
            longitude=73.137,
            altitude=550.0,
            heading=0.0,
            capture_time="",
            feature_count=len(kp),
            feature_algorithm="ORB",
            keypoints=kp,
            descriptors=des,
            metadata={},
        )
        verifier = GeometricVerifier(
            ratio_test_threshold=0.75, min_matches=5,
        )
        result = verifier.verify(kp, des, entry)
        assert result is not None
        assert result.inlier_count >= 5
        assert 0.0 <= result.confidence <= 1.0
        assert result.homography is not None
        assert result.homography.shape == (3, 3)

    def test_translated_image(self, textured_image):
        kp_ref, des_ref = _extract_orb(textured_image)
        shifted = np.roll(textured_image, 15, axis=1)
        kp_q, des_q = _extract_orb(shifted)

        entry = DatabaseEntry(
            id="ref",
            source_path="",
            latitude=33.747,
            longitude=73.137,
            altitude=550.0,
            heading=0.0,
            capture_time="",
            feature_count=len(kp_ref),
            feature_algorithm="ORB",
            keypoints=kp_ref,
            descriptors=des_ref,
            metadata={},
        )
        verifier = GeometricVerifier(min_matches=5)
        result = verifier.verify(kp_q, des_q, entry)
        assert result is not None
        assert result.inlier_count > 0

    def test_no_match_unrelated_images(self):
        rng = np.random.RandomState(7)
        img_a = rng.randint(0, 256, (480, 640), dtype=np.uint8)
        img_b = rng.randint(0, 256, (480, 640), dtype=np.uint8)

        kp_a, des_a = _extract_orb(img_a)
        kp_b, des_b = _extract_orb(img_b)

        entry = DatabaseEntry(
            id="unrelated",
            source_path="",
            latitude=0.0,
            longitude=0.0,
            altitude=0.0,
            heading=0.0,
            capture_time="",
            feature_count=len(kp_b),
            feature_algorithm="ORB",
            keypoints=kp_b,
            descriptors=des_b,
            metadata={},
        )
        verifier = GeometricVerifier(min_matches=10)
        result = verifier.verify(kp_a, des_a, entry)
        assert result is None

    def test_empty_descriptors(self):
        verifier = GeometricVerifier()
        entry = DatabaseEntry(
            id="empty",
            source_path="",
            latitude=0.0,
            longitude=0.0,
            altitude=0.0,
            heading=0.0,
            capture_time="",
            feature_count=0,
            feature_algorithm="ORB",
            keypoints=np.empty((0, 2)),
            descriptors=np.empty((0, 32), dtype=np.uint8),
            metadata={},
        )
        result = verifier.verify(
            np.empty((0, 2)), np.empty((0, 32), dtype=np.uint8), entry,
        )
        assert result is None
