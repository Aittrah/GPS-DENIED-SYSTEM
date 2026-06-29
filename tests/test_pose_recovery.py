"""Unit tests for the pose recovery stage."""

import numpy as np
import pytest

from vns.database.reference_db import DatabaseEntry
from vns.vision.pose_recovery import PoseRecovery
from vns.vision.types import VerificationResult

GEO_ORIGIN = {
    "origin_latitude": 33.7470,
    "origin_longitude": 73.1370,
    "origin_altitude": 550.0,
}

CAMERA_CFG = {"fx": 554.25, "fy": 554.25, "cx": 320.0, "cy": 240.0}


def _make_entry(
    lat: float = 33.7470,
    lon: float = 73.1370,
    alt: float = 550.0,
) -> DatabaseEntry:
    return DatabaseEntry(
        id="ref",
        source_path="",
        latitude=lat,
        longitude=lon,
        altitude=alt,
        heading=0.0,
        capture_time="",
        feature_count=0,
        feature_algorithm="ORB",
        keypoints=np.empty((0, 2)),
        descriptors=np.empty((0, 32), dtype=np.uint8),
        metadata={},
    )


class TestPoseRecovery:

    def test_zero_displacement(self):
        """Zero pixel shift should return the reference entry's position."""
        pts = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
        vr = VerificationResult(
            entry_id="ref",
            inlier_count=3,
            confidence=0.5,
            homography=np.eye(3),
            inliers_query=pts,
            inliers_ref=pts,
        )
        entry = _make_entry()
        pr = PoseRecovery()
        result = pr.recover(vr, entry, CAMERA_CFG, altitude=580.0, heading_deg=0.0, geo_origin=GEO_ORIGIN)

        assert result is not None
        geodetic, ned, yaw = result
        assert abs(geodetic[0] - entry.latitude) < 1e-5
        assert abs(geodetic[1] - entry.longitude) < 1e-5
        assert geodetic[2] == 580.0

    def test_positive_pixel_shift_moves_position(self):
        """A positive pixel offset should produce a non-zero metric offset."""
        ref_pts = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
        query_pts = ref_pts + np.array([20.0, 0.0])
        vr = VerificationResult(
            entry_id="ref",
            inlier_count=3,
            confidence=0.5,
            homography=np.eye(3),
            inliers_query=query_pts,
            inliers_ref=ref_pts,
        )
        entry = _make_entry()
        pr = PoseRecovery()
        result = pr.recover(vr, entry, CAMERA_CFG, altitude=580.0, heading_deg=0.0, geo_origin=GEO_ORIGIN)

        assert result is not None
        geodetic, ned, yaw = result
        assert abs(geodetic[0] - entry.latitude) > 1e-7 or abs(geodetic[1] - entry.longitude) > 1e-7

    def test_heading_rotates_offset(self):
        """Non-zero heading should rotate the offset direction."""
        ref_pts = np.array([[100, 100], [200, 200]], dtype=np.float32)
        query_pts = ref_pts + np.array([20.0, 0.0])
        vr = VerificationResult(
            entry_id="ref",
            inlier_count=2,
            confidence=0.5,
            homography=np.eye(3),
            inliers_query=query_pts,
            inliers_ref=ref_pts,
        )
        entry = _make_entry()
        pr = PoseRecovery()

        result_0 = pr.recover(vr, entry, CAMERA_CFG, 580.0, heading_deg=0.0, geo_origin=GEO_ORIGIN)
        result_90 = pr.recover(vr, entry, CAMERA_CFG, 580.0, heading_deg=90.0, geo_origin=GEO_ORIGIN)

        assert result_0 is not None and result_90 is not None
        geo_0, ned_0, _ = result_0
        geo_90, ned_90, _ = result_90
        assert not np.allclose(
            [geo_0[0], geo_0[1]], [geo_90[0], geo_90[1]], atol=1e-7, rtol=0,
        )

    def test_ned_sign_convention(self):
        """Down should be negative altitude-above-origin."""
        ref_pts = np.array([[100, 100]], dtype=np.float32)
        vr = VerificationResult(
            entry_id="ref",
            inlier_count=1,
            confidence=0.5,
            homography=np.eye(3),
            inliers_query=ref_pts,
            inliers_ref=ref_pts,
        )
        entry = _make_entry()
        pr = PoseRecovery()
        result = pr.recover(vr, entry, CAMERA_CFG, altitude=580.0, heading_deg=0.0, geo_origin=GEO_ORIGIN)

        assert result is not None
        _, ned, _ = result
        assert ned[2] == pytest.approx(-(580.0 - GEO_ORIGIN["origin_altitude"]))

    def test_empty_inliers(self):
        vr = VerificationResult(
            entry_id="ref",
            inlier_count=0,
            confidence=0.0,
            homography=np.eye(3),
            inliers_query=np.empty((0, 2)),
            inliers_ref=np.empty((0, 2)),
        )
        entry = _make_entry()
        pr = PoseRecovery()
        result = pr.recover(vr, entry, CAMERA_CFG, 580.0, 0.0, GEO_ORIGIN)
        assert result is None
