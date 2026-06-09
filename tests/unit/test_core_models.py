# tests/unit/test_core_models.py
import pytest
import numpy as np
from vns.core.geo_utils import geo_to_ned, ned_to_geo, GeoPoint, NEDPoint, QAU_ORIGIN
from vns.core.models import UAVState, VisualMatch, Waypoint

def test_geo_ned_roundtrip():
    """Convert to NED and back — should get same point"""
    original = GeoPoint(33.748, 73.138, 560.0)
    ned = geo_to_ned(original)
    recovered = ned_to_geo(ned)
    assert abs(recovered.latitude - original.latitude) < 1e-6
    assert abs(recovered.longitude - original.longitude) < 1e-6

def test_origin_is_zero():
    ned = geo_to_ned(QAU_ORIGIN)
    assert abs(ned.north) < 0.001
    assert abs(ned.east) < 0.001

def test_uav_state_reliability():
    pos = NEDPoint(0, 0, -50)
    state = UAVState(position=pos, confidence=0.25)
    assert not state.is_reliable
    state.confidence = 0.5
    assert state.is_reliable

def test_waypoint_reached():
    wp = Waypoint(NEDPoint(100, 100, -50), "wp1", tolerance_m=2.0)
    assert wp.is_reached(NEDPoint(101, 100, -50))   # within tolerance
    assert not wp.is_reached(NEDPoint(105, 100, -50)) # too far

def test_visual_match_validity():
    pos = NEDPoint(10, 20, -50)
    match = VisualMatch(pos, confidence=0.8, 
                        matched_reference_id="ref_01", num_inliers=15)
    assert match.is_valid
    bad = VisualMatch(pos, confidence=0.1, 
                      matched_reference_id="ref_01", num_inliers=5)
    assert not bad.is_valid