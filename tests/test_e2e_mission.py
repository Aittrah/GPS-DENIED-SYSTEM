"""End-to-end mission test: an autonomous straight-line leg toward a waypoint
that loses GPS mid-mission, asserting the full navigation mode-transition chain
(GPS-healthy -> denied -> vision/fused) produces a continuous, valid position
output across the transition and the drone keeps progressing to the waypoint.

Closes GitHub issue #25 ("Implement end-to-end mission test", requirement 7.1).

This drives the canonical ``VnsRuntime`` (no ROS / Gazebo / PX4) with controlled
synthetic inputs: a healthy GPS leg, then a GNSS dropout during which a stubbed
visual localizer reports poses continuing the leg in sub-continuity-threshold
steps so the blender stays fused on vision. Mission *completion* is asserted as a
position proxy (arrival within tolerance of the final waypoint); true PX4 mission
completion requires the manual SITL run, which remains the source of that proof.
"""

import math

import numpy as np

from vns.core.runtime import VnsRuntime
from vns.utils.coordinates import enu_to_geodetic
from vns.vision.types import LocalizationResult


GRAVITY_M_S2 = 9.80665

# sample_config's geo_reference origin (tests/conftest.py).
ORIGIN_LAT = 33.7470
ORIGIN_LON = 73.1370
ORIGIN_ALT = 550.0
MISSION_UP = 30.0  # flight altitude above the geodetic origin (= 580 m)

STEP_M = 1.5            # < position_continuity_threshold (2.0 m) so visual fixes are accepted
N_STEPS = 20
DENIAL_STEP = 10        # GPS healthy for steps [0, DENIAL_STEP); denied for [DENIAL_STEP, N_STEPS)
JUMP_THRESHOLD_M = 5.0  # max allowed step-to-step horizontal jump (continuity)
WAYPOINT_TOLERANCE_M = 2.0


def _haversine_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _trajectory_point(step):
    """Geodetic ``(lat, lon, alt)`` and NED ``(north, east, down)`` for a
    straight northbound leg, ``STEP_M`` metres apart."""
    north = step * STEP_M
    east = 0.0
    geo = enu_to_geodetic(east, north, MISSION_UP, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)
    ned = (north, east, -MISSION_UP)
    return geo, ned


class _ScriptedLocalizer:
    """Stand-in for ``VisualLocalizer.localize`` returning a scripted result."""

    def __init__(self):
        self.result = None

    def localize(self, frame, altitude, heading_deg, *, prior=None):
        return self.result


def _run_mission(monkeypatch, sample_config, sample_database):
    """Fly the northbound mission that loses GPS mid-flight; vision carries it.

    Returns the ordered ``[(navigation_mode, blended_pose), ...]`` per step.
    """
    runtime = VnsRuntime(sample_config, database=sample_database)
    scripted = _ScriptedLocalizer()
    assert runtime.localizer is not None
    monkeypatch.setattr(runtime.localizer, "localize", scripted.localize)

    frame = np.zeros((480, 640), dtype=np.uint8)
    base_t = 1000.0
    history = []

    for step in range(N_STEPS):
        t = base_t + step
        geo, ned = _trajectory_point(step)

        # Vision always reports the current trajectory point. It is ignored under a
        # healthy fix (the blender returns GPS) and used once GPS is denied.
        scripted.result = LocalizationResult(
            success=True,
            confidence=0.9,
            inlier_count=30,
            matched_ref_id="synth",
            reason="ok",
            pose_ned=ned,
            pose_geodetic=geo,
            yaw_rad=0.0,
            timestamp=t,
        )

        if step < DENIAL_STEP:
            runtime.update_gps(
                latitude=geo[0], longitude=geo[1], altitude=geo[2],
                has_fix=True, num_satellites=10, hdop=1.0, timestamp=t,
            )
        else:
            # GPS denied: the position passed here is ignored by the denied blend path.
            runtime.update_gps(
                latitude=geo[0], longitude=geo[1], altitude=geo[2],
                has_fix=False, num_satellites=0, hdop=99.9, timestamp=t,
            )

        # Continuous IMU keeps dead-reckoning primed across the dropout.
        runtime.update_imu(
            w=1.0, x=0.0, y=0.0, z=0.0,
            accel_x=0.0, accel_y=0.0, accel_z=-GRAVITY_M_S2,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            timestamp=t + 0.5,
        )

        result = runtime.process_frame(frame, timestamp=t)
        history.append((result.navigation_mode, result.blended_pose))

    return history


def test_mission_continues_through_gnss_denial(monkeypatch, sample_config, sample_database):
    history = _run_mission(monkeypatch, sample_config, sample_database)
    assert len(history) == N_STEPS

    modes = [mode for mode, _ in history]
    poses = [pose for _, pose in history]

    # (a) Validity: every step yields a finite position (no None/FAILSAFE, no NaN/inf).
    for mode, pose in history:
        assert pose is not None, f"missing pose in mode {mode}"
        assert all(math.isfinite(component) for component in pose)
    assert "FAILSAFE" not in modes

    # (b) Continuity: bounded step-to-step horizontal jump across the whole mission,
    #     including the GPS -> denied handoff.
    for (lat1, lon1, _), (lat2, lon2, _) in zip(poses, poses[1:]):
        assert _haversine_m(lat1, lon1, lat2, lon2) < JUMP_THRESHOLD_M

    # (c) Full mode-transition chain: healthy GPS, then vision/fused after denial.
    assert modes[0] == "GPS"
    assert modes[DENIAL_STEP - 1] == "GPS"
    assert modes[DENIAL_STEP] in {"VISION", "DR"}
    assert modes[-1] in {"VISION", "DR"}

    # (d) Progress + completion: monotonic approach and arrival at the waypoint.
    waypoint_geo, _ = _trajectory_point(N_STEPS - 1)
    dists = [_haversine_m(p[0], p[1], waypoint_geo[0], waypoint_geo[1]) for p in poses]
    for d_prev, d_next in zip(dists, dists[1:]):
        assert d_next <= d_prev + 1e-6  # never retreats from the waypoint
    assert dists[-1] < WAYPOINT_TOLERANCE_M  # mission-completion proxy (no PX4 in bare pytest)


def test_no_position_discontinuity_at_denial_boundary(
    monkeypatch, sample_config, sample_database
):
    history = _run_mission(monkeypatch, sample_config, sample_database)

    # The GPS -> denied handoff is step (DENIAL_STEP - 1) -> step DENIAL_STEP.
    before_mode, before_pose = history[DENIAL_STEP - 1]
    after_mode, after_pose = history[DENIAL_STEP]

    assert before_mode == "GPS"
    assert after_mode in {"VISION", "DR"}
    assert before_pose is not None and after_pose is not None

    jump = _haversine_m(before_pose[0], before_pose[1], after_pose[0], after_pose[1])
    assert jump < JUMP_THRESHOLD_M
