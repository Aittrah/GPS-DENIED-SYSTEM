"""Simulation integration tests for the GNSS health state machine and the
position blender's response to each state, exercised through the canonical
``VnsRuntime`` orchestration seam (no ROS / Gazebo / PX4 required).

Closes part of GitHub issue #24 ("Implement simulation integration tests",
requirements 5.1 / 5.3 / 5.5). Unlike ``tests/test_vns.py`` (which unit-tests
``GnssMonitor`` and the legacy blender ramp in isolation) and
``tests/test_runtime.py`` (single-step cases), these tests drive a live runtime
through a *sequenced* GNSS scenario and assert that the monitor state and the
blender's navigation mode respond together as an integrated state machine,
including GPS recovery back to the GNSS-primary mode and the GNSS-timeout path.

The full PX4 SITL run remains the source of true end-to-end proof; here the
sensor inputs are controlled synthetic values fed through the runtime's public
``update_*`` methods, and vision output is stubbed via ``monkeypatch`` exactly
as in ``tests/test_runtime.py``.
"""

from copy import deepcopy

import numpy as np
import pytest

from vns.core.gnss_monitor import GnssState
from vns.core.runtime import VnsRuntime
from vns.utils.coordinates import enu_to_geodetic
from vns.vision.types import LocalizationResult


GRAVITY_M_S2 = 9.80665

# sample_config's geo_reference origin (tests/conftest.py).
ORIGIN_LAT = 33.7470
ORIGIN_LON = 73.1370
ORIGIN_ALT = 550.0
MISSION_ALT = 580.0  # 30 m above the geodetic origin


def _frame() -> np.ndarray:
    # Frame content is irrelevant: the localizer is always stubbed below.
    return np.zeros((480, 640), dtype=np.uint8)


def _success_localization(pose_ned, pose_geodetic, timestamp):
    def _localize(frame, altitude, heading_deg, *, prior=None):
        return LocalizationResult(
            success=True,
            confidence=0.9,
            inlier_count=30,
            matched_ref_id="synth",
            reason="ok",
            pose_ned=pose_ned,
            pose_geodetic=pose_geodetic,
            yaw_rad=0.0,
            timestamp=timestamp,
        )

    return _localize


def _failed_localization(timestamp):
    def _localize(frame, altitude, heading_deg, *, prior=None):
        return LocalizationResult(
            success=False,
            confidence=0.1,
            inlier_count=3,
            matched_ref_id=None,
            reason="low_confidence",
            pose_ned=None,
            pose_geodetic=None,
            yaw_rad=0.0,
            timestamp=timestamp,
        )

    return _localize


def _healthy_gps(runtime, lat, lon, alt, timestamp, *, num_satellites=10, hdop=1.0):
    runtime.update_gps(
        latitude=lat,
        longitude=lon,
        altitude=alt,
        has_fix=True,
        num_satellites=num_satellites,
        hdop=hdop,
        timestamp=timestamp,
    )


def _deny_gps(runtime, lat, lon, alt, timestamp):
    runtime.update_gps(
        latitude=lat,
        longitude=lon,
        altitude=alt,
        has_fix=False,
        num_satellites=0,
        hdop=99.9,
        timestamp=timestamp,
    )


def test_gnss_state_machine_transitions_through_runtime(sample_config, sample_database):
    """HEALTHY -> DEGRADED -> DENIED driven through ``VnsRuntime.update_gps`` (req 5.1)."""
    runtime = VnsRuntime(sample_config, database=sample_database)

    _healthy_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=100.0)
    assert runtime.gnss_monitor.state is GnssState.HEALTHY

    # DEGRADED via too-few satellites.
    _healthy_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=101.0, num_satellites=3)
    assert runtime.gnss_monitor.state is GnssState.DEGRADED

    # DEGRADED via excessive HDOP.
    _healthy_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=102.0, hdop=6.0)
    assert runtime.gnss_monitor.state is GnssState.DEGRADED

    # DENIED via loss of fix.
    _deny_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=103.0)
    assert runtime.gnss_monitor.state is GnssState.DENIED


def test_blender_response_matrix_per_state(monkeypatch, sample_config, sample_database):
    """The blender's navigation mode tracks the GNSS state through the runtime."""
    frame = _frame()
    runtime = VnsRuntime(sample_config, database=sample_database)

    # HEALTHY -> "GPS" (visual result is irrelevant under a healthy fix).
    _healthy_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=100.0)
    monkeypatch.setattr(runtime.localizer, "localize", _failed_localization(100.0))
    result = runtime.process_frame(frame, timestamp=100.0)
    assert runtime.gnss_monitor.state is GnssState.HEALTHY
    assert result.navigation_mode == "GPS"

    # DENIED + accepted visual fix -> "VISION".
    visual_ned = (0.0, 0.0, -(MISSION_ALT - ORIGIN_ALT))
    visual_geo = enu_to_geodetic(
        visual_ned[1], visual_ned[0], -visual_ned[2], ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT
    )
    _deny_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=101.0)
    monkeypatch.setattr(
        runtime.localizer, "localize", _success_localization(visual_ned, visual_geo, 101.0)
    )
    result = runtime.process_frame(frame, timestamp=101.0)
    assert runtime.gnss_monitor.state is GnssState.DENIED
    assert result.navigation_mode == "VISION"

    # DENIED + failed visual but dead-reckoning primed -> "DR".
    monkeypatch.setattr(runtime.localizer, "localize", _failed_localization(102.0))
    runtime.update_imu(
        w=1.0, x=0.0, y=0.0, z=0.0,
        accel_x=0.0, accel_y=0.0, accel_z=-GRAVITY_M_S2,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        timestamp=101.5,
    )
    result = runtime.process_frame(frame, timestamp=102.0)
    assert result.navigation_mode == "DR"
    assert runtime.dead_reckoning_ready is True

    # Fresh runtime: DENIED + failed visual + no DR -> "FAILSAFE" with no pose.
    fresh = VnsRuntime(sample_config, database=sample_database)
    _deny_gps(fresh, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=200.0)
    monkeypatch.setattr(fresh.localizer, "localize", _failed_localization(200.0))
    result = fresh.process_frame(frame, timestamp=200.0)
    assert fresh.gnss_monitor.state is GnssState.DENIED
    assert result.navigation_mode == "FAILSAFE"
    assert result.blended_pose is None


def test_vns_activation_and_position_accuracy_under_denial(
    monkeypatch, sample_config, sample_database
):
    """Under GNSS denial a valid visual fix activates VNS and the fused output
    equals that fix (req 5.3 — VNS activation and position accuracy)."""
    frame = _frame()
    runtime = VnsRuntime(sample_config, database=sample_database)

    # Establish a healthy GPS estimate at the origin (sets the continuity anchor).
    _healthy_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=100.0)
    monkeypatch.setattr(runtime.localizer, "localize", _failed_localization(100.0))
    runtime.process_frame(frame, timestamp=100.0)

    # Deny GPS and supply a known visual fix 1 m north (within the 2 m continuity gate).
    known_ned = (1.0, 0.0, -(MISSION_ALT - ORIGIN_ALT))
    known_geo = enu_to_geodetic(
        known_ned[1], known_ned[0], -known_ned[2], ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT
    )
    _deny_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=101.0)
    monkeypatch.setattr(
        runtime.localizer, "localize", _success_localization(known_ned, known_geo, 101.0)
    )
    result = runtime.process_frame(frame, timestamp=101.0)

    assert result.success is True
    assert result.navigation_mode == "VISION"
    assert result.blended_pose == pytest.approx(known_geo)


def test_gps_recovery_returns_to_gnss_primary(monkeypatch, sample_config, sample_database):
    """HEALTHY -> DENIED (VISION) -> recovered HEALTHY returns to GNSS-primary "GPS"
    (req 5.5 — mode transition back to GNSS_PRIMARY)."""
    frame = _frame()
    runtime = VnsRuntime(sample_config, database=sample_database)

    _healthy_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=100.0)
    monkeypatch.setattr(runtime.localizer, "localize", _failed_localization(100.0))
    healthy_result = runtime.process_frame(frame, timestamp=100.0)
    assert runtime.gnss_monitor.state is GnssState.HEALTHY
    assert healthy_result.navigation_mode == "GPS"

    visual_ned = (1.0, 0.0, -(MISSION_ALT - ORIGIN_ALT))
    visual_geo = enu_to_geodetic(
        visual_ned[1], visual_ned[0], -visual_ned[2], ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT
    )
    _deny_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=101.0)
    monkeypatch.setattr(
        runtime.localizer, "localize", _success_localization(visual_ned, visual_geo, 101.0)
    )
    denied_result = runtime.process_frame(frame, timestamp=101.0)
    assert runtime.gnss_monitor.state is GnssState.DENIED
    assert denied_result.navigation_mode == "VISION"

    # GPS recovers.
    _healthy_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=102.0)
    monkeypatch.setattr(runtime.localizer, "localize", _failed_localization(102.0))
    recovered_result = runtime.process_frame(frame, timestamp=102.0)
    assert runtime.gnss_monitor.state is GnssState.HEALTHY
    assert recovered_result.navigation_mode == "GPS"
    assert recovered_result.blended_pose == pytest.approx((ORIGIN_LAT, ORIGIN_LON, MISSION_ALT))


def test_gnss_timeout_drives_denied_through_runtime(
    monkeypatch, sample_config, sample_database
):
    """A stale GNSS feed (no update within the timeout) is declared DENIED via
    ``check_gnss_timeout`` and the blender leaves GPS mode."""
    frame = _frame()
    runtime = VnsRuntime(sample_config, database=sample_database)

    t0 = 100.0
    _healthy_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=t0)
    assert runtime.gnss_monitor.state is GnssState.HEALTHY

    timeout = runtime.gnss_monitor.denied_timeout_seconds
    runtime.check_gnss_timeout(current_time=t0 + timeout + 0.1)
    assert runtime.gnss_monitor.state is GnssState.DENIED

    # The healthy fix anchored dead-reckoning, so the next frame carries on in "DR".
    monkeypatch.setattr(runtime.localizer, "localize", _failed_localization(t0 + timeout + 0.2))
    result = runtime.process_frame(frame, timestamp=t0 + timeout + 0.2)
    assert result.navigation_mode != "GPS"
    assert result.navigation_mode == "DR"


def test_short_dr_bridge_expires_to_failsafe(
    monkeypatch, sample_config, sample_database
):
    """A bounded DR bridge eventually expires to FAILSAFE if no correction arrives."""
    config = deepcopy(sample_config)
    config["dead_reckoning"] = {
        "max_bridge_duration_seconds": 1.0,
        "confidence_decay_per_second": 0.0,
    }
    frame = _frame()
    runtime = VnsRuntime(config, database=sample_database)

    _healthy_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=100.0)
    _deny_gps(runtime, ORIGIN_LAT, ORIGIN_LON, MISSION_ALT, timestamp=100.1)

    monkeypatch.setattr(runtime.localizer, "localize", _failed_localization(100.2))
    short_bridge = runtime.process_frame(frame, timestamp=100.2)
    assert short_bridge.navigation_mode == "DR"

    monkeypatch.setattr(runtime.localizer, "localize", _failed_localization(101.2))
    expired = runtime.process_frame(frame, timestamp=101.2)
    assert expired.navigation_mode == "FAILSAFE"
    assert expired.blended_pose is None
