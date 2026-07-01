import json
from copy import deepcopy

import numpy as np
import pytest

from vns.core.runtime import VnsRuntime
from vns.validation import JsonlEvaluationLogger
from vns.vision.types import LocalizationResult
from vns.vision.bovw_retrieval import BoVWIndex
from vns.utils.coordinates import enu_to_geodetic


GRAVITY_M_S2 = 9.80665


def test_runtime_reports_database_unavailable(sample_config, textured_image) -> None:
    runtime = VnsRuntime(sample_config, database=None)

    result = runtime.process_frame(textured_image, timestamp=100.0)
    diagnostics = runtime.get_diagnostics()

    assert not result.success
    assert result.reason == "database_unavailable"
    assert diagnostics.localizer_ready is False
    assert diagnostics.navigation_mode == "FAILSAFE"
    assert diagnostics.subsystems[0].state == "MISSING"


def test_runtime_detects_coverage_gap(sample_config, sample_database) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=34.5,
        longitude=74.5,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )

    result = runtime.process_frame(np.zeros((480, 640), dtype=np.uint8), timestamp=101.0)
    diagnostics = runtime.get_diagnostics()

    assert not result.success
    assert result.reason == "coverage_gap"
    assert diagnostics.coverage_gap_detected is True
    assert diagnostics.last_localization_reason == "coverage_gap"
    assert diagnostics.subsystems[2].details["coverage_gap_detected"] is True


def test_runtime_successful_localization_updates_diagnostics(
    sample_config,
    sample_database,
    textured_image,
) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_heading_from_quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

    result = runtime.process_frame(textured_image, timestamp=101.0)
    diagnostics = runtime.get_diagnostics()

    assert result.success
    assert result.localization.reason == "ok"
    assert result.blended_pose is not None
    assert diagnostics.localizer_ready is True
    assert diagnostics.database_entry_count == sample_database.entry_count
    assert diagnostics.coverage_gap_detected is False
    assert diagnostics.last_localization_reason == "ok"


def test_runtime_returns_dr_pose_when_localization_fails_under_gnss_denial(
    monkeypatch,
    sample_config,
    sample_database,
    textured_image,
) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_imu(
        w=1.0,
        x=0.0,
        y=0.0,
        z=0.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=-GRAVITY_M_S2,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        timestamp=100.5,
    )
    runtime.update_imu(
        w=1.0,
        x=0.0,
        y=0.0,
        z=0.0,
        accel_x=1.0,
        accel_y=0.0,
        accel_z=-GRAVITY_M_S2,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        timestamp=101.5,
    )
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=False,
        num_satellites=0,
        hdop=99.9,
        timestamp=102.0,
    )

    def fail_localization(frame, altitude, heading_deg, *, prior=None):
        return LocalizationResult(
            success=False,
            confidence=0.2,
            inlier_count=6,
            matched_ref_id=None,
            reason="low_confidence",
            pose_ned=None,
            pose_geodetic=None,
            yaw_rad=0.0,
            timestamp=103.0,
        )

    assert runtime.localizer is not None
    monkeypatch.setattr(runtime.localizer, "localize", fail_localization)

    result = runtime.process_frame(textured_image, timestamp=103.0)
    expected_pose = enu_to_geodetic(
        0.0,
        0.5,
        30.0,
        33.7470,
        73.1370,
        550.0,
    )

    assert result.success is False
    assert result.navigation_mode == "DR"
    assert result.reason == "low_confidence"
    assert result.blended_pose == pytest.approx(expected_pose)
    assert runtime.dead_reckoning_ready is True
    assert runtime.dead_reckoning.get_estimate(current_time=103.0).confidence == pytest.approx(
        0.7
    )


def test_runtime_update_imu_duplicate_timestamp_does_not_raise(sample_config) -> None:
    runtime = VnsRuntime(sample_config, database=None)
    imu_kwargs = dict(
        w=1.0,
        x=0.0,
        y=0.0,
        z=0.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=-GRAVITY_M_S2,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
    )

    runtime.update_imu(timestamp=100.0, **imu_kwargs)
    runtime.update_imu(timestamp=100.0, **imu_kwargs)

    assert runtime.dead_reckoning.duplicate_sample_count == 1
    diagnostics = runtime.get_diagnostics()
    dr_subsystem = next(s for s in diagnostics.subsystems if s.name == "dead_reckoning")
    assert dr_subsystem.details["duplicate_sample_count"] == 1


def test_runtime_update_imu_decreasing_timestamp_does_not_raise(sample_config) -> None:
    runtime = VnsRuntime(sample_config, database=None)
    imu_kwargs = dict(
        w=1.0,
        x=0.0,
        y=0.0,
        z=0.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=-GRAVITY_M_S2,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
    )

    runtime.update_imu(timestamp=100.0, **imu_kwargs)
    runtime.update_imu(timestamp=99.5, **imu_kwargs)

    assert runtime.dead_reckoning.stale_sample_count == 1


def test_runtime_update_imu_large_backward_jump_resets_without_raising(
    sample_config,
) -> None:
    runtime = VnsRuntime(sample_config, database=None)
    imu_kwargs = dict(
        w=1.0,
        x=0.0,
        y=0.0,
        z=0.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=-GRAVITY_M_S2,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
    )

    runtime.update_imu(timestamp=200.0, **imu_kwargs)
    runtime.update_imu(timestamp=0.0, **imu_kwargs)

    assert runtime.dead_reckoning.timestamp_reset_count == 1


def test_runtime_get_diagnostics_includes_dead_reckoning_subsystem(sample_config) -> None:
    runtime = VnsRuntime(sample_config, database=None)

    diagnostics = runtime.get_diagnostics()

    assert diagnostics.subsystems[0].name == "database"
    assert diagnostics.subsystems[2].name == "navigation"
    dr_subsystem = next(s for s in diagnostics.subsystems if s.name == "dead_reckoning")
    assert dr_subsystem.details == {
        "duplicate_sample_count": 0,
        "stale_sample_count": 0,
        "timestamp_reset_count": 0,
        "clamped_dt_count": 0,
        "last_timestamp": None,
    }


def test_runtime_visual_fix_resets_dead_reckoning_under_gnss_denial(
    monkeypatch,
    sample_config,
    sample_database,
    textured_image,
) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=False,
        num_satellites=0,
        hdop=99.9,
        timestamp=101.0,
    )
    visual_pose_ned = (12.0, -6.0, -30.0)
    visual_pose_geo = enu_to_geodetic(
        visual_pose_ned[1],
        visual_pose_ned[0],
        -visual_pose_ned[2],
        33.7470,
        73.1370,
        550.0,
    )

    def succeed_localization(frame, altitude, heading_deg, *, prior=None):
        return LocalizationResult(
            success=True,
            confidence=0.92,
            inlier_count=24,
            matched_ref_id="grid_center",
            reason="ok",
            pose_ned=visual_pose_ned,
            pose_geodetic=visual_pose_geo,
            yaw_rad=0.1,
            timestamp=102.0,
        )

    assert runtime.localizer is not None
    monkeypatch.setattr(runtime.localizer, "localize", succeed_localization)

    result = runtime.process_frame(textured_image, timestamp=102.0)
    estimate = runtime.dead_reckoning.get_estimate()

    assert result.success is True
    assert result.navigation_mode == "VISION"
    assert result.blended_pose == pytest.approx(visual_pose_geo)
    assert runtime.dead_reckoning_ready is True
    assert estimate.position.north == pytest.approx(visual_pose_ned[0])
    assert estimate.position.east == pytest.approx(visual_pose_ned[1])
    assert estimate.position.down == pytest.approx(visual_pose_ned[2])
    assert estimate.confidence == pytest.approx(0.92)


def test_runtime_dr_bridge_expires_without_fresh_correction(
    monkeypatch,
    sample_config,
    sample_database,
    textured_image,
) -> None:
    config = deepcopy(sample_config)
    config["dead_reckoning"] = {
        "max_bridge_duration_seconds": 1.0,
        "confidence_decay_per_second": 0.0,
    }
    runtime = VnsRuntime(config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_imu(
        w=1.0,
        x=0.0,
        y=0.0,
        z=0.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=-GRAVITY_M_S2,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        timestamp=100.5,
    )
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=False,
        num_satellites=0,
        hdop=99.9,
        timestamp=102.0,
    )

    def fail_localization(frame, altitude, heading_deg, *, prior=None):
        return LocalizationResult(
            success=False,
            confidence=0.2,
            inlier_count=6,
            matched_ref_id=None,
            reason="low_confidence",
            pose_ned=None,
            pose_geodetic=None,
            yaw_rad=0.0,
            timestamp=102.1,
        )

    assert runtime.localizer is not None
    monkeypatch.setattr(runtime.localizer, "localize", fail_localization)

    result = runtime.process_frame(textured_image, timestamp=102.1)

    assert result.navigation_mode == "FAILSAFE"
    assert result.blended_pose is None
    assert runtime.dead_reckoning_ready is False


def test_runtime_healthy_gps_reanchors_expired_dead_reckoning_bridge(
    monkeypatch,
    sample_config,
    sample_database,
    textured_image,
) -> None:
    config = deepcopy(sample_config)
    config["dead_reckoning"] = {
        "max_bridge_duration_seconds": 1.0,
        "confidence_decay_per_second": 0.0,
    }
    runtime = VnsRuntime(config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=False,
        num_satellites=0,
        hdop=99.9,
        timestamp=102.0,
    )

    def fail_localization(frame, altitude, heading_deg, *, prior=None):
        return LocalizationResult(
            success=False,
            confidence=0.2,
            inlier_count=6,
            matched_ref_id=None,
            reason="low_confidence",
            pose_ned=None,
            pose_geodetic=None,
            yaw_rad=0.0,
            timestamp=102.1,
        )

    assert runtime.localizer is not None
    monkeypatch.setattr(runtime.localizer, "localize", fail_localization)

    expired = runtime.process_frame(textured_image, timestamp=102.1)
    assert expired.navigation_mode == "FAILSAFE"

    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=103.0,
    )
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=False,
        num_satellites=0,
        hdop=99.9,
        timestamp=103.2,
    )

    recovered = runtime.process_frame(textured_image, timestamp=103.3)

    assert recovered.navigation_mode == "DR"
    assert recovered.blended_pose == pytest.approx((33.7470, 73.1370, 580.0))
    assert runtime.dead_reckoning_ready is True


def test_runtime_uses_bovw_when_configured(
    monkeypatch,
    sample_bovw_config,
    sample_bovw_database,
    textured_image,
) -> None:
    calls = {"count": 0}
    original_query = BoVWIndex.query

    def spy_query(self, descriptors, k=5):
        calls["count"] += 1
        return original_query(self, descriptors, k=k)

    monkeypatch.setattr(BoVWIndex, "query", spy_query)

    runtime = VnsRuntime(sample_bovw_config, database=sample_bovw_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_heading_from_quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

    result = runtime.process_frame(textured_image, timestamp=101.0)

    assert result.success
    assert runtime.localizer is not None
    assert runtime.localizer._retrieval.backend_name == "bovw"
    assert calls["count"] == 1


def test_runtime_stores_ground_truth_pose(sample_config) -> None:
    runtime = VnsRuntime(sample_config, database=None)

    runtime.update_ground_truth_pose(
        latitude=33.7471,
        longitude=73.1372,
        altitude=580.5,
        heading_deg=12.0,
        timestamp=100.0,
    )

    ground_truth = runtime.last_ground_truth

    assert ground_truth is not None
    assert ground_truth.latitude == pytest.approx(33.7471)
    assert ground_truth.longitude == pytest.approx(73.1372)
    assert ground_truth.altitude == pytest.approx(580.5)
    assert ground_truth.heading_deg == pytest.approx(12.0)
    assert ground_truth.timestamp == pytest.approx(100.0)


def test_runtime_result_can_be_logged_with_ground_truth(
    sample_config,
    sample_database,
    textured_image,
    tmp_path,
) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_heading_from_quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    runtime.update_ground_truth_pose(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        heading_deg=0.0,
        timestamp=100.5,
    )

    result = runtime.process_frame(textured_image, timestamp=101.0)
    logger = JsonlEvaluationLogger(enabled=True, log_dir=tmp_path)

    wrote = logger.write_frame(
        timestamp=101.0,
        ros_time=101.0,
        result=result,
        ground_truth=runtime.last_ground_truth,
        gnss_status=runtime.gnss_monitor.state.name,
        record_without_ground_truth=False,
    )
    log_path = logger.log_path
    logger.close()

    assert wrote is True
    assert log_path is not None

    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["ground_truth_available"] is True
    assert payload["localization_success"] is True
    assert payload["navigation_mode"] == result.navigation_mode
    assert payload["gnss_status"] == runtime.gnss_monitor.state.name
    assert payload["estimated_lat"] == pytest.approx(result.blended_pose[0])
    assert payload["visual_lat"] == pytest.approx(result.localization.pose_geodetic[0])
    assert payload["horizontal_error"] is not None
    assert payload["position_error_m"] is not None


def test_logger_skips_missing_ground_truth_without_crashing(
    sample_config,
    sample_database,
    textured_image,
    tmp_path,
) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_heading_from_quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

    result = runtime.process_frame(textured_image, timestamp=101.0)
    logger = JsonlEvaluationLogger(enabled=True, log_dir=tmp_path)

    wrote = logger.write_frame(
        timestamp=101.0,
        ros_time=None,
        result=result,
        ground_truth=None,
        gnss_status=runtime.gnss_monitor.state.name,
        record_without_ground_truth=False,
    )
    log_path = logger.log_path
    logger.close()

    assert wrote is False
    assert log_path is not None
    assert log_path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# Geo-gate position prior selection (FR-9)
# ---------------------------------------------------------------------------


def test_current_position_prior_uses_gps_when_healthy(
    sample_config, sample_database
) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    assert runtime._current_position_prior() == (33.7470, 73.1370)


def test_current_position_prior_none_without_position(
    sample_config, sample_database
) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    # The configured geo-origin is never treated as a trustworthy prior.
    assert runtime._current_position_prior() is None


def test_current_position_prior_falls_back_to_blended_when_denied(
    sample_config, sample_database, textured_image
) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_heading_from_quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    result = runtime.process_frame(textured_image, timestamp=101.0)
    assert result.blended_pose is not None
    blended = result.blended_pose

    # GNSS lost: GPS is no longer trustworthy, so the prior comes from the
    # most recent fused estimate.
    runtime.update_gps(
        latitude=0.0,
        longitude=0.0,
        altitude=0.0,
        has_fix=False,
        num_satellites=0,
        hdop=99.0,
        timestamp=102.0,
    )
    assert runtime.gnss_monitor.state.name == "DENIED"
    assert runtime._current_position_prior() == (blended[0], blended[1])


def test_flann_and_bovw_geo_gate_runtimes_agree(
    sample_config,
    sample_bovw_config,
    sample_database,
    sample_bovw_database,
    textured_image,
) -> None:
    """Both retrieval backends localise the same frame to the same pose/mode."""
    bovw_geo_config = deepcopy(sample_bovw_config)
    bovw_geo_config["retrieval"]["bovw"]["geo_gate"] = True
    bovw_geo_config["retrieval"]["bovw"]["geo_gate_radius_deg"] = 0.0001

    def run(config, database):
        runtime = VnsRuntime(config, database=database)
        runtime.update_gps(
            latitude=33.7470,
            longitude=73.1370,
            altitude=580.0,
            has_fix=True,
            num_satellites=10,
            hdop=1.0,
            timestamp=100.0,
        )
        runtime.update_heading_from_quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        return runtime, runtime.process_frame(textured_image, timestamp=101.0)

    flann_runtime, flann_result = run(sample_config, sample_database)
    bovw_runtime, bovw_result = run(bovw_geo_config, sample_bovw_database)

    assert flann_runtime.localizer._retrieval.backend_name == "flann"
    assert bovw_runtime.localizer._retrieval.backend_name == "bovw"
    assert flann_result.success and bovw_result.success
    assert flann_result.navigation_mode == bovw_result.navigation_mode
    assert (
        flann_result.localization.matched_ref_id
        == bovw_result.localization.matched_ref_id
    )
    assert bovw_result.localization.confidence == pytest.approx(
        flann_result.localization.confidence, rel=1e-6
    )
    assert bovw_result.blended_pose[0] == pytest.approx(
        flann_result.blended_pose[0], abs=1e-9
    )
    assert bovw_result.blended_pose[1] == pytest.approx(
        flann_result.blended_pose[1], abs=1e-9
    )


def test_canonical_jsonl_is_superset_of_legacy_gt_fields(
    sample_config, sample_database, textured_image, tmp_path
) -> None:
    """The canonical evaluation record carries every field the retired
    ``visual_navigation.py`` JSONL logged (plus more)."""
    legacy_fields = {
        "timestamp",
        "estimated_lat",
        "estimated_lon",
        "estimated_alt",
        "true_lat",
        "true_lon",
        "true_alt",
        "estimated_heading",
        "true_heading",
        "horizontal_error",
        "vertical_error",
        "heading_error",
        "mode",
    }

    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_heading_from_quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    runtime.update_ground_truth_pose(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        heading_deg=0.0,
        timestamp=100.5,
    )

    result = runtime.process_frame(textured_image, timestamp=101.0)
    logger = JsonlEvaluationLogger(enabled=True, log_dir=tmp_path)
    logger.write_frame(
        timestamp=101.0,
        ros_time=101.0,
        result=result,
        ground_truth=runtime.last_ground_truth,
        gnss_status=runtime.gnss_monitor.state.name,
        record_without_ground_truth=False,
    )
    log_path = logger.log_path
    logger.close()

    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert legacy_fields.issubset(payload.keys())
