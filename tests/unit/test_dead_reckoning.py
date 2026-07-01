import importlib
import math

import pytest

from vns.core.dead_reckoning import DeadReckoning
from vns.core.models import IMUSample, NEDPoint, UAVState


GRAVITY_M_S2 = 9.80665


def _sample(
    *,
    accel_x: float = 0.0,
    accel_y: float = 0.0,
    accel_z: float = -GRAVITY_M_S2,
    gyro_x: float = 0.0,
    gyro_y: float = 0.0,
    gyro_z: float = 0.0,
    timestamp: float = 0.0,
) -> IMUSample:
    return IMUSample(
        accel_x=accel_x,
        accel_y=accel_y,
        accel_z=accel_z,
        gyro_x=gyro_x,
        gyro_y=gyro_y,
        gyro_z=gyro_z,
        timestamp=timestamp,
    )


def test_integrates_constant_forward_acceleration_at_100hz() -> None:
    dr = DeadReckoning()

    for step in range(100):
        estimate = dr.update(
            _sample(
                accel_x=1.0,
                accel_z=-GRAVITY_M_S2,
                timestamp=(step + 1) * 0.01,
            ),
            dt=0.01,
        )

    assert estimate.velocity_north == pytest.approx(1.0, abs=1e-3)
    assert estimate.velocity_east == pytest.approx(0.0, abs=1e-6)
    assert estimate.velocity_down == pytest.approx(0.0, abs=1e-6)
    assert estimate.position.north == pytest.approx(0.5, abs=1e-3)
    assert estimate.position.east == pytest.approx(0.0, abs=1e-6)
    assert estimate.position.down == pytest.approx(0.0, abs=1e-6)
    assert estimate.roll == pytest.approx(0.0, abs=1e-6)
    assert estimate.pitch == pytest.approx(0.0, abs=1e-6)
    assert estimate.yaw == pytest.approx(0.0, abs=1e-6)


def test_integrates_constant_yaw_rate() -> None:
    dr = DeadReckoning()

    for step in range(100):
        estimate = dr.update(
            _sample(
                accel_z=-GRAVITY_M_S2,
                gyro_z=math.pi / 2.0,
                timestamp=(step + 1) * 0.01,
            ),
            dt=0.01,
        )

    assert estimate.yaw == pytest.approx(math.pi / 2.0, abs=2e-3)
    assert estimate.roll == pytest.approx(0.0, abs=1e-6)
    assert estimate.pitch == pytest.approx(0.0, abs=1e-6)
    assert estimate.position.north == pytest.approx(0.0, abs=1e-6)
    assert estimate.position.east == pytest.approx(0.0, abs=1e-6)
    assert estimate.position.down == pytest.approx(0.0, abs=1e-6)


def test_stationary_specific_force_stays_near_origin() -> None:
    dr = DeadReckoning()

    for step in range(1000):
        estimate = dr.update(
            _sample(
                accel_z=-GRAVITY_M_S2,
                timestamp=(step + 1) * 0.01,
            ),
            dt=0.01,
        )

    assert estimate.velocity_north == pytest.approx(0.0, abs=1e-9)
    assert estimate.velocity_east == pytest.approx(0.0, abs=1e-9)
    assert estimate.velocity_down == pytest.approx(0.0, abs=1e-9)
    assert estimate.position.north == pytest.approx(0.0, abs=1e-9)
    assert estimate.position.east == pytest.approx(0.0, abs=1e-9)
    assert estimate.position.down == pytest.approx(0.0, abs=1e-9)


def test_small_bias_has_expected_order_of_magnitude_drift() -> None:
    dr = DeadReckoning()

    # A 0.01 m/s^2 residual north bias over 10 s should drift by ~0.1 m/s in
    # velocity and ~0.5 m in position for this simplified integrator.
    for step in range(1000):
        estimate = dr.update(
            _sample(
                accel_x=0.01,
                accel_z=-GRAVITY_M_S2,
                timestamp=(step + 1) * 0.01,
            ),
            dt=0.01,
        )

    assert estimate.velocity_north == pytest.approx(0.1, abs=5e-3)
    assert estimate.position.north == pytest.approx(0.5, abs=5e-3)
    assert estimate.velocity_east == pytest.approx(0.0, abs=1e-6)
    assert estimate.velocity_down == pytest.approx(0.0, abs=1e-6)


def test_configured_gyro_bias_is_removed_before_quaternion_propagation() -> None:
    uncompensated = DeadReckoning()
    compensated = DeadReckoning(gyro_bias_rad_s=(0.0, 0.0, 0.02))

    for step in range(1000):
        sample = _sample(
            accel_z=-GRAVITY_M_S2,
            gyro_z=0.02,
            timestamp=(step + 1) * 0.01,
        )
        uncompensated.update(sample, dt=0.01)
        estimate = compensated.update(sample, dt=0.01)

    assert uncompensated.get_estimate().yaw == pytest.approx(0.2, abs=2e-3)
    assert estimate.yaw == pytest.approx(0.0, abs=1e-6)


def test_configured_accelerometer_bias_is_removed_before_translation() -> None:
    uncompensated = DeadReckoning()
    compensated = DeadReckoning(accelerometer_bias_m_s2=(0.0, 0.0, 0.05))

    for step in range(1000):
        sample = _sample(
            accel_z=-GRAVITY_M_S2 + 0.05,
            timestamp=(step + 1) * 0.01,
        )
        uncompensated.update(sample, dt=0.01)
        estimate = compensated.update(sample, dt=0.01)

    uncomp_estimate = uncompensated.get_estimate()
    assert uncomp_estimate.velocity_down == pytest.approx(0.5, abs=5e-3)
    assert uncomp_estimate.position.down == pytest.approx(2.5, abs=5e-3)
    assert estimate.velocity_down == pytest.approx(0.0, abs=1e-9)
    assert estimate.position.down == pytest.approx(0.0, abs=1e-9)


def test_confidence_decays_and_bridge_expires_without_correction() -> None:
    dr = DeadReckoning(
        initial_state=UAVState(
            position=NEDPoint(0.0, 0.0, 0.0),
            confidence=1.0,
            source="fused",
            timestamp=100.0,
        ),
        max_bridge_duration_seconds=3.0,
        confidence_decay_per_second=0.2,
    )

    assert dr.is_bridge_valid(current_time=101.0) is True
    assert dr.get_estimate(current_time=101.0).confidence == pytest.approx(0.8)
    assert dr.get_estimate(current_time=102.5).confidence == pytest.approx(0.5)
    assert dr.is_bridge_valid(current_time=103.1) is False
    assert dr.get_estimate(current_time=103.1).confidence == pytest.approx(0.0)


def test_trusted_correction_reanchors_state_and_next_timestamp_integrates() -> None:
    dr = DeadReckoning()
    corrected_state = UAVState(
        position=NEDPoint(north=10.0, east=2.0, down=-30.0),
        velocity_north=0.0,
        velocity_east=0.0,
        velocity_down=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        confidence=0.9,
        source="fused",
        timestamp=10.0,
    )

    dr.correct(corrected_state)
    estimate = dr.update(
        _sample(accel_x=1.0, accel_z=-GRAVITY_M_S2, timestamp=11.0)
    )

    assert estimate.position.north == pytest.approx(10.5, abs=1e-6)
    assert estimate.position.east == pytest.approx(2.0, abs=1e-6)
    assert estimate.position.down == pytest.approx(-30.0, abs=1e-6)
    assert estimate.velocity_north == pytest.approx(1.0, abs=1e-6)
    assert estimate.confidence == pytest.approx(0.8, abs=1e-6)


def test_timestamp_updates_match_explicit_dt_at_100hz() -> None:
    explicit = DeadReckoning()
    timestamped = DeadReckoning()

    for step in range(100):
        explicit.update(
            _sample(
                accel_x=0.5,
                accel_z=-GRAVITY_M_S2,
                gyro_z=0.1,
                timestamp=(step + 1) * 0.01,
            ),
            dt=0.01,
        )

    timestamped.update(
        _sample(
            accel_x=0.5,
            accel_z=-GRAVITY_M_S2,
            gyro_z=0.1,
            timestamp=0.0,
        )
    )
    for step in range(1, 101):
        timestamped.update(
            _sample(
                accel_x=0.5,
                accel_z=-GRAVITY_M_S2,
                gyro_z=0.1,
                timestamp=step * 0.01,
            )
        )

    explicit_estimate = explicit.get_estimate()
    timestamp_estimate = timestamped.get_estimate()

    assert timestamp_estimate.timestamp == pytest.approx(1.0, abs=1e-9)
    assert timestamp_estimate.position.north == pytest.approx(
        explicit_estimate.position.north,
        abs=1e-9,
    )
    assert timestamp_estimate.position.east == pytest.approx(
        explicit_estimate.position.east,
        abs=1e-9,
    )
    assert timestamp_estimate.position.down == pytest.approx(
        explicit_estimate.position.down,
        abs=1e-9,
    )
    assert timestamp_estimate.velocity_north == pytest.approx(
        explicit_estimate.velocity_north,
        abs=1e-9,
    )
    assert timestamp_estimate.velocity_east == pytest.approx(
        explicit_estimate.velocity_east,
        abs=1e-9,
    )
    assert timestamp_estimate.velocity_down == pytest.approx(
        explicit_estimate.velocity_down,
        abs=1e-9,
    )
    assert timestamp_estimate.yaw == pytest.approx(explicit_estimate.yaw, abs=1e-9)


def test_duplicate_timestamp_is_skipped_and_counted() -> None:
    dr = DeadReckoning()
    dr.update(_sample(timestamp=1.0))  # bootstrap, no integration
    before = dr.get_estimate(current_time=1.0)

    estimate = dr.update(_sample(accel_x=1.0, timestamp=1.0))

    assert dr.duplicate_sample_count == 1
    assert dr.last_update_outcome == "duplicate"
    assert dr.last_timestamp == pytest.approx(1.0)
    assert estimate.position.north == pytest.approx(before.position.north, abs=1e-9)
    assert estimate.velocity_north == pytest.approx(before.velocity_north, abs=1e-9)


def test_stale_out_of_order_timestamp_is_skipped_and_counted() -> None:
    dr = DeadReckoning()
    dr.update(_sample(timestamp=5.0))  # bootstrap
    dr.update(_sample(accel_x=1.0, accel_z=-GRAVITY_M_S2, timestamp=5.1))
    estimate_before = dr.get_estimate(current_time=5.1)

    estimate = dr.update(_sample(accel_x=1.0, accel_z=-GRAVITY_M_S2, timestamp=5.05))

    assert dr.stale_sample_count == 1
    assert dr.last_update_outcome == "stale"
    assert dr.last_timestamp == pytest.approx(5.1)
    assert estimate.position.north == pytest.approx(
        estimate_before.position.north, abs=1e-9
    )


def test_large_backward_jump_resets_timing_safely() -> None:
    dr = DeadReckoning()
    dr.update(_sample(timestamp=100.0))  # bootstrap
    pre_reset = dr.update(
        _sample(accel_x=1.0, accel_z=-GRAVITY_M_S2, timestamp=100.1)
    )

    estimate = dr.update(_sample(accel_x=0.0, accel_z=-GRAVITY_M_S2, timestamp=0.0))

    assert dr.timestamp_reset_count == 1
    assert dr.last_update_outcome == "reset"
    assert dr.last_timestamp == pytest.approx(0.0)
    assert dr.get_estimate().timestamp == pytest.approx(0.0)
    # Reset only re-anchors timing; pose from the prior integration is preserved.
    assert estimate.position.north == pytest.approx(pre_reset.position.north, abs=1e-9)
    assert estimate.velocity_north == pytest.approx(
        pre_reset.velocity_north, abs=1e-9
    )


def test_excessive_forward_dt_is_clamped_and_counted() -> None:
    dr = DeadReckoning()
    dr.update(_sample(timestamp=0.0))  # bootstrap

    estimate = dr.update(_sample(accel_x=1.0, accel_z=-GRAVITY_M_S2, timestamp=50.0))

    assert dr.clamped_dt_count == 1
    assert dr.last_update_outcome == "clamped"
    # Integrated using the clamped 1.0s step, not the raw 50s gap.
    assert estimate.velocity_north == pytest.approx(1.0, abs=1e-6)
    assert estimate.position.north == pytest.approx(0.5, abs=1e-6)
    # The raw, unclamped sample timestamp is still recorded.
    assert dr.last_timestamp == pytest.approx(50.0)


def test_normal_increasing_timestamps_are_unaffected_by_new_policy() -> None:
    dr = DeadReckoning()

    for step in range(10):
        dr.update(
            _sample(
                accel_x=1.0,
                accel_z=-GRAVITY_M_S2,
                timestamp=(step + 1) * 0.01,
            )
        )

    assert dr.last_update_outcome == "normal"
    assert dr.duplicate_sample_count == 0
    assert dr.stale_sample_count == 0
    assert dr.timestamp_reset_count == 0
    assert dr.clamped_dt_count == 0


def test_custom_max_dt_and_reset_threshold_are_configurable() -> None:
    dr = DeadReckoning(max_dt_seconds=0.2, timestamp_reset_threshold_seconds=0.05)
    dr.update(_sample(timestamp=0.0))  # bootstrap

    dr.update(_sample(accel_x=1.0, accel_z=-GRAVITY_M_S2, timestamp=0.3))
    assert dr.clamped_dt_count == 1

    dr.update(_sample(accel_x=1.0, accel_z=-GRAVITY_M_S2, timestamp=0.2))
    assert dr.timestamp_reset_count == 1


def test_max_dt_seconds_and_timestamp_reset_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError):
        DeadReckoning(max_dt_seconds=0.0)
    with pytest.raises(ValueError):
        DeadReckoning(timestamp_reset_threshold_seconds=-1.0)


def test_import_and_update_do_not_require_ros2() -> None:
    module = importlib.import_module("vns.core.dead_reckoning")
    dr = module.DeadReckoning()

    estimate = dr.update(_sample(timestamp=0.01), dt=0.01)

    assert estimate.source == "imu"
    assert module.DeadReckoning is DeadReckoning
