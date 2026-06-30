import importlib
import math

import pytest

from vns.core.dead_reckoning import DeadReckoning
from vns.core.models import IMUSample


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


def test_import_and_update_do_not_require_ros2() -> None:
    module = importlib.import_module("vns.core.dead_reckoning")
    dr = module.DeadReckoning()

    estimate = dr.update(_sample(timestamp=0.01), dt=0.01)

    assert estimate.source == "imu"
    assert module.DeadReckoning is DeadReckoning
