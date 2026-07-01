"""Tests for VnsNode ROS2 callback robustness.

Skips automatically wherever the ROS2 Python stack (rclpy/sensor_msgs/etc.)
is not importable, matching this repo's existing optional-dependency pattern
(see tests/unit/test_fr1_map_loader.py, tests/unit/test_faiss_database.py).

These tests call ``VnsNode._on_imu`` directly against a minimal instance
created via ``VnsNode.__new__`` (no ``rclpy.init()``/full ``Node``
construction, config, or database required) so they stay fast and avoid the
ROS2-environment fragility called out in specs/FR-11.md criterion 1.
"""

import pytest

vns_node_module = pytest.importorskip("vns.core.vns_node")
VnsNode = vns_node_module.VnsNode


class _FakeRuntime:
    def __init__(self, *, raises: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._raises = raises

    def update_imu(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises


class _FakeLogger:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message) -> None:
        self.errors.append(message)


class _Vector3:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class _Quaternion:
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.w, self.x, self.y, self.z = w, x, y, z


class _Stamp:
    def __init__(self, sec: int = 1, nanosec: int = 0) -> None:
        self.sec, self.nanosec = sec, nanosec


class _Header:
    def __init__(self, stamp: _Stamp) -> None:
        self.stamp = stamp


class _FakeImu:
    def __init__(self, *, sec: int = 1, nanosec: int = 0) -> None:
        self.header = _Header(_Stamp(sec, nanosec))
        self.orientation = _Quaternion()
        self.linear_acceleration = _Vector3(z=-9.80665)
        self.angular_velocity = _Vector3()


def _make_node(runtime: _FakeRuntime) -> tuple[VnsNode, _FakeLogger]:
    node = VnsNode.__new__(VnsNode)
    node._runtime = runtime
    logger = _FakeLogger()
    node.get_logger = lambda: logger
    return node, logger


def test_on_imu_forwards_fields_and_timestamp_to_runtime() -> None:
    runtime = _FakeRuntime()
    node, logger = _make_node(runtime)

    VnsNode._on_imu(node, _FakeImu(sec=5, nanosec=500_000_000))

    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    assert call["timestamp"] == pytest.approx(5.5)
    assert call["accel_z"] == pytest.approx(-9.80665)
    assert logger.errors == []


def test_on_imu_treats_all_zero_stamp_as_missing_timestamp() -> None:
    runtime = _FakeRuntime()
    node, _logger = _make_node(runtime)

    VnsNode._on_imu(node, _FakeImu(sec=0, nanosec=0))

    assert runtime.calls[0]["timestamp"] is None


def test_on_imu_duplicate_timestamp_does_not_raise() -> None:
    runtime = _FakeRuntime()
    node, logger = _make_node(runtime)

    VnsNode._on_imu(node, _FakeImu(sec=1))
    VnsNode._on_imu(node, _FakeImu(sec=1))

    assert len(runtime.calls) == 2
    assert logger.errors == []


def test_on_imu_swallows_unexpected_exception_and_logs_error() -> None:
    runtime = _FakeRuntime(raises=ValueError("boom"))
    node, logger = _make_node(runtime)

    VnsNode._on_imu(node, _FakeImu(sec=1))

    assert len(logger.errors) == 1
    assert "IMU callback failed" in logger.errors[0]
