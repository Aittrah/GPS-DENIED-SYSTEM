from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import time

import numpy as np

from .geo_utils import NEDPoint
from .models import IMUSample, UAVState


class DeadReckoning:
    """Standalone strapdown inertial propagator in the local NED frame.

    The class maintains a dead-reckoned pose estimate using body-frame
    accelerometer and gyro inputs. Inputs are expected to be calibrated body
    specific force and angular velocity in the forward-right-down body frame:
    accelerometer units are ``m/s^2`` and gyro units are ``rad/s``.
    """

    def __init__(
        self,
        initial_state: UAVState | None = None,
        *,
        gravity_m_s2: float = 9.80665,
    ) -> None:
        self._gravity_m_s2 = float(gravity_m_s2)
        self._estimate = self._resolve_initial_state(initial_state)
        self._orientation_quaternion = self._euler_to_quaternion(
            self._estimate.roll,
            self._estimate.pitch,
            self._estimate.yaw,
        )
        self._last_timestamp: float | None = None

    @property
    def initialized(self) -> bool:
        return True

    @property
    def orientation_quaternion(self) -> tuple[float, float, float, float]:
        return tuple(float(component) for component in self._orientation_quaternion)

    def reset(self, initial_state: UAVState | None = None) -> None:
        self._estimate = self._resolve_initial_state(initial_state)
        self._orientation_quaternion = self._euler_to_quaternion(
            self._estimate.roll,
            self._estimate.pitch,
            self._estimate.yaw,
        )
        self._last_timestamp = None

    def update(
        self,
        sample: IMUSample,
        *,
        dt: float | None = None,
    ) -> UAVState:
        step_dt = self._resolve_dt(sample.timestamp, dt)
        if step_dt is None:
            return self.get_estimate()

        gyro_body = np.array(
            [sample.gyro_x, sample.gyro_y, sample.gyro_z],
            dtype=np.float64,
        )
        accel_body = np.array(
            [sample.accel_x, sample.accel_y, sample.accel_z],
            dtype=np.float64,
        )

        delta_quaternion = self._delta_quaternion(gyro_body, step_dt)
        self._orientation_quaternion = self._normalize_quaternion(
            self._quaternion_multiply(
                self._orientation_quaternion,
                delta_quaternion,
            )
        )

        rotation_body_to_ned = self._quaternion_to_rotation_matrix(
            self._orientation_quaternion
        )
        gravity_ned = np.array([0.0, 0.0, self._gravity_m_s2], dtype=np.float64)
        acceleration_ned = rotation_body_to_ned @ accel_body + gravity_ned

        previous_position = np.array(
            [
                self._estimate.position.north,
                self._estimate.position.east,
                self._estimate.position.down,
            ],
            dtype=np.float64,
        )
        previous_velocity = np.array(
            [
                self._estimate.velocity_north,
                self._estimate.velocity_east,
                self._estimate.velocity_down,
            ],
            dtype=np.float64,
        )

        new_position = (
            previous_position
            + previous_velocity * step_dt
            + 0.5 * acceleration_ned * step_dt * step_dt
        )
        new_velocity = previous_velocity + acceleration_ned * step_dt
        roll, pitch, yaw = self._quaternion_to_euler(self._orientation_quaternion)
        timestamp = sample.timestamp if sample.timestamp is not None else time.time()

        self._estimate = replace(
            self._estimate,
            position=NEDPoint(
                north=float(new_position[0]),
                east=float(new_position[1]),
                down=float(new_position[2]),
            ),
            velocity_north=float(new_velocity[0]),
            velocity_east=float(new_velocity[1]),
            velocity_down=float(new_velocity[2]),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            source="imu",
            timestamp=timestamp,
        )
        self._last_timestamp = timestamp
        return self.get_estimate()

    def get_estimate(self) -> UAVState:
        return replace(
            self._estimate,
            position=NEDPoint(
                north=self._estimate.position.north,
                east=self._estimate.position.east,
                down=self._estimate.position.down,
            ),
        )

    def _resolve_initial_state(self, initial_state: UAVState | None) -> UAVState:
        if initial_state is None:
            return UAVState(
                position=NEDPoint(0.0, 0.0, 0.0),
                velocity_north=0.0,
                velocity_east=0.0,
                velocity_down=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                confidence=0.0,
                source="imu",
            )

        state = deepcopy(initial_state)
        state.source = "imu"
        return state

    def _resolve_dt(
        self,
        timestamp: float | None,
        dt: float | None,
    ) -> float | None:
        if dt is not None:
            if dt <= 0.0:
                raise ValueError("dt must be positive.")
            return float(dt)

        if timestamp is None:
            raise ValueError("timestamp is required when dt is not provided.")

        if self._last_timestamp is None:
            self._last_timestamp = float(timestamp)
            self._estimate = replace(self._estimate, timestamp=float(timestamp))
            return None

        step_dt = float(timestamp) - self._last_timestamp
        if step_dt <= 0.0:
            raise ValueError("IMU sample timestamps must be strictly increasing.")
        return step_dt

    @staticmethod
    def _delta_quaternion(gyro_body: np.ndarray, dt: float) -> np.ndarray:
        rotation_vector = gyro_body * dt
        rotation_magnitude = float(np.linalg.norm(rotation_vector))
        if rotation_magnitude < 1e-12:
            return np.array(
                [
                    1.0,
                    0.5 * rotation_vector[0],
                    0.5 * rotation_vector[1],
                    0.5 * rotation_vector[2],
                ],
                dtype=np.float64,
            )

        half_angle = 0.5 * rotation_magnitude
        axis = rotation_vector / rotation_magnitude
        sin_half = math.sin(half_angle)
        return np.array(
            [
                math.cos(half_angle),
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(quaternion))
        if norm == 0.0:
            raise ValueError("orientation quaternion cannot be zero.")
        return quaternion / norm

    @staticmethod
    def _quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = lhs
        rw, rx, ry, rz = rhs
        return np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
        w, x, y, z = quaternion
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quaternion_to_euler(quaternion: np.ndarray) -> tuple[float, float, float]:
        w, x, y, z = quaternion

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        half_roll = 0.5 * roll
        half_pitch = 0.5 * pitch
        half_yaw = 0.5 * yaw

        cr = math.cos(half_roll)
        sr = math.sin(half_roll)
        cp = math.cos(half_pitch)
        sp = math.sin(half_pitch)
        cy = math.cos(half_yaw)
        sy = math.sin(half_yaw)

        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
