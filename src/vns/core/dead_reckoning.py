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
        gyro_bias_rad_s: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
        accelerometer_bias_m_s2: tuple[float, float, float] | list[float] = (
            0.0,
            0.0,
            0.0,
        ),
        max_bridge_duration_seconds: float = 10.0,
        confidence_decay_per_second: float = 0.1,
        max_dt_seconds: float = 1.0,
        timestamp_reset_threshold_seconds: float = 1.0,
    ) -> None:
        self._gravity_m_s2 = float(gravity_m_s2)
        self._gyro_bias_rad_s = self._coerce_vector3(gyro_bias_rad_s, "gyro_bias_rad_s")
        self._accelerometer_bias_m_s2 = self._coerce_vector3(
            accelerometer_bias_m_s2,
            "accelerometer_bias_m_s2",
        )
        self._max_bridge_duration_seconds = float(max_bridge_duration_seconds)
        if self._max_bridge_duration_seconds <= 0.0:
            raise ValueError("max_bridge_duration_seconds must be positive.")
        self._confidence_decay_per_second = float(confidence_decay_per_second)
        if self._confidence_decay_per_second < 0.0:
            raise ValueError("confidence_decay_per_second must be non-negative.")
        self._max_dt_seconds = float(max_dt_seconds)
        if self._max_dt_seconds <= 0.0:
            raise ValueError("max_dt_seconds must be positive.")
        self._timestamp_reset_threshold_seconds = float(timestamp_reset_threshold_seconds)
        if self._timestamp_reset_threshold_seconds <= 0.0:
            raise ValueError("timestamp_reset_threshold_seconds must be positive.")
        self._estimate = self._resolve_initial_state(initial_state)
        self._orientation_quaternion = self._euler_to_quaternion(
            self._estimate.roll,
            self._estimate.pitch,
            self._estimate.yaw,
        )
        self._last_timestamp: float | None = None
        self._last_trusted_correction_timestamp: float | None = None
        self._base_correction_confidence = 0.0
        self._duplicate_sample_count = 0
        self._stale_sample_count = 0
        self._timestamp_reset_count = 0
        self._clamped_dt_count = 0
        self._last_update_outcome = "uninitialized"
        self._sync_state(
            self._estimate,
            trusted_correction=initial_state is not None and initial_state.confidence > 0.0,
        )

    @property
    def initialized(self) -> bool:
        return True

    @property
    def orientation_quaternion(self) -> tuple[float, float, float, float]:
        return tuple(float(component) for component in self._orientation_quaternion)

    @property
    def gyro_bias_rad_s(self) -> tuple[float, float, float]:
        return tuple(float(component) for component in self._gyro_bias_rad_s)

    @property
    def accelerometer_bias_m_s2(self) -> tuple[float, float, float]:
        return tuple(float(component) for component in self._accelerometer_bias_m_s2)

    @property
    def last_trusted_correction_timestamp(self) -> float | None:
        return self._last_trusted_correction_timestamp

    @property
    def last_timestamp(self) -> float | None:
        return self._last_timestamp

    @property
    def duplicate_sample_count(self) -> int:
        return self._duplicate_sample_count

    @property
    def stale_sample_count(self) -> int:
        return self._stale_sample_count

    @property
    def timestamp_reset_count(self) -> int:
        return self._timestamp_reset_count

    @property
    def clamped_dt_count(self) -> int:
        return self._clamped_dt_count

    @property
    def last_update_outcome(self) -> str:
        """One of: explicit_dt, bootstrap, normal, duplicate, stale, reset, clamped."""
        return self._last_update_outcome

    def reset(self, initial_state: UAVState | None = None) -> None:
        state = self._resolve_initial_state(initial_state)
        self._orientation_quaternion = self._euler_to_quaternion(
            state.roll,
            state.pitch,
            state.yaw,
        )
        self._sync_state(
            state,
            trusted_correction=initial_state is not None and initial_state.confidence > 0.0,
        )

    def correct(self, corrected_state: UAVState) -> UAVState:
        self._orientation_quaternion = self._euler_to_quaternion(
            corrected_state.roll,
            corrected_state.pitch,
            corrected_state.yaw,
        )
        self._sync_state(
            self._resolve_initial_state(corrected_state),
            trusted_correction=True,
        )
        return self.get_estimate(current_time=corrected_state.timestamp)

    def update(
        self,
        sample: IMUSample,
        *,
        dt: float | None = None,
    ) -> UAVState:
        step_dt = self._resolve_dt(sample.timestamp, dt)
        if step_dt is None:
            return self.get_estimate(current_time=self._estimate.timestamp)

        gyro_body = np.array(
            [sample.gyro_x, sample.gyro_y, sample.gyro_z],
            dtype=np.float64,
        ) - self._gyro_bias_rad_s
        accel_body = np.array(
            [sample.accel_x, sample.accel_y, sample.accel_z],
            dtype=np.float64,
        ) - self._accelerometer_bias_m_s2

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
            confidence=self._confidence_at(timestamp),
            source="imu",
            timestamp=timestamp,
        )
        self._last_timestamp = timestamp
        return self.get_estimate(current_time=timestamp)

    def get_estimate(self, *, current_time: float | None = None) -> UAVState:
        estimate_time = (
            self._estimate.timestamp if current_time is None else float(current_time)
        )
        return replace(
            self._estimate,
            position=NEDPoint(
                north=self._estimate.position.north,
                east=self._estimate.position.east,
                down=self._estimate.position.down,
            ),
            confidence=self._confidence_at(estimate_time),
        )

    def time_since_last_trusted_correction(
        self,
        *,
        current_time: float | None = None,
    ) -> float | None:
        if self._last_trusted_correction_timestamp is None:
            return None
        now = self._estimate.timestamp if current_time is None else float(current_time)
        return max(0.0, now - self._last_trusted_correction_timestamp)

    def is_bridge_valid(self, *, current_time: float | None = None) -> bool:
        age = self.time_since_last_trusted_correction(current_time=current_time)
        if age is None:
            return False
        return age <= self._max_bridge_duration_seconds

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

    def _sync_state(
        self,
        state: UAVState,
        *,
        trusted_correction: bool,
    ) -> None:
        self._estimate = replace(
            state,
            confidence=self._clamp_confidence(state.confidence),
            source="imu",
        )
        if trusted_correction:
            correction_timestamp = float(self._estimate.timestamp)
            self._last_timestamp = correction_timestamp
            self._last_trusted_correction_timestamp = correction_timestamp
            self._base_correction_confidence = self._clamp_confidence(
                self._estimate.confidence
            )
        else:
            self._last_timestamp = None
            self._last_trusted_correction_timestamp = None
            self._base_correction_confidence = 0.0
        self._estimate = replace(
            self._estimate,
            confidence=self._confidence_at(self._estimate.timestamp),
        )

    def _confidence_at(self, current_time: float) -> float:
        if not self.is_bridge_valid(current_time=current_time):
            return 0.0
        age = self.time_since_last_trusted_correction(current_time=current_time)
        if age is None:
            return 0.0
        return self._clamp_confidence(
            self._base_correction_confidence
            - self._confidence_decay_per_second * age
        )

    def _resolve_dt(
        self,
        timestamp: float | None,
        dt: float | None,
    ) -> float | None:
        if dt is not None:
            if dt <= 0.0:
                raise ValueError("dt must be positive.")
            self._last_update_outcome = "explicit_dt"
            return float(dt)

        if timestamp is None:
            raise ValueError("timestamp is required when dt is not provided.")

        if self._last_timestamp is None:
            self._last_timestamp = float(timestamp)
            self._estimate = replace(self._estimate, timestamp=float(timestamp))
            self._last_update_outcome = "bootstrap"
            return None

        step_dt = float(timestamp) - self._last_timestamp

        if step_dt == 0.0:
            self._duplicate_sample_count += 1
            self._last_update_outcome = "duplicate"
            return None

        if step_dt < 0.0:
            if -step_dt >= self._timestamp_reset_threshold_seconds:
                self._last_timestamp = float(timestamp)
                self._estimate = replace(self._estimate, timestamp=float(timestamp))
                self._timestamp_reset_count += 1
                self._last_update_outcome = "reset"
                return None
            self._stale_sample_count += 1
            self._last_update_outcome = "stale"
            return None

        if step_dt > self._max_dt_seconds:
            self._clamped_dt_count += 1
            self._last_update_outcome = "clamped"
            return self._max_dt_seconds

        self._last_update_outcome = "normal"
        return step_dt

    @staticmethod
    def _coerce_vector3(
        value: tuple[float, float, float] | list[float],
        field_name: str,
    ) -> np.ndarray:
        if len(value) != 3:
            raise ValueError(f"{field_name} must contain exactly 3 values.")
        return np.array([float(component) for component in value], dtype=np.float64)

    @staticmethod
    def _clamp_confidence(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

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
