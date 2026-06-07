# src/vns/core/models.py
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np
import time
from .geo_utils import NEDPoint, GeoPoint

@dataclass
class IMUSample:
    """Raw IMU reading from sensor"""
    accel_x: float      # m/s²
    accel_y: float      # m/s²
    accel_z: float      # m/s²
    gyro_x: float       # rad/s
    gyro_y: float       # rad/s
    gyro_z: float       # rad/s
    timestamp: float = field(default_factory=time.time)

@dataclass
class UAVState:
    """Complete UAV state estimate — output of sensor fusion"""
    position: NEDPoint
    velocity_north: float = 0.0   # m/s
    velocity_east: float = 0.0    # m/s
    velocity_down: float = 0.0    # m/s
    roll: float = 0.0             # radians
    pitch: float = 0.0            # radians
    yaw: float = 0.0              # radians
    confidence: float = 0.0       # 0.0 to 1.0
    source: Literal["vision","imu","fused","unknown"] = "unknown"
    timestamp: float = field(default_factory=time.time)

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 0.3

@dataclass
class ImagePatch:
    """A fixed-size image patch with location metadata"""
    data: np.ndarray              # (256, 256, 3) BGR
    center_position: NEDPoint
    patch_id: str
    source: Literal["uav", "satellite"]

@dataclass
class FeatureSet:
    """Extracted features from an image patch"""
    keypoints: np.ndarray         # (N, 2) pixel coordinates
    descriptors: np.ndarray       # (N, 32) ORB descriptors
    patch_id: str
    position: NEDPoint

@dataclass
class VisualMatch:
    """Result of matching UAV image against satellite reference"""
    estimated_position: NEDPoint
    confidence: float             # 0.0 to 1.0
    matched_reference_id: str
    num_inliers: int
    timestamp: float = field(default_factory=time.time)

    @property
    def is_valid(self) -> bool:
        return self.confidence >= 0.3 and self.num_inliers >= 10

@dataclass
class NavigationCommand:
    """Control output sent to PX4"""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0
    mode: Literal["normal","loiter","land","failsafe"] = "normal"
    timestamp: float = field(default_factory=time.time)

@dataclass
class Waypoint:
    """A mission waypoint"""
    position: NEDPoint
    waypoint_id: str
    tolerance_m: float = 2.0      # arrival radius in meters

    def is_reached(self, current: NEDPoint) -> bool:
        dn = current.north - self.position.north
        de = current.east - self.position.east
        return np.sqrt(dn**2 + de**2) <= self.tolerance_m