from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VnsBaseModel(BaseModel):
    """Shared base model for all VNS configuration sections."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class CameraConfig(VnsBaseModel):
    width: int = 640
    height: int = 480
    fx: float = 554.25
    fy: float = 554.25
    cx: float = 320.0
    cy: float = 240.0
    distortion: list[float] = Field(default_factory=lambda: [0.0] * 5)
    mount_position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    mount_orientation: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )

    @field_validator("distortion")
    @classmethod
    def _validate_distortion(cls, value: list[float]) -> list[float]:
        if len(value) not in {4, 5, 8}:
            raise ValueError("camera.distortion must contain 4, 5, or 8 values.")
        return [float(item) for item in value]

    @field_validator("mount_position", "mount_orientation")
    @classmethod
    def _validate_vector3(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("camera vectors must contain exactly 3 values.")
        return [float(item) for item in value]


class PreprocessingConfig(VnsBaseModel):
    target_width: int = 640
    target_height: int = 480
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    undistort: bool = True


class FeatureExtractionConfig(VnsBaseModel):
    algorithm: Literal["ORB"] = "ORB"
    max_features: int = 500
    scale_factor: float = 1.2
    n_levels: int = 8


class MatchingConfig(VnsBaseModel):
    confidence_threshold: float = 0.6
    ratio_test_threshold: float = 0.75
    min_matches: int = 10


class RetrievalBoVWConfig(VnsBaseModel):
    enabled: bool = True
    fallback_to_flann: bool = False
    vocab_size: int = 1000
    metric: Literal["cosine", "l2"] = "cosine"
    random_state: int = 42
    batch_size: int = 1000
    geo_gate: bool = True
    geo_gate_radius_deg: float = 0.002


class RetrievalConfig(VnsBaseModel):
    top_k: int = 5
    mode: Literal["flann", "bovw"] = "flann"
    backend: str = "flann_lsh"
    bovw: RetrievalBoVWConfig = Field(default_factory=RetrievalBoVWConfig)

    @staticmethod
    def _normalize_mode(value: object) -> str:
        if value is None:
            return "flann"

        normalized = str(value).strip().lower()
        if normalized in {"flann", "flann_lsh"}:
            return "flann"
        if normalized == "bovw":
            return "bovw"
        raise ValueError(
            "retrieval mode must be one of: 'flann', 'flann_lsh', or 'bovw'."
        )

    @model_validator(mode="before")
    @classmethod
    def _normalize_backend_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        mode = normalized.get("mode")
        backend = normalized.get("backend")
        normalized["mode"] = cls._normalize_mode(mode if mode is not None else backend)
        if backend is None:
            normalized["backend"] = (
                "bovw" if normalized["mode"] == "bovw" else "flann_lsh"
            )
        return normalized


class GnssConfig(VnsBaseModel):
    degraded_hdop: float = 5.0
    degraded_satellites: int = 4
    denied_timeout_seconds: float = 2.0


class NavigationConfig(VnsBaseModel):
    uncertainty_failsafe_threshold: float = 10.0
    visual_timeout_seconds: float = 300.0
    fusion_blend_duration: float = 5.0
    position_continuity_threshold: float = 2.0


class DatabaseConfig(VnsBaseModel):
    path: str = "../database/qau_campus.vnsdb"
    query_radius_deg: float = 0.001


class MavlinkConfig(VnsBaseModel):
    connection_string: str = "udp://:14551"
    system_id: int = 1
    component_id: int = 196
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    connect_timeout: float = 10.0


class RosConfig(VnsBaseModel):
    camera_topic: str = "/vns_drone/camera"
    gps_topic: str = "/vns_drone/gps"
    imu_topic: str = "/vns_drone/imu"
    vision_pose_topic: str = "/vns/vision_pose"


class LoggingConfig(VnsBaseModel):
    level: str = "INFO"
    log_dir: str = "./logs"
    format: Literal["text", "json"] = "json"
    max_size_mb: int = 100
    retention_count: int = 5
    log_to_console: bool = True
    log_images: bool = False
    log_features: bool = False


class SimulationConfig(VnsBaseModel):
    enabled: bool = True
    ground_truth_topic: str = "/vns_drone/ground_truth"
    log_ground_truth: bool = True
    position_error_threshold: float = 5.0


class GeoReferenceConfig(VnsBaseModel):
    origin_latitude: float = 33.7470
    origin_longitude: float = 73.1370
    origin_altitude: float = 550.0


class VnsConfig(VnsBaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    feature_extraction: FeatureExtractionConfig = Field(
        default_factory=FeatureExtractionConfig
    )
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    gnss: GnssConfig = Field(default_factory=GnssConfig)
    navigation: NavigationConfig = Field(default_factory=NavigationConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    mavlink: MavlinkConfig = Field(default_factory=MavlinkConfig)
    ros: RosConfig = Field(default_factory=RosConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    geo_reference: GeoReferenceConfig = Field(default_factory=GeoReferenceConfig)

    @model_validator(mode="after")
    def _validate_camera_center(self) -> "VnsConfig":
        camera = self.camera
        if camera.width <= 0 or camera.height <= 0:
            raise ValueError("camera.width and camera.height must be positive.")
        if self.preprocessing.target_width <= 0 or self.preprocessing.target_height <= 0:
            raise ValueError(
                "preprocessing.target_width and target_height must be positive."
            )
        if camera.cx < 0 or camera.cy < 0:
            raise ValueError("camera.cx and camera.cy must be non-negative.")
        return self
