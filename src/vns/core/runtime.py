from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from copy import deepcopy
from typing import Mapping, Optional

import numpy as np

from vns.config import ConfigManager, ConfigValue, VnsConfig
from vns.core.blender import PositionBlender
from vns.core.diagnostics import SubsystemDiagnostic, VnsDiagnostics
from vns.core.gnss_monitor import GnssMonitor
from vns.database.reference_db import ReferenceDatabase
from vns.interfaces.mavlink_interface import MAVLinkInterface
from vns.vision.localizer import VisualLocalizer
from vns.vision.types import LocalizationResult


@dataclass
class FrameProcessingResult:
    """Result of processing a single camera frame through the runtime."""

    success: bool
    navigation_mode: str
    reason: str
    blended_pose: Optional[tuple[float, float, float]]
    localization: LocalizationResult


class VnsRuntime:
    """Own runtime state shared across ROS, scripts, and tests."""

    def __init__(
        self,
        config: ConfigManager | VnsConfig | Mapping[str, object],
        database: ReferenceDatabase | None = None,
        mavlink: MAVLinkInterface | None = None,
    ) -> None:
        if isinstance(config, ConfigManager):
            self._config_model = config.model
            self._config_dict = config.data
        elif isinstance(config, VnsConfig):
            self._config_model = config.model_copy(deep=True)
            self._config_dict = self._config_model.model_dump(mode="python")
        else:
            self._config_model = VnsConfig.model_validate(config)
            self._config_dict = self._config_model.model_dump(mode="python")

        self._database = database
        self._mavlink = mavlink
        self._localizer = (
            VisualLocalizer(self._config_dict, database) if database is not None else None
        )
        self._gnss_monitor = GnssMonitor(
            degraded_hdop=self._config_model.gnss.degraded_hdop,
            degraded_satellites=self._config_model.gnss.degraded_satellites,
            denied_timeout_seconds=self._config_model.gnss.denied_timeout_seconds,
        )
        self._blender = PositionBlender(
            uncertainty_failsafe_threshold=(
                self._config_model.navigation.uncertainty_failsafe_threshold
            ),
            visual_timeout_seconds=self._config_model.navigation.visual_timeout_seconds,
            fusion_blend_duration=self._config_model.navigation.fusion_blend_duration,
            position_continuity_threshold=(
                self._config_model.navigation.position_continuity_threshold
            ),
        )

        self._last_gps: tuple[float, float, float] | None = None
        self._last_altitude = self._config_model.geo_reference.origin_altitude
        self._last_heading_deg = 0.0
        self._last_yaw_rad = 0.0
        self._last_navigation_mode = "UNINITIALIZED"
        self._last_localization_reason: str | None = None
        self._last_localization: LocalizationResult | None = None
        self._last_blended_pose: tuple[float, float, float] | None = None
        self._coverage_gap_detected = False

    @property
    def config(self) -> dict[str, ConfigValue]:
        return deepcopy(self._config_dict)

    @property
    def config_model(self) -> VnsConfig:
        return self._config_model.model_copy(deep=True)

    @property
    def database(self) -> ReferenceDatabase | None:
        return self._database

    @property
    def localizer(self) -> VisualLocalizer | None:
        return self._localizer

    @property
    def gnss_monitor(self) -> GnssMonitor:
        return self._gnss_monitor

    @property
    def blender(self) -> PositionBlender:
        return self._blender

    @property
    def geo_origin(self) -> dict[str, float]:
        return deepcopy(self._config_dict["geo_reference"])  # type: ignore[return-value]

    @property
    def last_gps(self) -> tuple[float, float, float] | None:
        return self._last_gps

    @property
    def last_heading_deg(self) -> float:
        return self._last_heading_deg

    @property
    def last_yaw_rad(self) -> float:
        return self._last_yaw_rad

    def attach_mavlink(self, mavlink: MAVLinkInterface) -> None:
        self._mavlink = mavlink

    def update_gps(
        self,
        *,
        latitude: float,
        longitude: float,
        altitude: float,
        has_fix: bool,
        num_satellites: int,
        hdop: float,
        timestamp: float | None = None,
    ) -> None:
        self._last_gps = (latitude, longitude, altitude)
        self._last_altitude = altitude
        self._gnss_monitor.update(
            has_fix=has_fix,
            num_satellites=num_satellites,
            hdop=hdop,
            timestamp=timestamp,
        )

    def update_heading_from_quaternion(
        self,
        *,
        w: float,
        x: float,
        y: float,
        z: float,
    ) -> None:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        self._last_yaw_rad = yaw_rad
        self._last_heading_deg = math.degrees(yaw_rad)

    def update_ground_truth_altitude(self, local_up_m: float) -> None:
        self._last_altitude = self._config_model.geo_reference.origin_altitude + local_up_m

    def check_gnss_timeout(self, current_time: float | None = None) -> None:
        self._gnss_monitor.check_timeout(current_time=current_time)

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> FrameProcessingResult:
        ts = time.time() if timestamp is None else timestamp
        if self._localizer is None:
            localization = LocalizationResult(
                success=False,
                confidence=0.0,
                inlier_count=0,
                matched_ref_id=None,
                reason="database_unavailable",
                pose_ned=None,
                pose_geodetic=None,
                yaw_rad=0.0,
                timestamp=ts,
            )
            self._last_localization = localization
            self._last_localization_reason = localization.reason
            self._last_navigation_mode = "FAILSAFE"
            self._last_blended_pose = None
            self._coverage_gap_detected = False
            return FrameProcessingResult(
                success=False,
                navigation_mode=self._last_navigation_mode,
                reason=localization.reason,
                blended_pose=None,
                localization=localization,
            )

        localization = self._localizer.localize(
            frame,
            altitude=self._last_altitude,
            heading_deg=self._last_heading_deg,
        )
        self._last_localization = localization

        if not localization.success:
            reason = localization.reason
            self._coverage_gap_detected = self._detect_coverage_gap()
            if self._coverage_gap_detected:
                reason = "coverage_gap"
            self._last_localization_reason = reason
            localization = replace(localization, reason=reason)
            self._last_localization = localization
            self._last_blended_pose = None
            return FrameProcessingResult(
                success=False,
                navigation_mode=self._last_navigation_mode,
                reason=reason,
                blended_pose=None,
                localization=localization,
            )

        assert localization.pose_geodetic is not None
        blended_pose, nav_mode = self._blender.blend(
            self._last_gps,
            localization.pose_geodetic,
            self._gnss_monitor.state,
            ts,
        )
        self._coverage_gap_detected = False
        self._last_navigation_mode = nav_mode
        self._last_localization_reason = localization.reason
        self._last_blended_pose = blended_pose
        return FrameProcessingResult(
            success=True,
            navigation_mode=nav_mode,
            reason=localization.reason,
            blended_pose=blended_pose,
            localization=localization,
        )

    def get_diagnostics(self) -> VnsDiagnostics:
        db_loaded = self._database is not None
        mission_status = self._mavlink.status if self._mavlink is not None else None
        subsystems = [
            SubsystemDiagnostic(
                name="database",
                healthy=db_loaded,
                state="READY" if db_loaded else "MISSING",
                details={
                    "entry_count": 0 if self._database is None else self._database.entry_count,
                },
            ),
            SubsystemDiagnostic(
                name="gnss",
                healthy=self._gnss_monitor.state.name != "DENIED",
                state=self._gnss_monitor.state.name,
                details={
                    "last_gps": self._last_gps,
                },
            ),
            SubsystemDiagnostic(
                name="navigation",
                healthy=not self._blender.failsafe_active,
                state=self._last_navigation_mode,
                details={
                    "coverage_gap_detected": self._coverage_gap_detected,
                    "last_reason": self._last_localization_reason,
                },
            ),
        ]
        if mission_status is not None:
            subsystems.append(
                SubsystemDiagnostic(
                    name="mavlink",
                    healthy=mission_status.connected,
                    state="CONNECTED" if mission_status.connected else "DISCONNECTED",
                    details={
                        "armed": mission_status.armed,
                        "in_air": mission_status.in_air,
                        "flight_mode": mission_status.flight_mode,
                        "mission_current": mission_status.mission_current,
                        "mission_total": mission_status.mission_total,
                        "last_error": mission_status.last_error,
                    },
                )
            )

        return VnsDiagnostics(
            localizer_ready=self._localizer is not None,
            database_entry_count=0 if self._database is None else self._database.entry_count,
            gnss_state=self._gnss_monitor.state.name,
            navigation_mode=self._last_navigation_mode,
            last_localization_reason=self._last_localization_reason,
            coverage_gap_detected=self._coverage_gap_detected,
            failsafe_active=self._blender.failsafe_active,
            subsystems=subsystems,
        )

    def _detect_coverage_gap(self) -> bool:
        if self._database is None or self._last_gps is None:
            return False
        lat, lon, _alt = self._last_gps
        if not self._database.bounds.contains(lat, lon):
            return True
        nearby = self._database.query_region(
            lat,
            lon,
            self._config_model.database.query_radius_deg,
        )
        return len(nearby) == 0
