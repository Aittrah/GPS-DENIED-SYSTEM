import logging
import math
import time
from dataclasses import replace
from typing import Mapping, Optional, Tuple

from vns.core.dead_reckoning import DeadReckoning
from vns.core.gnss_monitor import GnssState
from vns.core.models import NEDPoint, UAVState
from vns.utils.coordinates import enu_to_geodetic

logger = logging.getLogger("vns.core")


GeoPose = Tuple[float, float, float]
NedPose = Tuple[float, float, float]


class PositionBlender:
    """Blends and transitions position estimates between GNSS and Visual Navigation."""

    def __init__(
        self,
        uncertainty_failsafe_threshold: float = 10.0,
        visual_timeout_seconds: float = 300.0,
        fusion_blend_duration: float = 5.0,
        position_continuity_threshold: float = 2.0
    ) -> None:
        self.uncertainty_failsafe_threshold = uncertainty_failsafe_threshold
        self.visual_timeout_seconds = visual_timeout_seconds
        self.fusion_blend_duration = fusion_blend_duration
        self.position_continuity_threshold = position_continuity_threshold
        
        self._last_estimate: Optional[Tuple[float, float, float]] = None
        self._last_gps: Optional[Tuple[float, float, float]] = None
        self._last_visual: Optional[Tuple[float, float, float]] = None
        
        self._last_visual_time = 0.0
        self._transition_start_time: Optional[float] = None
        self._failsafe_active = False

    def blend(
        self,
        gps_pos: Optional[GeoPose],
        visual_pos: Optional[GeoPose],
        gnss_state: GnssState,
        current_time: Optional[float] = None,
        *,
        dead_reckoning: DeadReckoning | None = None,
        dead_reckoning_estimate: UAVState | None = None,
        dead_reckoning_ready: bool = False,
        visual_pose_ned: NedPose | None = None,
        visual_confidence: float | None = None,
        geo_origin: Mapping[str, float] | None = None,
    ) -> Tuple[GeoPose, str]:
        """
        Blend GPS and Visual positions based on GNSS health state.
        Returns:
            blended_position: (lat, lon, alt)
            mode: "GPS", "BLENDING", "VISION", "DR", or "FAILSAFE"
        """
        now = time.time() if current_time is None else current_time

        if dead_reckoning is not None and geo_origin is not None:
            return self._blend_dead_reckoning(
                gps_pos,
                visual_pos,
                gnss_state,
                now,
                dead_reckoning=dead_reckoning,
                dead_reckoning_estimate=dead_reckoning_estimate,
                dead_reckoning_ready=dead_reckoning_ready,
                visual_pose_ned=visual_pose_ned,
                visual_confidence=visual_confidence,
                geo_origin=geo_origin,
            )

        return self._blend_legacy(gps_pos, visual_pos, gnss_state, now)

    def _blend_dead_reckoning(
        self,
        gps_pos: Optional[GeoPose],
        visual_pos: Optional[GeoPose],
        gnss_state: GnssState,
        now: float,
        *,
        dead_reckoning: DeadReckoning,
        dead_reckoning_estimate: UAVState | None,
        dead_reckoning_ready: bool,
        visual_pose_ned: NedPose | None,
        visual_confidence: float | None,
        geo_origin: Mapping[str, float],
    ) -> Tuple[GeoPose, str]:
        self._transition_start_time = None
        if gps_pos is not None:
            self._last_gps = gps_pos

        accepted_visual = self._accept_visual_fix(visual_pos, now)

        if gnss_state == GnssState.HEALTHY:
            self._failsafe_active = False
            if self._last_gps is not None:
                self._last_estimate = self._last_gps
                return self._last_gps, "GPS"
            return self._activate_failsafe()

        if accepted_visual is not None and visual_pose_ned is not None:
            self._reset_dead_reckoning(
                dead_reckoning,
                dead_reckoning_estimate,
                visual_pose_ned,
                visual_confidence,
                now,
            )
            self._last_estimate = accepted_visual
            self._failsafe_active = False
            return accepted_visual, "VISION"

        if accepted_visual is not None and visual_pose_ned is None:
            logger.warning("Visual pose correction missing NED pose; continuing with DR.")

        dr_ready = dead_reckoning_ready or self._last_estimate is not None
        if dead_reckoning_estimate is not None and dr_ready:
            dr_position = self._dead_reckoning_to_geodetic(
                dead_reckoning_estimate,
                geo_origin,
            )
            self._last_estimate = dr_position
            self._failsafe_active = False
            return dr_position, "DR"

        return self._activate_failsafe()

    def _blend_legacy(
        self,
        gps_pos: Optional[GeoPose],
        visual_pos: Optional[GeoPose],
        gnss_state: GnssState,
        now: float,
    ) -> Tuple[GeoPose, str]:
        """Maintain the legacy GPS/visual transition behavior for older callers."""
        
        # Track last inputs
        if gps_pos is not None:
            self._last_gps = gps_pos
        self._accept_visual_fix(visual_pos, now)

        # Evaluate failsafe
        time_since_visual = now - self._last_visual_time
        if gnss_state == GnssState.DENIED and (self._last_visual is None or time_since_visual > self.visual_timeout_seconds):
            return self._activate_failsafe()

        self._failsafe_active = False

        # Blending State Machine
        if gnss_state == GnssState.HEALTHY:
            # reset transition
            self._transition_start_time = None
            if self._last_gps is not None:
                self._last_estimate = self._last_gps
                return self._last_gps, "GPS"
            else:
                return (0.0, 0.0, 0.0), "FAILSAFE"
                
        # GNSS is Degraded or Denied
        if self._last_visual is None:
            # No visual navigation yet, fallback to last GPS if available
            fallback = self._last_gps or (0.0, 0.0, 0.0)
            return fallback, "GPS"
            
        # Visual is available, GNSS is degraded/denied: start blending
        if self._transition_start_time is None:
            self._transition_start_time = now
            logger.info("Starting blending transition from GPS to Visual Navigation.")
            
        elapsed = now - self._transition_start_time
        
        if elapsed >= self.fusion_blend_duration:
            # Fully transitioned to Vision
            self._last_estimate = self._last_visual
            return self._last_visual, "VISION"
            
        # Blending active
        w_vis = elapsed / self.fusion_blend_duration
        anchor_gps = self._last_gps or self._last_visual # fallback if no gps
        
        blended_lat = (1.0 - w_vis) * anchor_gps[0] + w_vis * self._last_visual[0]
        blended_lon = (1.0 - w_vis) * anchor_gps[1] + w_vis * self._last_visual[1]
        blended_alt = (1.0 - w_vis) * anchor_gps[2] + w_vis * self._last_visual[2]
        
        blended_pos = (blended_lat, blended_lon, blended_alt)
        self._last_estimate = blended_pos
        return blended_pos, "BLENDING"

    def _accept_visual_fix(
        self,
        visual_pos: Optional[GeoPose],
        now: float,
    ) -> Optional[GeoPose]:
        if visual_pos is None:
            return None

        if self._last_estimate is not None:
            dist_m = self._horizontal_distance_m(visual_pos, self._last_estimate)
            if dist_m > self.position_continuity_threshold:
                logger.warning(
                    "Visual position rejected due to discontinuity: %.2fm",
                    dist_m,
                )
                return None

        self._last_visual = visual_pos
        self._last_visual_time = now
        return visual_pos

    def _reset_dead_reckoning(
        self,
        dead_reckoning: DeadReckoning,
        estimate: UAVState | None,
        visual_pose_ned: NedPose,
        visual_confidence: float | None,
        timestamp: float,
    ) -> None:
        base_state = estimate if estimate is not None else dead_reckoning.get_estimate()
        corrected_state = replace(
            base_state,
            position=NEDPoint(
                north=float(visual_pose_ned[0]),
                east=float(visual_pose_ned[1]),
                down=float(visual_pose_ned[2]),
            ),
            confidence=max(base_state.confidence, float(visual_confidence or 0.0)),
            timestamp=timestamp,
        )
        dead_reckoning.reset(corrected_state)
        logger.info("Dead reckoning reset from visual correction.")

    def _dead_reckoning_to_geodetic(
        self,
        estimate: UAVState,
        geo_origin: Mapping[str, float],
    ) -> GeoPose:
        lat, lon, alt = enu_to_geodetic(
            estimate.position.east,
            estimate.position.north,
            -estimate.position.down,
            float(geo_origin.get("origin_latitude", 0.0)),
            float(geo_origin.get("origin_longitude", 0.0)),
            float(geo_origin.get("origin_altitude", 0.0)),
        )
        return float(lat), float(lon), float(alt)

    def _activate_failsafe(self) -> Tuple[GeoPose, str]:
        self._failsafe_active = True
        logger.error("No valid navigation sources available! Failsafe triggered.")
        fallback = self._last_estimate or self._last_gps or (0.0, 0.0, 0.0)
        return fallback, "FAILSAFE"

    @staticmethod
    def _horizontal_distance_m(lhs: GeoPose, rhs: GeoPose) -> float:
        # Approximate planar distance in metres. Longitude degrees are
        # scaled by cos(latitude); without it the east component is
        # overstated (~17% at QAU's ~33.7 deg latitude).
        dlat = lhs[0] - rhs[0]
        dlon = lhs[1] - rhs[1]
        mean_lat_rad = math.radians((lhs[0] + rhs[0]) / 2.0)
        dlon_scaled = dlon * math.cos(mean_lat_rad)
        return 111000.0 * math.sqrt(dlat * dlat + dlon_scaled * dlon_scaled)

    @property
    def failsafe_active(self) -> bool:
        return self._failsafe_active
