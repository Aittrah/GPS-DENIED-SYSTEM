import logging
import math
import time
from typing import Optional, Tuple

from vns.core.gnss_monitor import GnssState

logger = logging.getLogger("vns.core")

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
        gps_pos: Optional[Tuple[float, float, float]],
        visual_pos: Optional[Tuple[float, float, float]],
        gnss_state: GnssState,
        current_time: Optional[float] = None
    ) -> Tuple[Tuple[float, float, float], str]:
        """
        Blend GPS and Visual positions based on GNSS health state.
        Returns:
            blended_position: (lat, lon, alt)
            mode: "GPS", "BLENDING", "VISION", or "FAILSAFE"
        """
        now = time.time() if current_time is None else current_time
        
        # Track last inputs
        if gps_pos is not None:
            self._last_gps = gps_pos
            
        if visual_pos is not None:
            # Position continuity check (outlier rejection)
            if self._last_estimate is not None:
                # Approximate planar distance in metres. Longitude degrees are
                # scaled by cos(latitude); without it the east component is
                # overstated (~17% at QAU's ~33.7 deg latitude), which would
                # reject valid visual fixes.
                dlat = visual_pos[0] - self._last_estimate[0]
                dlon = visual_pos[1] - self._last_estimate[1]
                mean_lat_rad = math.radians(
                    (visual_pos[0] + self._last_estimate[0]) / 2.0
                )
                dlon_scaled = dlon * math.cos(mean_lat_rad)
                dist_m = 111000.0 * (dlat**2 + dlon_scaled**2)**0.5
                if dist_m > self.position_continuity_threshold:
                    logger.warning("Visual position rejected due to discontinuity: %.2fm", dist_m)
                    visual_pos = None
            
            if visual_pos is not None:
                self._last_visual = visual_pos
                self._last_visual_time = now

        # Evaluate failsafe
        time_since_visual = now - self._last_visual_time
        if gnss_state == GnssState.DENIED and (self._last_visual is None or time_since_visual > self.visual_timeout_seconds):
            self._failsafe_active = True
            logger.error("No valid navigation sources available! Failsafe triggered.")
            # Fallback to last estimate or last GPS or (0,0,0)
            fallback = self._last_estimate or self._last_gps or (0.0, 0.0, 0.0)
            return fallback, "FAILSAFE"

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

    @property
    def failsafe_active(self) -> bool:
        return self._failsafe_active
