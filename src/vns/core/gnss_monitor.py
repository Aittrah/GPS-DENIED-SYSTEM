import logging
import time
from enum import Enum

logger = logging.getLogger("vns.core")

class GnssState(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DENIED = "DENIED"

class GnssMonitor:
    """Monitors incoming GNSS signal quality and detects dropouts or degradation."""

    def __init__(
        self,
        degraded_hdop: float = 5.0,
        degraded_satellites: int = 4,
        denied_timeout_seconds: float = 2.0
    ) -> None:
        self.degraded_hdop = degraded_hdop
        self.degraded_satellites = degraded_satellites
        self.denied_timeout_seconds = denied_timeout_seconds
        
        self._last_update_time = 0.0
        self._current_state = GnssState.DENIED

    def update(
        self,
        has_fix: bool,
        num_satellites: int,
        hdop: float,
        timestamp: float = None
    ) -> GnssState:
        """Update GNSS status and return the current state classification."""
        current_time = timestamp or time.time()
        self._last_update_time = current_time
        
        if not has_fix:
            self._current_state = GnssState.DENIED
            return self._current_state
            
        if num_satellites < self.degraded_satellites or hdop > self.degraded_hdop:
            self._current_state = GnssState.DEGRADED
        else:
            self._current_state = GnssState.HEALTHY
            
        return self._current_state

    def check_timeout(self, current_time: float = None) -> GnssState:
        """Check if GPS signal has dropped out based on update timeout."""
        now = current_time or time.time()
        if self._current_state != GnssState.DENIED and (now - self._last_update_time) > self.denied_timeout_seconds:
            logger.warning("GNSS update timeout! Declaring GNSS DENIED.")
            self._current_state = GnssState.DENIED
        return self._current_state

    @property
    def state(self) -> GnssState:
        return self._current_state
