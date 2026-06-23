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

    def _set_state(self, new_state: GnssState, timestamp: float) -> GnssState:
        previous = self._current_state
        self._current_state = new_state
        if previous != new_state:
            logger.info(
                "GNSS state transition: %s -> %s",
                previous.value,
                new_state.value,
                extra={
                    "context": {
                        "previous_state": previous.value,
                        "new_state": new_state.value,
                        "timestamp": timestamp,
                    }
                },
            )
        return self._current_state

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
            return self._set_state(GnssState.DENIED, current_time)
            
        if num_satellites < self.degraded_satellites or hdop > self.degraded_hdop:
            return self._set_state(GnssState.DEGRADED, current_time)
        return self._set_state(GnssState.HEALTHY, current_time)

    def check_timeout(self, current_time: float = None) -> GnssState:
        """Check if GPS signal has dropped out based on update timeout."""
        now = current_time or time.time()
        if self._current_state != GnssState.DENIED and (now - self._last_update_time) > self.denied_timeout_seconds:
            logger.warning("GNSS update timeout! Declaring GNSS DENIED.")
            return self._set_state(GnssState.DENIED, now)
        return self._current_state

    @property
    def state(self) -> GnssState:
        return self._current_state

    @property
    def last_update_time(self) -> float:
        return self._last_update_time
