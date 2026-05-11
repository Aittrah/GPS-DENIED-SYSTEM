"""
MAVLink interface for flight controller communication using MAVSDK.

Supports serial and UDP connections with automatic reconnection on failure.
"""

import asyncio
import contextlib
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

try:
    from mavsdk import System
    MAVSDK_AVAILABLE = True
except ImportError:
    MAVSDK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GpsStatus:
    """GPS status snapshot from the flight controller."""

    fix_type: int = 0
    """0=no GPS, 1=no fix, 2=2D fix, 3=3D fix, 6=RTK float, 7=RTK fixed"""

    num_satellites: int = 0
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    absolute_altitude_m: float = 0.0
    hdop: float = 99.9
    timestamp: float = field(default_factory=time.time)

    @property
    def has_fix(self) -> bool:
        return self.fix_type >= 2

    @property
    def is_healthy(self) -> bool:
        """True when signal quality meets normal-operations thresholds."""
        return self.has_fix and self.num_satellites >= 4 and self.hdop <= 5.0


class MAVLinkInterface:
    """
    Flight-controller interface over MAVLink (MAVSDK backend).

    Usage::

        iface = MAVLinkInterface("udp://:14551")
        await iface.connect()
        await iface.subscribe_gps(lambda s: print(s))
        ...
        await iface.disconnect()

    Connection strings:
        UDP:    ``udp://:14551``  or  ``udp://192.168.1.1:14550``
        Serial: ``serial:///dev/ttyUSB0:57600``
    """

    def __init__(
        self,
        connection_string: str = "udp://:14551",
        system_id: int = 1,
        component_id: int = 196,
        max_reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
        connect_timeout: float = 10.0,
    ) -> None:
        self._connection_string = connection_string
        self._system_id = system_id
        self._component_id = component_id
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._connect_timeout = connect_timeout

        self._system: Optional["System"] = None
        self._connected = False
        self._shutdown = False
        self._reconnect_attempts = 0

        self._gps_callbacks: List[Callable[[GpsStatus], None]] = []
        self._gps_status = GpsStatus()
        self._gps_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """
        Connect to the flight controller.

        Raises:
            ImportError: if mavsdk is not installed.
            ConnectionError: if the connection or handshake fails.
        """
        if not MAVSDK_AVAILABLE:
            raise ImportError(
                "mavsdk is not installed. Run: pip install mavsdk"
            )
        self._shutdown = False
        self._reconnect_attempts = 0
        return await self._do_connect()

    async def subscribe_gps(
        self, callback: Callable[[GpsStatus], None]
    ) -> None:
        """
        Register *callback* to receive GPS status updates.

        The callback is called on every fix-info or position update.
        Multiple callbacks may be registered; each receives a shallow copy
        of the current GpsStatus.
        """
        self._gps_callbacks.append(callback)
        if self._gps_task is None or self._gps_task.done():
            self._gps_task = asyncio.create_task(
                self._gps_loop(), name="mavlink_gps_stream"
            )

    async def disconnect(self) -> None:
        """Cancel all background streams and release the connection."""
        self._shutdown = True
        self._connected = False
        if self._gps_task and not self._gps_task.done():
            self._gps_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._gps_task
        logger.info("MAVLink interface disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @classmethod
    def from_config(cls, config: dict) -> "MAVLinkInterface":
        """Construct from the ``mavlink`` section of simulation.yaml."""
        return cls(
            connection_string=config.get("connection_string", "udp://:14551"),
            system_id=config.get("system_id", 1),
            component_id=config.get("component_id", 196),
        )

    # ------------------------------------------------------------------
    # Connection internals
    # ------------------------------------------------------------------

    async def _do_connect(self) -> bool:
        """Open connection and wait for the heartbeat handshake."""
        self._system = System(
            sysid=self._system_id, compid=self._component_id
        )
        try:
            await self._system.connect(
                system_address=self._connection_string
            )
            logger.info(
                "Waiting for drone on %s (timeout=%.1fs)",
                self._connection_string,
                self._connect_timeout,
            )
            await asyncio.wait_for(
                self._wait_for_heartbeat(), timeout=self._connect_timeout
            )
            self._connected = True
            self._reconnect_attempts = 0
            logger.info("Connected to flight controller")
            return True
        except asyncio.TimeoutError:
            self._connected = False
            raise ConnectionError(
                f"Timed out connecting to {self._connection_string} "
                f"after {self._connect_timeout}s"
            )
        except Exception as exc:
            self._connected = False
            raise ConnectionError(
                f"Failed to connect to {self._connection_string}: {exc}"
            ) from exc

    async def _wait_for_heartbeat(self) -> None:
        async for state in self._system.core.connection_state():
            if state.is_connected:
                return

    # ------------------------------------------------------------------
    # GPS streaming
    # ------------------------------------------------------------------

    async def _gps_loop(self) -> None:
        """Outer loop: (re)start the three telemetry sub-streams."""
        while not self._shutdown:
            if not self._connected or self._system is None:
                await asyncio.sleep(0.1)
                continue
            tasks = [
                asyncio.create_task(self._stream_gps_info()),
                asyncio.create_task(self._stream_position()),
                asyncio.create_task(self._stream_raw_gps()),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            if self._shutdown:
                return
            for t in done:
                exc = t.exception()
                if exc:
                    logger.warning("GPS telemetry stream lost: %s", exc)
                    await self._handle_reconnect()
                    break

    async def _stream_gps_info(self) -> None:
        """Stream fix type and satellite count."""
        async for gps_info in self._system.telemetry.gps_info():
            if self._shutdown:
                return
            self._gps_status.fix_type = int(gps_info.fix_type.value)
            self._gps_status.num_satellites = gps_info.num_satellites
            self._gps_status.timestamp = time.time()
            self._notify_callbacks()

    async def _stream_position(self) -> None:
        """Stream lat/lon/altitude."""
        async for position in self._system.telemetry.position():
            if self._shutdown:
                return
            self._gps_status.latitude_deg = position.latitude_deg
            self._gps_status.longitude_deg = position.longitude_deg
            self._gps_status.absolute_altitude_m = position.absolute_altitude_m
            self._gps_status.timestamp = time.time()
            self._notify_callbacks()

    async def _stream_raw_gps(self) -> None:
        """Stream HDOP from raw GPS (best-effort; silently skipped if unavailable)."""
        try:
            async for raw in self._system.telemetry.raw_gps():
                if self._shutdown:
                    return
                self._gps_status.hdop = raw.hdop
                self._gps_status.timestamp = time.time()
                self._notify_callbacks()
        except Exception as exc:
            # raw_gps() is optional; log once and let this stream die quietly
            logger.debug("raw_gps stream unavailable, HDOP will not update: %s", exc)

    def _notify_callbacks(self) -> None:
        snapshot = copy.copy(self._gps_status)
        for cb in self._gps_callbacks:
            try:
                cb(snapshot)
            except Exception as exc:
                logger.error("GPS callback raised an exception: %s", exc)

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    async def _handle_reconnect(self) -> None:
        """Retry connection with exponential backoff."""
        self._connected = False
        unlimited = self._max_reconnect_attempts == 0
        attempt = 0
        while not self._shutdown:
            if not unlimited and attempt >= self._max_reconnect_attempts:
                logger.error(
                    "Giving up after %d reconnection attempts",
                    self._max_reconnect_attempts,
                )
                return
            delay = self._reconnect_delay * (2 ** min(attempt, 6))
            limit_str = (
                "unlimited" if unlimited
                else str(self._max_reconnect_attempts)
            )
            logger.info(
                "Reconnecting in %.1fs (attempt %d/%s)...",
                delay, attempt + 1, limit_str,
            )
            await asyncio.sleep(delay)
            try:
                await self._do_connect()
                return
            except ConnectionError as exc:
                logger.warning("Reconnect attempt %d failed: %s", attempt + 1, exc)
                attempt += 1
