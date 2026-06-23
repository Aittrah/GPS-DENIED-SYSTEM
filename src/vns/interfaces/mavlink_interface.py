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


@dataclass
class VehicleStatus:
    """Best-effort status snapshot of the connected flight controller."""

    connected: bool = False
    armed: bool = False
    in_air: bool = False
    flight_mode: str = "UNKNOWN"
    mission_current: int = 0
    mission_total: int = 0
    gps_status: GpsStatus = field(default_factory=GpsStatus)
    last_vision_update_usec: int = 0
    last_error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


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
        self._reconnect_lock = asyncio.Lock()

        self._gps_callbacks: List[Callable[[GpsStatus], None]] = []
        self._status_callbacks: List[Callable[[VehicleStatus], None]] = []
        self._gps_status = GpsStatus()
        self._status = VehicleStatus()
        self._gps_task: Optional[asyncio.Task] = None
        self._status_task: Optional[asyncio.Task] = None

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
        connected = await self._do_connect()
        self._ensure_background_tasks()
        return connected

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
        self._ensure_background_tasks()

    async def subscribe_status(
        self, callback: Callable[[VehicleStatus], None]
    ) -> None:
        """Register *callback* to receive vehicle status updates."""
        self._status_callbacks.append(callback)
        self._ensure_background_tasks()

    async def disconnect(self) -> None:
        """Cancel all background streams and release the connection."""
        self._shutdown = True
        self._connected = False
        if self._gps_task and not self._gps_task.done():
            self._gps_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._gps_task
        if self._status_task and not self._status_task.done():
            self._status_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._status_task
        self._status.connected = False
        self._status.timestamp = time.time()
        logger.info("MAVLink interface disconnected")

    async def send_vision_position_estimate(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        yaw_rad: float = 0.0,
        time_usec: int = 0,
    ) -> bool:
        """
        Send VISION_POSITION_ESTIMATE to PX4 EKF2.

        Coordinates are local NED (North-East-Down) relative to EKF2 origin.
        PX4 fuses this when EKF2_EV_CTRL is set appropriately.

        Returns True on success, False if not connected.
        """
        if not self._connected or self._system is None:
            logger.warning("Cannot send vision estimate: not connected")
            return False
        try:
            from mavsdk.mocap import VisionPositionEstimate, PositionBody, AngleBody, Covariance

            position = PositionBody(x_m, y_m, z_m)
            angle = AngleBody(roll_rad, pitch_rad, yaw_rad)
            covariance = Covariance([float('nan')] * 21)
            estimate = VisionPositionEstimate(
                time_usec=time_usec,
                position_body=position,
                angle_body=angle,
                pose_covariance=covariance,
            )
            await self._system.mocap.set_vision_position_estimate(estimate)
            self._status.last_vision_update_usec = time_usec
            self._status.last_error = None
            self._status.timestamp = time.time()
            self._notify_status_callbacks()
            return True
        except Exception as exc:
            self._status.last_error = str(exc)
            self._status.timestamp = time.time()
            self._notify_status_callbacks()
            logger.warning("Failed to send vision position estimate: %s", exc)
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def status(self) -> VehicleStatus:
        snapshot = copy.deepcopy(self._status)
        snapshot.gps_status = copy.copy(self._gps_status)
        return snapshot

    @classmethod
    def from_config(cls, config: dict) -> "MAVLinkInterface":
        """Construct from the ``mavlink`` section of simulation.yaml."""
        return cls(
            connection_string=config.get("connection_string", "udp://:14551"),
            system_id=config.get("system_id", 1),
            component_id=config.get("component_id", 196),
            max_reconnect_attempts=config.get("max_reconnect_attempts", 5),
            reconnect_delay=config.get("reconnect_delay", 2.0),
            connect_timeout=config.get("connect_timeout", 10.0),
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
            self._status.connected = True
            self._status.last_error = None
            self._status.timestamp = time.time()
            self._notify_status_callbacks()
            logger.info("Connected to flight controller")
            return True
        except asyncio.TimeoutError:
            self._connected = False
            self._status.connected = False
            self._status.last_error = (
                f"Timed out connecting to {self._connection_string}"
            )
            self._status.timestamp = time.time()
            raise ConnectionError(
                f"Timed out connecting to {self._connection_string} "
                f"after {self._connect_timeout}s"
            )
        except Exception as exc:
            self._connected = False
            self._status.connected = False
            self._status.last_error = str(exc)
            self._status.timestamp = time.time()
            raise ConnectionError(
                f"Failed to connect to {self._connection_string}: {exc}"
            ) from exc

    def _ensure_background_tasks(self) -> None:
        if self._gps_task is None or self._gps_task.done():
            self._gps_task = asyncio.create_task(
                self._gps_loop(), name="mavlink_gps_stream"
            )
        if self._status_task is None or self._status_task.done():
            self._status_task = asyncio.create_task(
                self._status_loop(), name="mavlink_status_stream"
            )

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

    async def _status_loop(self) -> None:
        """Outer loop: keep a best-effort controller status snapshot updated."""
        while not self._shutdown:
            if not self._connected or self._system is None:
                await asyncio.sleep(0.1)
                continue
            tasks = [
                asyncio.create_task(self._stream_armed()),
                asyncio.create_task(self._stream_in_air()),
                asyncio.create_task(self._stream_flight_mode()),
                asyncio.create_task(self._stream_mission_progress()),
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
                    logger.warning("MAVLink status stream lost: %s", exc)
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
            self._notify_gps_callbacks()

    async def _stream_position(self) -> None:
        """Stream lat/lon/altitude."""
        async for position in self._system.telemetry.position():
            if self._shutdown:
                return
            self._gps_status.latitude_deg = position.latitude_deg
            self._gps_status.longitude_deg = position.longitude_deg
            self._gps_status.absolute_altitude_m = position.absolute_altitude_m
            self._gps_status.timestamp = time.time()
            self._notify_gps_callbacks()

    async def _stream_raw_gps(self) -> None:
        """Stream HDOP from raw GPS (best-effort; silently skipped if unavailable)."""
        try:
            async for raw in self._system.telemetry.raw_gps():
                if self._shutdown:
                    return
                self._gps_status.hdop = raw.hdop
                self._gps_status.timestamp = time.time()
                self._notify_gps_callbacks()
        except Exception as exc:
            # raw_gps() is optional; log once and let this stream die quietly
            logger.debug("raw_gps stream unavailable, HDOP will not update: %s", exc)

    async def _stream_armed(self) -> None:
        async for armed in self._system.telemetry.armed():
            if self._shutdown:
                return
            self._status.armed = bool(armed)
            self._status.timestamp = time.time()
            self._notify_status_callbacks()

    async def _stream_in_air(self) -> None:
        async for in_air in self._system.telemetry.in_air():
            if self._shutdown:
                return
            self._status.in_air = bool(in_air)
            self._status.timestamp = time.time()
            self._notify_status_callbacks()

    async def _stream_flight_mode(self) -> None:
        async for flight_mode in self._system.telemetry.flight_mode():
            if self._shutdown:
                return
            value = getattr(flight_mode, "name", str(flight_mode))
            self._status.flight_mode = str(value)
            self._status.timestamp = time.time()
            self._notify_status_callbacks()

    async def _stream_mission_progress(self) -> None:
        try:
            async for progress in self._system.mission.mission_progress():
                if self._shutdown:
                    return
                self._status.mission_current = int(progress.current)
                self._status.mission_total = int(progress.total)
                self._status.timestamp = time.time()
                self._notify_status_callbacks()
        except Exception as exc:
            logger.debug("mission_progress stream unavailable: %s", exc)

    def _notify_gps_callbacks(self) -> None:
        self._status.gps_status = copy.copy(self._gps_status)
        self._status.timestamp = time.time()
        snapshot = copy.copy(self._gps_status)
        for cb in self._gps_callbacks:
            try:
                cb(snapshot)
            except Exception as exc:
                logger.error("GPS callback raised an exception: %s", exc)
        self._notify_status_callbacks()

    def _notify_status_callbacks(self) -> None:
        snapshot = self.status
        for cb in self._status_callbacks:
            try:
                cb(snapshot)
            except Exception as exc:
                logger.error("Vehicle status callback raised an exception: %s", exc)

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    async def _handle_reconnect(self) -> None:
        """Retry connection with exponential backoff."""
        async with self._reconnect_lock:
            if self._shutdown or self._connected:
                return

            self._connected = False
            self._status.connected = False
            self._status.timestamp = time.time()
            self._notify_status_callbacks()

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
                    self._status.last_error = str(exc)
                    self._status.timestamp = time.time()
                    self._notify_status_callbacks()
                    logger.warning(
                        "Reconnect attempt %d failed: %s",
                        attempt + 1,
                        exc,
                    )
                    attempt += 1
