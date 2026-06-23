from vns.interfaces.mavlink_interface import GpsStatus, MAVLinkInterface


def test_status_returns_isolated_snapshot() -> None:
    interface = MAVLinkInterface()
    interface._gps_status = GpsStatus(
        fix_type=3,
        num_satellites=11,
        latitude_deg=33.7470,
        longitude_deg=73.1370,
        absolute_altitude_m=580.0,
        hdop=0.9,
    )
    interface._status.connected = True
    interface._status.flight_mode = "MISSION"

    snapshot = interface.status
    snapshot.connected = False
    snapshot.gps_status.fix_type = 0

    fresh_snapshot = interface.status
    assert fresh_snapshot.connected is True
    assert fresh_snapshot.flight_mode == "MISSION"
    assert fresh_snapshot.gps_status.fix_type == 3
    assert fresh_snapshot.gps_status.has_fix is True


def test_from_config_applies_retry_settings() -> None:
    interface = MAVLinkInterface.from_config(
        {
            "connection_string": "serial:///dev/ttyUSB0:57600",
            "system_id": 2,
            "component_id": 197,
            "max_reconnect_attempts": 0,
            "reconnect_delay": 1.5,
            "connect_timeout": 4.0,
        }
    )

    assert interface._connection_string == "serial:///dev/ttyUSB0:57600"
    assert interface._system_id == 2
    assert interface._component_id == 197
    assert interface._max_reconnect_attempts == 0
    assert interface._reconnect_delay == 1.5
    assert interface._connect_timeout == 4.0
